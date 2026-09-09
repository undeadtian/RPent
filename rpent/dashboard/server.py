"""Dashboard 实时监控页面的 HTTP 服务。

该模块是浏览器前端与 :class:`~rpent.dashboard.state.DashboardState` 之间的
传输边界，主要负责：

- 提供 Dashboard HTML、JavaScript、CSS 等静态资源；
- 在 Session 真正启动前完成浏览器启动页与 CLI 主线程之间的配置握手；
- 暴露任务状态、推理记录、实时画面和视频等只读接口；
- 接收用户追加消息、撤回排队消息和中断智能体等控制请求；
- 通过 Server-Sent Events（SSE）持续向浏览器推送状态快照。

FastAPI/uvicorn 在守护线程中运行，因此不会阻塞智能体主循环。服务会在任务结束后
继续存在，便于用户查看最终状态和回放；只有宿主进程结束时才会随守护线程退出。
这里的路由名称和响应结构是 ``rpent/dashboard/index.html`` 及其 JavaScript 的固定
前后端协议，修改时需要同步检查前端调用。
"""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from pathlib import Path
from typing import Any

import uvicorn
from fastapi import Body, FastAPI
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    JSONResponse,
    Response,
    StreamingResponse,
)
from fastapi.staticfiles import StaticFiles

from rpent.dashboard.interaction import (
    DashboardMessageConflictError,
    InteractionUnavailableError,
    UnknownDashboardMessageError,
)
from rpent.dashboard.state import DashboardState


class DashboardServer:
    """在线程内运行的 Dashboard FastAPI 服务。

    一个实例只绑定一个长生命周期 Dashboard Session。构造时会创建 FastAPI
    应用和全部路由，但不会监听端口；调用 :meth:`start` 后才启动 uvicorn。
    随后通过 :meth:`register` 绑定唯一的 :class:`DashboardState`，所有运行时
    API 都从该状态对象读取数据或提交控制请求。

    Attributes:
        host: uvicorn 监听地址。默认仅监听本机，避免无意暴露机器人控制接口。
        port: 请求或实际使用的 TCP 端口；传入 ``0`` 表示自动分配空闲端口。
        runs_dir: 前端显示的运行目录。未显式指定时，从注册状态的输出目录推导。
    """

    def __init__(
        self,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        runs_dir: str = "",
        language: str = "en",
        dashboard_spec: dict[str, Any],
    ) -> None:
        """构建尚未监听端口的 Dashboard 服务。

        Args:
            host: HTTP 服务监听地址。
            port: HTTP 端口；``0`` 表示启动时向操作系统申请空闲端口。
            runs_dir: 可选的运行目录展示值。
            language: 前端语言，目前 ``zh-cn`` 使用简体中文，其余值回退英文。
            dashboard_spec: 环境提供的前端能力描述，包括任务参数、运行时组件、
                相机画面频道和输入模式等。
        """
        self.host = host
        self.port = int(port)
        self.runs_dir = runs_dir
        self._dashboard_spec = dashboard_spec

        # HTML 在构造时一次性读入内存，后续请求只做语言占位符替换，避免每次刷新
        # 都访问磁盘。静态目录则交给 Starlette StaticFiles 高效处理。
        dashboard_dir = Path(__file__).parent
        self._language = "zh-cn" if language == "zh-cn" else "en"
        self._index_html = (dashboard_dir / "index.html").read_text(encoding="utf-8")
        self._static_dir = dashboard_dir / "static"

        # 服务启动和 Session 状态注册是两个独立阶段：启动页必须先可访问，用户确认
        # 配置后 CLI 才创建 DashboardState 及昂贵的 VLA/SAM3 共享运行时。
        self._state: DashboardState | None = None
        self._app = self._build_app()
        self._server: uvicorn.Server | None = None

        # 启动页握手状态。CLI 主线程在 wait_for_launch() 中等待 Event；FastAPI
        # 请求线程在 POST /api/launch/run 中写入最终配置并 set() 唤醒主线程。
        self._launch_enabled = False
        self._launch_defaults: dict[str, Any] = {}
        self._launch_config: dict[str, Any] | None = None
        self._launch_event = threading.Event()

    def register(self, state: DashboardState) -> None:
        """将唯一的 Session 状态注册到 HTTP 服务。

        重复注册同一个对象是幂等操作；注册不同状态会抛错，以免同一组路由意外
        读写另一个 Session。若未显式配置 ``runs_dir``，默认展示 Session 输出
        目录的父目录。

        Args:
            state: 供 REST/SSE 路由访问的长生命周期 Session 状态。

        Raises:
            ValueError: 当前服务已经绑定另一个状态对象。
        """
        if self._state is not None and self._state is not state:
            raise ValueError("Dashboard Session is already registered")
        self._state = state
        if not self.runs_dir:
            self.runs_dir = str(state.output_dir.parent)

    def _resolve(self, state_id: str) -> DashboardState | None:
        """按公开 run/session ID 查找当前状态。

        当前实现一个服务只承载一个 Session，因此这里不是通用状态表，而是严格的
        ID 校验边界。路由不能仅因“存在状态”就返回数据，否则陈旧浏览器标签页可能
        使用旧 ID 对新 Session 发出控制请求。
        """
        state = self._state
        if state is None or state.run_id != state_id:
            return None
        return state

    def wait_for_launch(self, defaults: dict[str, Any]) -> dict[str, Any]:
        """启用启动页，并阻塞到用户提交 Session 配置。

        ``defaults`` 通常来自当前 CLI 参数，会由 ``GET /api/launch/state`` 返回给
        浏览器填充表单。用户提交后，``POST /api/launch/run`` 合并默认值与覆盖值、
        触发线程事件，本方法随即返回最终配置。

        Args:
            defaults: 启动页表单的默认 Session 配置。

        Returns:
            用户确认后的配置字典；理论上的空结果会安全回退为空字典。

        Note:
            这是有意的同步阻塞点，应由 CLI 主线程调用，不能在 FastAPI 事件循环
            中调用，否则会造成请求死锁。
        """
        # 复制调用者字典，防止等待期间外部修改导致前端看到不一致默认值。
        self._launch_defaults = dict(defaults)
        self._launch_enabled = True
        self._launch_event.wait()
        return self._launch_config or {}

    def start(self) -> str:
        """在守护线程中启动 uvicorn，并返回浏览器访问地址。

        若构造时端口为 ``0``，先通过 :func:`_free_port` 选择空闲端口。方法最多
        等待 10 秒确认 uvicorn 已进入 started 状态，保证返回 URL 时服务可访问。

        Returns:
            形如 ``http://127.0.0.1:12345`` 的 Dashboard 根 URL。

        Raises:
            RuntimeError: uvicorn 在 10 秒内未成功启动。
        """
        port = self.port or _free_port(self.host)
        config = uvicorn.Config(
            self._app, host=self.host, port=port, log_level="warning"
        )
        self._server = uvicorn.Server(config)

        # daemon=True 表示宿主进程结束时无需额外 join HTTP 线程；Session 运行期间
        # self._server 持有服务对象，线程会持续处理请求。
        threading.Thread(target=self._server.run, daemon=True).start()

        # uvicorn 启动发生在另一个线程，轮询 started 可避免浏览器立即访问时遇到
        # connection refused。短 sleep 同时避免忙等占满 CPU。
        t0 = time.time()
        while not self._server.started and time.time() - t0 < 10:
            time.sleep(0.05)
        if not self._server.started:
            raise RuntimeError(f"dashboard server did not start on {self.host}:{port}")

        # 将自动分配的端口写回实例，确保日志和调用方拿到实际监听值。
        self.port = port
        return f"http://{self.host}:{port}"

    # -- FastAPI 路由 ------------------------------------------------------

    def _build_app(self) -> FastAPI:
        """创建 FastAPI 应用并注册固定的 Dashboard 前后端协议。"""
        app = FastAPI(title="RPent dashboard")

        # /static 只映射包内固定目录，浏览器从中加载 JS/CSS；任务产物不通过该
        # 目录暴露，而是由下方受状态对象约束的 frame/video 路由提供。
        app.mount(
            "/static",
            StaticFiles(directory=self._static_dir),
            name="dashboard-static",
        )

        @app.get("/")
        def index() -> HTMLResponse:
            """返回单页 Dashboard 入口。

            HTML 模板只保留一个语言占位符；服务端替换 ``<html lang>`` 后，前端
            JavaScript 会据此选择对应的静态文案和渲染方式。
            """
            html = self._index_html.replace(
                "__DASHBOARD_LANGUAGE__",
                self._language,
            )
            return HTMLResponse(html)

        @app.get("/healthz")
        def healthz() -> JSONResponse:
            """供进程管理器或人工检查 HTTP 服务是否可响应。"""
            # 该接口只代表 Web 服务存活，不代表 env/VLA/SAM3 已就绪；后者通过
            # DashboardState 的 runtime 状态展示。
            return JSONResponse({"ok": True})

        @app.get("/api/commands")
        def api_commands() -> JSONResponse:
            """返回环境定义的前端能力和任务命令规格。"""
            # 前端使用该结构动态生成任务输入、相机标签和运行时状态组件，因此无需
            # 为每一种机器人环境维护独立 HTML。
            return JSONResponse(self._dashboard_spec)

        # -- 启动页配置握手 -----------------------------------------------

        @app.get("/api/launch/state")
        def api_launch_state() -> JSONResponse:
            """返回启动页是否启用、是否仍待提交以及表单默认值。"""
            return JSONResponse(
                {
                    "enabled": self._launch_enabled,
                    # Event 一旦 set，说明某个请求已成功提交；前端应停止重复启动。
                    "pending": self._launch_enabled and not self._launch_event.is_set(),
                    "defaults": self._launch_defaults,
                }
            )

        @app.post("/api/launch/run")
        def api_launch_run(payload: dict[str, Any] = Body(default={})) -> JSONResponse:
            """接受启动页配置并唤醒等待中的 CLI 主线程。

            默认配置先展开，浏览器提交字段后展开，因此用户值具有覆盖优先级。
            Event 同时充当一次性闩锁：只允许第一个有效请求启动 Session。
            """
            if not self._launch_enabled:
                # CLI 尚未调用 wait_for_launch，启动器没有被“武装”。
                return JSONResponse({"error": "launcher not armed"}, status_code=409)
            if self._launch_event.is_set():
                # 防止双击按钮或多个浏览器标签页重复启动同一组共享服务。
                return JSONResponse({"error": "already launched"}, status_code=409)
            self._launch_config = {
                **self._launch_defaults,
                **payload,
            }
            # 先完整写入配置，再 set；Event 提供线程间 happens-before 保证，等待线程
            # 被唤醒后能看到完整的 _launch_config。
            self._launch_event.set()
            return JSONResponse({"ok": True})

        # -- Session 发现和交互控制 ---------------------------------------

        @app.get("/api/runs")
        def api_runs() -> JSONResponse:
            """列出本服务当前承载的 Session 摘要。

            虽然字段名为 ``runs``，当前架构每个 DashboardServer 只注册一个状态；
            注册前返回空列表，使启动页到实时监控页的轮询可以平滑过渡。
            """
            state = self._state
            return JSONResponse(
                {
                    "runs_dir": self.runs_dir,
                    "runs": [] if state is None else [state.run_info()],
                }
            )

        @app.post("/api/sessions/{session_id:path}/messages")
        def api_submit_message(
            session_id: str,
            payload: dict[str, Any] = Body(default={}),
        ) -> JSONResponse:
            """向活动智能体提交或排队一条用户消息。

            ``session_id:path`` 允许 ID 自身包含斜杠，例如
            ``dashboard-session/<timestamp>``。消息是否立即进入模型或等待下一个安全
            边界由 DashboardState/Planner 交互控制器决定。
            """
            live = self._resolve(session_id)
            if live is None:
                return JSONResponse({"error": "unknown session"}, status_code=404)
            try:
                live.submit_input(payload.get("text"))
            except ValueError as exc:
                # 422：请求 JSON 合法，但 text 缺失、为空或不符合消息约束。
                return JSONResponse({"error": str(exc)}, status_code=422)
            except InteractionUnavailableError as exc:
                # 409：Session 存在，但当前阶段尚不能接收消息或已经结束。
                return JSONResponse({"error": str(exc)}, status_code=409)
            # 202 表示消息已接受；智能体执行是异步的，并不在本 HTTP 请求内完成。
            return JSONResponse({"ok": True}, status_code=202)

        @app.delete(
            "/api/sessions/{session_id:path}/messages/{message_id}",
        )
        def api_withdraw_message(
            session_id: str,
            message_id: str,
        ) -> JSONResponse:
            """撤回一条尚未开始处理的排队消息。"""
            live = self._resolve(session_id)
            if live is None:
                return JSONResponse({"error": "unknown session"}, status_code=404)
            try:
                live.withdraw_message(message_id)
            except UnknownDashboardMessageError as exc:
                # Session 正确但消息 ID 不存在，按资源不存在返回 404。
                return JSONResponse({"error": str(exc)}, status_code=404)
            except (DashboardMessageConflictError, InteractionUnavailableError) as exc:
                # 消息已开始执行、已撤回，或交互控制器不可用时不能再改变其状态。
                return JSONResponse({"error": str(exc)}, status_code=409)
            return JSONResponse({"ok": True})

        @app.post("/api/sessions/{session_id:path}/interrupt")
        def api_interrupt(session_id: str) -> JSONResponse:
            """请求在下一个安全边界中断当前智能体工作。

            中断不是强杀线程：DashboardState 将请求转交 Planner 和 Toolkit，活动工具
            在可安全停止的位置取消，避免留下半写文件或失控的机器人动作。
            """
            live = self._resolve(session_id)
            if live is None:
                return JSONResponse({"error": "unknown session"}, status_code=404)
            try:
                result = live.request_interrupt()
            except InteractionUnavailableError as exc:
                return JSONResponse({"error": str(exc)}, status_code=409)
            if result == "noop":
                # 当前没有活动或排队工作，无需异步等待，直接返回 200。
                return JSONResponse(
                    {
                        "status": "noop",
                        "interrupt_requested": False,
                    }
                )
            # requested 和 duplicate 都表示已有异步中断待处理，故返回 202。重复请求
            # 明确标记 deduplicated，前端无需继续发送相同操作。
            return JSONResponse(
                {
                    "status": "requested",
                    "interrupt_requested": True,
                    "deduplicated": result == "duplicate",
                },
                status_code=202,
            )

        # -- 运行状态、transcript 与媒体读取 -------------------------------

        @app.get("/api/run")
        def api_run(run: str) -> JSONResponse:
            """返回一个 Session 的完整前端状态投影。"""
            live = self._resolve(run)
            if live is None:
                return JSONResponse({"error": "unknown run"}, status_code=404)
            return JSONResponse(live.run_detail())

        @app.get("/api/run/transcript")
        def api_transcript(run: str, since: int = 0) -> JSONResponse:
            """增量返回从 ``since`` 下标开始的推理与工具事件。

            前端记录已渲染事件数量，并在轮询/SSE 通知后只拉取新增部分，避免每次
            传输和重绘完整 transcript。未知 run 返回空事件数组，便于前端在状态
            注册前无错误地持续轮询。
            """
            live = self._resolve(run)
            events = live.events_since(since) if live else []
            return JSONResponse({"events": events})

        @app.get("/api/run/frame")
        def api_frame(
            run: str,
            kind: str = self._dashboard_spec["frame_channels"][0]["name"],
            t: str = "",
        ) -> Response:
            """返回指定实时画面频道的最新 PNG 帧。

            Args:
                run: Session ID。
                kind: ``DashboardSpec.frame_channels`` 中声明的频道名。
                t: 前端附加的缓存破坏参数；服务端无需读取其值。
            """
            live = self._resolve(run)
            try:
                png = live.frame(kind) if live else None
            except ValueError as exc:
                # run 存在但频道名不合法，返回可诊断的 422，而非模糊 404。
                return JSONResponse({"detail": str(exc)}, status_code=422)
            if png is None:
                # Session 尚未产生第一帧，或指定 run 不存在。
                return Response(status_code=404)
            return Response(png, media_type="image/png")

        @app.get("/api/run/video")
        def api_video(run: str) -> Response:
            """流式返回当前 TaskRun 的完整 MP4 回放。"""
            live = self._resolve(run)
            if live is None or not live.has_video():
                return Response(status_code=404)
            return FileResponse(
                # 路径由 DashboardState 从受控输出目录生成，而不是接受用户文件路径，
                # 因此不会形成任意文件读取接口。
                live.video_path,
                media_type="video/mp4",
                # 录像在运行中可能持续更新；禁止浏览器缓存陈旧版本。
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        @app.get("/api/run/action-video")
        def api_action_video(run: str, step: int) -> Response:
            """返回指定环境步骤对应的单动作 MP4 片段。"""
            live = self._resolve(run)
            path = live.action_video_path(step) if live else None
            if path is None:
                return Response(status_code=404)
            return FileResponse(
                path,
                media_type="video/mp4",
                headers={"Cache-Control": "no-store, max-age=0"},
            )

        # -- 实时状态推送 --------------------------------------------------

        @app.get("/api/stream")
        def api_stream(run: str) -> StreamingResponse:
            """建立 Server-Sent Events 连接并持续推送状态快照。

            SSE 是单向、基于普通 HTTP 的长连接，浏览器 ``EventSource`` 可自动重连。
            每 100 ms 获取一次轻量 snapshot；较大的 transcript、PNG 和 MP4 不放入
            SSE，而由上方独立接口按需读取，避免阻塞实时状态更新。
            """

            async def gen():
                # 每个浏览器连接拥有独立异步生成器。客户端断开后 StreamingResponse
                # 会取消生成器；sleep 将控制权交还事件循环，不会忙等占用线程。
                while True:
                    live = self._resolve(run)
                    if live is not None:
                        # SSE 每条消息以 data: 开头并以空行结束。snapshot 必须是 JSON
                        # 可序列化的前端状态投影。
                        yield f"data: {json.dumps(live.snapshot())}\n\n"
                    else:
                        # 注释行不会触发浏览器 message 事件，但可保持代理/连接存活，
                        # 同时允许前端在 state 注册前就建立 EventSource。
                        yield ": keepalive\n\n"
                    await asyncio.sleep(0.1)

            return StreamingResponse(gen(), media_type="text/event-stream")

        return app


def _free_port(host: str) -> int:
    """让操作系统为指定监听地址选择一个当前空闲的 TCP 端口。

    将临时 socket 绑定到端口 ``0`` 后，内核会分配临时端口；函数读取端口号并
    关闭 socket，随后 uvicorn 使用该端口。这里存在很小的“检查后使用”竞态窗口，
    若端口恰好被其他进程抢占，:meth:`DashboardServer.start` 会在超时后报告失败。

    Args:
        host: 端口需要可绑定的本地地址。

    Returns:
        操作系统分配的端口号。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])
