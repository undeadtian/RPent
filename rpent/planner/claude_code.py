"""Claude Agent SDK planner.

A thin, SDK-first backend for RPent. ``solve()`` does four things:
prepare output files, bind the in-process tool runtime, drive the SDK
query, and assemble a ``PlannerResult``. Event rendering and stats
collection live in a single observation layer (``_Recorder``) that has
no backend state of its own.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import json
import queue
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rpent.cli.tui import next_user_line
from rpent.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import (
    PlannerResult,
    add_mcp_prefix,
    strip_mcp_prefix,
)
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger, init_output_dir

logger = get_logger("claude")

# SDK 子进程/传输层允许缓存的最大消息字节数。机器人视觉工具可能返回 base64 图片，
# 默认缓冲过小会在工具结果尚未被 Recorder 消费前触发 SDK buffer overflow。
_MAX_STREAM_BUFFER_BYTES = 8 * 1024 * 1024

# =============================================================================
# Claude Agent SDK 后端数据流
# =============================================================================
#
#   CLI / Dashboard
#        │ Planner.solve(system_prompt, user_message, Toolkit)
#        ▼
#   ClaudeCodePlanner
#        ├─ 构造 ClaudeAgentOptions
#        ├─ 把 Toolkit 包装为进程内 MCP server
#        ├─ 选择非交互 / 终端 steering / Dashboard 会话
#        ▼
#   Claude Agent SDK client/query
#        │ SystemMessage / AssistantMessage / UserMessage / ResultMessage
#        ▼
#   _Recorder
#        ├─ 文本日志与 JSONL 原始流
#        ├─ Dashboard TranscriptEvent / UsageEvent
#        └─ finish、token、cost、turn、tool_calls
#
# 非交互模式直接消费 sdk.query()；终端和 Dashboard 则共享一个长生命周期
# _ClaudeSessionDriver。Driver 同时运行“SDK 消息消费者”和“外部命令泵”，任一结束后
# 取消另一方并关闭 client，保证每个 SDK session 始终只有一个消息消费者。
#
# 工具不经 HTTP：_build_rpent_server() 把 Toolkit 注册表转换为 SDK 进程内 MCP server。
# 模型看到 mcp__rpent__<name>，handler 最终仍在线程中调用同步 execute_tool()。
# =============================================================================

# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class ClaudeCodePlanner:
    """Planner backed by the Claude Agent SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        dashboard_events: DashboardEventSink,
        repo_root: str | Path | None = None,
        model: str = "sonnet",
        allowed_tools: str = "Bash Read Write Glob Grep",
        timeout_s: int = 600,
        max_budget_usd: float = 10.0,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
    ):
        """Initialize the Claude Agent SDK backend."""
        # SDK 运行期间的日志/artifact 工作目录；与 Claude 的 cwd（仓库根）用途不同。
        self._output_dir = str(output_dir)
        # cwd 限定 Bash/Read/Write 等内置工具的默认工作区。
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        self._model = model
        # 字符串在 _build_options 中拆成内置工具白名单；Toolkit MCP 工具会另外追加。
        self._allowed_tools = allowed_tools
        # 非交互和 Dashboard 的整体会话超时；终端人工监督模式故意不应用此上限。
        self._timeout_s = timeout_s
        # Claude SDK 原生美元预算，与 max_turns 一起限制失控会话成本。
        self._max_budget_usd = max_budget_usd
        # 允许 Agent 访问仓库之外的受控目录，例如 RPent 持久化环境记忆。
        self._extra_dirs = extra_dirs or []
        # 显式路径用于稳定落盘；未提供时 _solve_async 创建不会自动删除的临时文件。
        self._output_path = Path(output_path) if output_path else None
        # Planner 与 Recorder 通过统一 sink 发布 transcript/usage；普通 CLI 可使用 null sink。
        self._dashboard_events = dashboard_events

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Run a Claude Agent SDK session for the given prompt."""
        if input_queue is not None and dashboard_interaction is not None:
            # 两者分别拥有不同的输入确认和中断策略，不能同时控制同一个 SDK client。
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        # Claude SDK 的 query 只有一个 prompt 参数，因此 system 和首条 user 文本在入口
        # 合并；initial_user_text 仍单独传递，用于 Dashboard 展示而不泄露 system prompt。
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        # Planner 协议是同步接口；内部统一用 asyncio 驱动 SDK 和 MCP 工具。
        return asyncio.run(
            self._solve_async(
                prompt,
                initial_user_text=user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
                dashboard_interaction=dashboard_interaction,
            )
        )

    # -- internal lifecycle -------------------------------------------------

    async def _solve_async(
        self,
        prompt: str,
        *,
        initial_user_text: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        # 延迟导入可选 SDK，使未选择 claude_code 后端时无需安装/加载其依赖。
        import claude_agent_sdk

        sdk = claude_agent_sdk
        if self._output_path is None:
            # delete=False 让调用结束后 transcript 仍可由日志定位和人工排障。
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="claude_agent_task_", delete=False
            ) as f:
                output_path = Path(f.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        # 人类可读输出和完整 SDK JSONL 分离：前者可摘要，后者保留原始消息结构。
        raw_stream_path = output_path.with_suffix(output_path.suffix + ".stream.jsonl")
        recorder = _Recorder(
            max_turns=max_turns,
            dashboard_events=self._dashboard_events,
        )

        # 确保 TaskRun 输出目录存在，再把它加入 Claude Agent 可访问目录。
        init_output_dir(self._output_dir)
        options = self._build_options(sdk, toolkit=toolkit, max_turns=max_turns)

        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Claude Agent SDK model %s (timeout=%ds, budget=$%s)",
            self._model,
            self._timeout_s,
            self._max_budget_usd,
        )

        started = time.time()
        error: str | None = None
        rendered_chunks: list[str] = []
        # with 统一管理两个输出文件；任何 timeout/异常路径离开后都会 flush/close。
        with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:

            def _emit(message: Any) -> None:
                # 每条 SDK 消息先完整写 JSONL，再由 Recorder 生成可读文本和 Dashboard 事件。
                _write_jsonl(raw_f, _message_to_json(message))
                if rendered := recorder.observe(message):
                    rendered_chunks.append(rendered)
                    out_f.write(rendered)
                    # 长会话逐消息 flush，进程异常退出时也尽量保留已观察内容。
                    out_f.flush()
                    logger.info(rendered.rstrip())

            def _emit_user(line: str, *, initial_prompt: bool = False) -> None:
                # 初始合并 prompt 含 system 指令，不写入完整文本，避免日志重复或泄露；
                # 后续 steering 则记录用户原文。
                display_text = (
                    "[initial task instructions submitted]" if initial_prompt else line
                )
                rendered = f"\n[user] {display_text}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                logger.info(rendered.strip())
                payload: dict[str, Any] = {"type": "user", "text": line}
                if initial_prompt:
                    # 前端已有任务描述来源，只需要 signal，不重复携带合并后的 prompt。
                    payload = {"type": "initial_prompt"}
                self._dashboard_events.emit(TranscriptEvent(payload))

            try:
                if input_queue is None:
                    if dashboard_interaction is not None:
                        # Dashboard：长生命周期 client 由状态版本、消息 ACK、中断和任务替换
                        # 共同驱动；整体 timeout 防止无人值守会话永久占用资源。
                        await asyncio.wait_for(
                            self._run_dashboard_session(
                                sdk,
                                prompt,
                                options,
                                recorder,
                                toolkit=toolkit,
                                dashboard_interaction=dashboard_interaction,
                                initial_user_text=initial_user_text,
                                emit=_emit,
                                emit_user=_emit_user,
                            ),
                            timeout=self._timeout_s,
                        )
                    else:

                        async def consume_stream() -> None:
                            # 非交互批处理不需要持有 ClaudeSDKClient；sdk.query 提供一次性流。
                            async for message in sdk.query(
                                prompt=prompt, options=options
                            ):
                                _emit(message)
                                if recorder.finish_result is not None:
                                    # finish 工具结果成功后即可停止消费，不等待模型额外文字。
                                    logger.info(
                                        "FINISH called: %s", recorder.finish_result
                                    )
                                    break

                        await asyncio.wait_for(
                            consume_stream(), timeout=self._timeout_s
                        )
                else:
                    # 终端模式允许用户无限思考/输入，因此不套整体 wait_for；/quit 或 EOF
                    # 由 adapter 中断 client 并退出。
                    await self._run_interactive(
                        sdk,
                        prompt,
                        options,
                        recorder,
                        input_queue,
                        emit=_emit,
                        emit_user=_emit_user,
                    )
            except asyncio.TimeoutError:
                # wait_for 会取消正在运行的 Dashboard/stream 协程，其 finally 负责 client
                # 与 Toolkit 清理；这里将超时转为 PlannerResult.error 并同步落盘。
                error = f"Claude Agent SDK timed out after {self._timeout_s}s"
                rendered = f"\n[cc-planner] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
            except Exception as e:
                # transport、SDK 配置或 adapter 异常统一封装，保留此前已输出的消息。
                error = f"{type(e).__name__}: {e}"
                rendered = f"\n[cc-planner] {error}\n"
                rendered_chunks.append(rendered)
                out_f.write(rendered)
                out_f.flush()
                _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())

        elapsed = time.time() - started
        # 正常路径直接拼接内存 chunks；极早阶段没有 rendered 时回读文件作为兜底。
        text = "".join(rendered_chunks) or output_path.read_text(errors="replace")
        # 外层 transport/timeout 错误优先于 ResultMessage 报告的 SDK 业务错误。
        error = error or recorder.error

        logger.info("Claude Agent SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            # 只有 finish 工具调用及其结果成功配对后才设置，不以 ResultMessage 冒充 finish。
            finish_result=recorder.finish_result,
            # 统一 Planner 协议只需可序列化消息；原始逐条 SDK 消息另存 JSONL。
            messages=[{"role": "claude_agent_sdk", "content": text}],
            stats={
                "backend": "claude_agent_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                **recorder.stats(),
            },
            error=error,
        )

    async def _run_interactive(
        self,
        sdk: Any,
        prompt: str,
        options: Any,
        recorder: "_Recorder",
        input_queue,
        *,
        emit,
        emit_user,
    ) -> None:
        """Drive a stateful ``ClaudeSDKClient`` with live steering.

        The opening prompt is sent, then the agent streams autonomously while a
        background pump forwards each user-typed line into the same session via
        ``client.query`` so it steers the agent at its next turn. A quit/EOF
        sentinel (or ``/quit``) interrupts the run; the ``finish`` tool ends it
        normally. Because a human supervises, there is no wall-clock cap here.
        """
        # Adapter 只定义终端输入策略；Driver 统一拥有 SDK client 和唯一消息消费者。
        adapter = _TerminalSessionAdapter(input_queue=input_queue, emit_user=emit_user)
        driver = _ClaudeSessionDriver(
            sdk=sdk,
            options=options,
            recorder=recorder,
            emit=emit,
        )
        await driver.run(prompt, adapter)

    async def _run_dashboard_session(
        self,
        sdk: Any,
        prompt: str,
        options: Any,
        recorder: "_Recorder",
        *,
        toolkit: Toolkit,
        dashboard_interaction: DashboardInteractionPort,
        initial_user_text: str,
        emit,
        emit_user,
    ) -> None:
        """Drive the Dashboard-owned long-lived Claude session."""
        # Adapter 将 SDK 生命周期边界翻译为共享 DashboardPlannerControl 的消息 ACK、
        # interrupt 和 task replacement 操作；Toolkit 取消函数用于安全停止环境动作。
        adapter = _ClaudeDashboardAdapter(
            interaction=dashboard_interaction,
            cancel_active_and_wait=toolkit.cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=lambda: emit_user(
                # Dashboard 展示原始 user_message，不展示与 system prompt 合并后的 SDK prompt。
                initial_user_text,
                initial_prompt=True,
            ),
        )
        driver = _ClaudeSessionDriver(
            sdk=sdk,
            options=options,
            recorder=recorder,
            emit=emit,
        )
        await driver.run(prompt, adapter)

    # -- options + tool bridge ---------------------------------------------

    def _build_options(self, sdk: Any, *, toolkit: Toolkit, max_turns: int) -> Any:
        # 同时接受空格和逗号分隔的 CLI 配置，并丢弃空项。
        allowed = [
            part for part in self._allowed_tools.replace(",", " ").split() if part
        ]
        # 不带 MCP `__` 命名空间的项目才可作为 SDK 内置 tools；MCP 工具由 server 提供。
        builtins = [name for name in allowed if "__" not in name]
        # Toolkit 内部短名在 SDK 权限白名单中必须使用 mcp__rpent__<name>。
        allowed.extend(
            add_mcp_prefix(str(spec["name"])) for spec in toolkit.get_tools_spec()
        )

        return sdk.ClaudeAgentOptions(
            cwd=self._repo_root,
            model=self._model,
            # SDK 自身限制模型 turn 和美元开销，独立于外层 wall-clock timeout。
            max_turns=max_turns,
            max_budget_usd=self._max_budget_usd,
            max_buffer_size=_MAX_STREAM_BUFFER_BYTES,
            tools=builtins or None,
            # dict 保序去重，避免用户配置与自动追加 MCP 工具产生重复项。
            allowed_tools=list(dict.fromkeys(allowed)),
            mcp_servers={
                # 进程内 MCP server，不开放端口，也不启动额外 daemon。
                "rpent": _build_rpent_server(
                    sdk,
                    toolkit=toolkit,
                ),
            },
            # 输出目录和显式 extra_dirs 是 repo cwd 外唯一授权访问区域。
            add_dirs=[self._output_dir, *self._extra_dirs],
            # Ignore user/project .claude configuration; RPent owns the loop.
            setting_sources=[],
            # SDK stderr 进入 debug 日志，不混入 PlannerResult transcript。
            stderr=lambda line: logger.debug("[claude-sdk] %s", line.rstrip()),
        )


# ---------------------------------------------------------------------------
# Shared long-lived session driver
# ---------------------------------------------------------------------------


class _ClaudeSessionDriver:
    """Own one SDK client and its single message consumer."""

    def __init__(
        self,
        *,
        sdk: Any,
        options: Any,
        recorder: "_Recorder",
        emit: Callable[[Any], None],
    ) -> None:
        # SDK 模块与 options 由 Planner 注入，便于保持本 Driver 不直接依赖可选包类型。
        self._sdk = sdk
        self._options = options
        # Recorder 跨多个 query 共享统计和 finish/error 状态。
        self._recorder = recorder
        # emit 同时负责 JSONL、可读日志和 Recorder.observe。
        self._emit = emit
        # 仅在 async client context 内非空；所有 query/interrupt 都检查连接状态。
        self._client: Any | None = None

    async def run(self, prompt: str, adapter: Any) -> None:
        """Open one client, submit the first query, and run one input adapter."""
        # tasks 在 client 建立前保持空，确保初始 query 失败时 finally 仍可安全清理。
        tasks: list[asyncio.Task[Any]] = []
        async with self._sdk.ClaudeSDKClient(options=self._options) as client:
            self._client = client
            try:
                # Submit before starting the consumer, matching the SDK's
                # streaming-client contract while its transport buffers output.
                await self.query(prompt)
                # Dashboard 在初始 query 成功后才开放输入；终端 adapter 此处为空操作。
                await adapter.initial_query_succeeded(self)

                # consumer 是唯一 SDK 消息读取者；command_pump 只处理外部输入/控制请求。
                consumer = asyncio.create_task(self._consume(adapter))
                command_pump = asyncio.create_task(adapter.run(self))
                tasks = [consumer, command_pump]
                done, _ = await asyncio.wait(
                    tasks,
                    # finish/SDK 流结束或外部 adapter 退出，任一都应终止整个 session。
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in done:
                    # 读取已完成 task 的异常，使其进入 _solve_async 统一错误处理。
                    await task
            finally:
                # Adapter cleanup first wakes terminal queue reads or Dashboard
                # condition waits. Then cancel and drain both long-lived tasks
                # before the SDK client context closes.
                with contextlib.suppress(Exception):
                    # Dashboard adapter 先取消 Toolkit 再封存交互；终端 adapter 投递 EOF。
                    await adapter.close()
                for task in tasks:
                    task.cancel()
                for task in tasks:
                    # 必须 await 已 cancel task，避免悬挂协程在 client 关闭后继续访问 transport。
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
                # 离开 client context 前先清空公开引用，阻止迟到调用继续 query/interrupt。
                self._client = None

    async def query(self, text: str) -> None:
        """Submit one user turn to the owned client."""
        if self._client is None:
            raise RuntimeError("Claude session is not connected")
        # query 只提交输入；响应始终由唯一 _consume task 读取，避免多消费者竞态。
        await self._client.query(text)

    async def submit(self, text: str) -> int:
        """Submit Dashboard input as a new Claude query."""
        await self.query(text)
        # DashboardPlannerControl 用返回值登记一个尚待 ResultMessage 结算的 completion。
        return 1

    async def interrupt(self) -> int:
        """Interrupt the owned client and suppress its expected error result."""
        if self._client is None:
            raise RuntimeError("Claude session is not connected")
        # SDK 把用户主动 interrupt 的当前 query 也表示为 is_error ResultMessage。提前设置
        # 标志让 Recorder 只忽略这一条预期错误，不把正常 Esc/任务替换记为 Planner 失败。
        self._recorder.suppress_next_result_error = True
        try:
            await self._client.interrupt()
        except BaseException:
            # A failed interrupt must not hide a later unrelated SDK error.
            # interrupt 命令自身失败时，后续 ResultMessage 可能是真实故障，必须恢复记录。
            self._recorder.suppress_next_result_error = False
            raise
        # Claude emits a ResultMessage for the interrupted query, so the
        # matching completion is accounted for by the normal message path.
        # 返回 0 表示 Control 此刻不立即扣 completion，稍后的 ResultMessage.complete 扣除。
        return 0

    async def _consume(self, adapter: Any) -> None:
        if self._client is None:
            raise RuntimeError("Claude session is not connected")
        async for message in self._client.receive_messages():
            # 先记录消息；Recorder 可能在工具结果中设置 finish_result。
            self._emit(message)
            # A successful finish tool result owns the boundary: end the
            # session without giving queued Dashboard input a chance to flush.
            if self._recorder.finish_result is not None:
                logger.info("FINISH called: %s", self._recorder.finish_result)
                return
            # Adapter 只观察已记录后的边界：ResultMessage 或 tool result 可触发 Dashboard flush。
            await adapter.on_message(self, message)


class _TerminalSessionAdapter:
    """Preserve the terminal TUI's interrupt-then-query steering policy."""

    def __init__(self, *, input_queue: Any, emit_user) -> None:
        # input_queue 由 TUI 线程安全地写入 str/None；emit_user 负责 transcript 展示。
        self._input_queue = input_queue
        self._emit_user = emit_user

    async def initial_query_succeeded(self, driver: _ClaudeSessionDriver) -> None:
        # 终端没有需开放的 Dashboard 输入状态，初始 query 成功后无需额外动作。
        return None

    async def run(self, driver: _ClaudeSessionDriver) -> None:
        while True:
            # next_user_line 内部阻塞 queue.get，移到线程以免卡住 SDK 消息 consumer。
            nxt = await asyncio.to_thread(next_user_line, self._input_queue)
            if nxt is None:
                # EOF、/quit 或 close 哨兵：尽力中断当前 query 后结束 command pump。
                with contextlib.suppress(Exception):
                    await driver.interrupt()
                return
            # 先记录用户 steering，再采用既有“中断当前 turn -> 同 session 新 query”策略。
            self._emit_user(nxt)
            # Keep the current terminal semantics: every steering line
            # interrupts the in-flight turn, then enters the same session.
            with contextlib.suppress(Exception):
                await driver.interrupt()
            with contextlib.suppress(Exception):
                await driver.query(nxt)

    async def on_message(
        self,
        driver: _ClaudeSessionDriver,
        message: Any,
    ) -> None:
        # 终端 steering 不需要按 SDK 消息边界更新 Dashboard completion 状态。
        return None

    async def close(self) -> None:
        # Unblock next_user_line() if the consumer (for example finish) won.
        # put(None) 让仍阻塞在线程中的 queue.get 退出，避免 Driver 清理悬挂。
        self._input_queue.put(None)


class _ClaudeDashboardAdapter:
    """Translate Claude SDK lifecycle events into shared control boundaries."""

    def __init__(
        self,
        *,
        interaction: DashboardInteractionPort,
        cancel_active_and_wait: Callable[[], None],
        emit_user: Callable[[str], None],
        emit_initial_user: Callable[[], None],
    ) -> None:
        # Claude adapter 不自行实现消息/中断状态机，复用 API/Codex 共用 Control。
        # defer_message_ack 默认 False：client.query 成功即视为消息已进入 SDK session。
        self._control = DashboardPlannerControl(
            interaction=interaction,
            cancel_active_and_wait=cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=emit_initial_user,
        )

    async def initial_query_succeeded(self, driver: _ClaudeSessionDriver) -> None:
        # 初始 SDK query 成功后才登记 completion、设为 busy 并开放 Dashboard 输入。
        await self._control.start()

    async def run(self, driver: _ClaudeSessionDriver) -> None:
        # Control 等待 interaction_version 变化并处理替换 > Esc > pending message。
        await self._control.run(driver)

    async def on_message(
        self,
        driver: _ClaudeSessionDriver,
        message: Any,
    ) -> None:
        if _kind(message) == "ResultMessage":
            # 一个 query 已完整结束：扣减 completion、切换 busy/idle，并安全 flush 消息。
            await self._control.complete(driver)
        elif _has_tool_result(message):
            # 工具结果落地意味着 call/result 已配对，是运行中注入排队消息的安全边界。
            await self._control.tool_completed(driver)

    async def close(self) -> None:
        try:
            # 先等待机器人/环境同步工具到达取消边界，避免封存后动作仍在后台继续。
            await self._control.cancel_active_toolkit()
        finally:
            # 即使 Toolkit 取消失败也必须 ended，关闭输入并把未完成消息保留为 unsent。
            self._control.end()


def _has_tool_result(message: Any) -> bool:
    # SDK 版本可能把工具结果编码为带 parent_tool_use_id 的 UserMessage，而不是 block list。
    kind = _kind(message)
    if kind == "UserMessage" and _get(message, "parent_tool_use_id"):
        return True
    if kind not in {"AssistantMessage", "UserMessage"}:
        return False
    content = _get(message, "content", [])
    # 同时兼容 dataclass 类名 ToolResultBlock 与 dict 协议 type=tool_result。
    return isinstance(content, list) and any(
        _kind(block) in {"ToolResultBlock", "tool_result"} for block in content
    )


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume SDK messages, emit text + accumulate stats.

    Holds no backend state. Errors that the SDK itself reports become
    ``recorder.error``; transport-level errors are written beside the transcript.
    """

    max_turns: int
    dashboard_events: DashboardEventSink
    # turn 按顶层 Assistant response 计数；同 message_id 的流式片段只算一次。
    turns: int = 0
    _seen_assistant_ids: set[str] = field(default_factory=set)
    # tool_calls 在工具结果落地时递增，代表已完成调用数而非仅 ToolUseBlock 数。
    tool_calls: int = 0
    # tool_use_id -> 去掉 MCP 前缀的短名，用于把后续 ToolResult 关联回工具。
    tool_names: dict[str, str] = field(default_factory=dict)
    # finish 参数必须等对应结果成功才提升为 finish_result，防止执行失败却误报结束。
    pending_finish: dict[str, dict[str, Any]] = field(default_factory=dict)
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_cache_creation_input_tokens": 0,
            "total_cache_read_input_tokens": 0,
        }
    )
    # ResultMessage 提供 SDK 结算后的美元成本；尚未提供时保持 None。
    total_cost_usd: float | None = None
    # 首个成功 finish 工具结果；Driver 观察到后立即停止会话。
    finish_result: dict[str, Any] | None = None
    # SDK ResultMessage 报告的真实运行错误，外层 transport 错误由 Planner 单独保存。
    error: str | None = None
    #: Set by the interactive loop before a user-initiated ``interrupt`` so the
    #: next result (which the CLI may flag ``is_error``) is not mistaken for a
    #: real failure. Cleared by the next result regardless of its outcome.
    suppress_next_result_error: bool = False

    # -- public ------------------------------------------------------------

    def stats(self) -> dict[str, int | float | None]:
        # 返回普通可序列化快照；usage 展开后供三个 Planner 后端统一写 stats。
        return {
            "turns_used": self.turns,
            "tool_calls": self.tool_calls,
            "total_cost_usd": self.total_cost_usd,
            **self.usage,
        }

    def observe(self, message: Any) -> str:
        # SDK 消息通常按 System -> Assistant -> User(tool result) -> Result 生命周期到达；
        # _kind 同时兼容 dataclass 消息和 JSON/dict 形状。
        kind = _kind(message)
        if kind == "SystemMessage":
            rendered = self._system(message)
        elif kind == "AssistantMessage":
            rendered = self._assistant(message)
        elif kind == "UserMessage":
            rendered = self._user(message)
        elif kind == "ResultMessage":
            rendered = self._result(message)
        else:
            # 未识别 SDK 扩展消息仍保留在 raw JSONL，但不写人类 transcript。
            rendered = ""
        # 每条消息后发布当前累计快照；DashboardState 采用替换而非累加，重复安全。
        self.dashboard_events.emit(
            UsageEvent(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        )
        return rendered

    # -- per-message handlers ---------------------------------------------

    def _system(self, message: Any) -> str:
        subtype = _get(message, "subtype", "")
        if subtype == "thinking_tokens":
            # 高频 thinking token bookkeeping 不具备用户可读价值，raw JSONL 仍会保留。
            return ""
        data = _get(message, "data", {})
        session = data.get("session_id") if isinstance(data, dict) else ""
        return f"[cc-system] subtype={subtype} session={session}\n"

    def _assistant(self, message: Any) -> str:
        # AssistantMessage usage 可能是当前分片增量，先累加；最终 ResultMessage 会校准覆盖。
        self._add_usage(_get(message, "usage"))
        lines: list[str] = []
        if _get(message, "parent_tool_use_id") is None:
            # 带 parent_tool_use_id 的嵌套/子代理响应不占顶层 RPent turn。
            # One Claude response may arrive as multiple messages with the same ID.
            assistant_id = _get(message, "message_id") or _get(message, "uuid")
            if not assistant_id or assistant_id not in self._seen_assistant_ids:
                if assistant_id:
                    self._seen_assistant_ids.add(str(assistant_id))
                self.turns += 1
                lines.append(f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n")
        for block in _get(message, "content", []) or []:
            block_kind = _kind(block)
            if block_kind == "TextBlock":
                text = str(_get(block, "text", "")).strip()
                if text:
                    lines.append(f"[claude] {text}\n")
                    # Dashboard transcript 与磁盘日志分开发布，前端只接收结构化纯文本。
                    self.dashboard_events.emit(
                        TranscriptEvent({"type": "text", "text": text})
                    )
            elif block_kind == "ThinkingBlock":
                thinking = str(_get(block, "thinking", "")).strip()
                if thinking:
                    lines.append(f"[claude-thinking] {thinking}\n")
                    self.dashboard_events.emit(
                        TranscriptEvent({"type": "thinking", "text": thinking})
                    )
            elif block_kind == "ToolUseBlock":
                tool_id = str(_get(block, "id", ""))
                # SDK 返回 MCP 全名，RPent transcript/Toolkit 始终使用短名。
                name = strip_mcp_prefix(str(_get(block, "name", "tool")))
                # 保存 ID 关联，工具结果可能在后续 UserMessage 中单独到达。
                self.tool_names[tool_id] = name
                tool_input = _get(block, "input", {}) or {}
                if name == "finish" and isinstance(tool_input, dict):
                    # 此时只暂存参数；必须等对应 ToolResult 成功才认定任务结束。
                    self.pending_finish[tool_id] = dict(tool_input)
                lines.append(f"[tool->] {name}: {_short_json(tool_input, limit=500)}\n")
                # 仅投影“模型请求调用”，实际工具执行由进程内 MCP handler 完成。
                self.dashboard_events.emit(
                    TranscriptEvent(
                        {"type": "tool_call", "tool": name, "args": tool_input}
                    )
                )
            elif block_kind == "ToolResultBlock":
                # 某些 SDK 版本把结果直接放在 AssistantMessage，复用统一结果处理器。
                lines.append(self._tool_result(block))
        if assistant_error := _get(message, "error"):
            # Assistant 层错误写日志；最终 Planner error 仍以 ResultMessage/transport 为准。
            lines.append(f"[cc-assistant-error] {assistant_error}\n")
        return "".join(lines)

    def _user(self, message: Any) -> str:
        tool_use_id = _get(message, "parent_tool_use_id")
        content = _get(message, "content", "")
        if tool_use_id:
            # Claude SDK 常把 MCP 返回包装为带 parent ID 的 UserMessage。
            return self._tool_result_content(content, tool_use_id=str(tool_use_id))
        if isinstance(content, list):
            # 兼容无 parent ID、content 中包含一个或多个 ToolResultBlock 的消息形状。
            return "".join(
                self._tool_result(block)
                for block in content
                if _kind(block) == "ToolResultBlock"
            )
        # 普通用户输入由 _emit_user 单独记录，避免 SDK echo 重复写 transcript。
        return ""

    def _tool_result(self, block: Any, *, tool_use_id: str | None = None) -> str:
        return self._tool_result_content(
            _get(block, "content", ""),
            # 显式 parent ID 优先，否则读取 block 自身关联字段。
            tool_use_id=tool_use_id or str(_get(block, "tool_use_id", "")),
            is_error=_get(block, "is_error"),
        )

    def _tool_result_content(
        self,
        content: Any,
        *,
        tool_use_id: str,
        is_error: Any = None,
    ) -> str:
        # 调用在结果到达时计数，因此 tool_calls 表示已结算调用；错误结果也算一次调用。
        self.tool_calls += 1
        name = self.tool_names.get(tool_use_id, "tool_result")
        # Dashboard 只接收大小/图片数/错误标志摘要，完整多模态 content 留在 raw 流。
        summary: dict[str, Any] = {"size": _payload_size(content)}
        image_count = _content_image_count(content)
        if image_count:
            summary["images"] = image_count
        if is_error:
            summary["is_error"] = bool(is_error)
        # Promote the finish payload once the tool result lands successfully.
        # pop 保证同一 tool_use_id 最多提升一次；失败 finish 会被移除而不会终止会话。
        pending = self.pending_finish.pop(tool_use_id, None)
        if pending is not None and not is_error and self.finish_result is None:
            # 首个成功 finish 胜出，后续重复 finish 不覆盖最终状态。
            self.finish_result = {"_finish": True, **pending}
        self.dashboard_events.emit(
            TranscriptEvent(
                {
                    "type": "tool_result",
                    "tool": name,
                    "result": {**summary, "is_error": bool(is_error)},
                }
            )
        )
        return f"[tool<-] {name}: {json.dumps(summary, ensure_ascii=False)}\n"

    def _result(self, message: Any) -> str:
        if usage := _get(message, "usage"):
            # ResultMessage 给出本 query 的最终权威统计，覆盖 Assistant 分片累计值。
            self._set_usage(usage)
        if cost := _get(message, "total_cost_usd"):
            self.total_cost_usd = float(cost)
        # 标志无论结果是否 error 都只消费一次，确保只抑制紧随主动 interrupt 的结果。
        suppress = self.suppress_next_result_error
        self.suppress_next_result_error = False
        if _get(message, "is_error", False) and not suppress:
            self.error = f"Claude Agent SDK result {_get(message, 'subtype', 'error')}"

        # 人类日志只写结果元数据和 result 字符数，不复制可能很长的最终 payload。
        parts = ["[cc-result]", str(_get(message, "subtype", ""))]
        if duration_ms := _get(message, "duration_ms"):
            parts.append(f"duration={duration_ms / 1000:.1f}s")
        if self.total_cost_usd is not None:
            parts.append(f"cost=${self.total_cost_usd:.4f}")
        if result := str(_get(message, "result", "") or ""):
            parts.append(f"result_size={len(result)}")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cache_create={self.usage['total_cache_creation_input_tokens']} "
            f"cache_read={self.usage['total_cache_read_input_tokens']} "
            f"out={self.usage['total_output_tokens']} tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- usage helpers ----------------------------------------------------

    def _add_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        # AssistantMessage 可能分片到达，每片 usage 按增量累加。
        self.usage["total_input_tokens"] += int(usage.get("input_tokens") or 0)
        self.usage["total_output_tokens"] += int(usage.get("output_tokens") or 0)
        self.usage["total_cache_creation_input_tokens"] += int(
            usage.get("cache_creation_input_tokens") or 0
        )
        self.usage["total_cache_read_input_tokens"] += int(
            usage.get("cache_read_input_tokens") or 0
        )

    def _set_usage(self, usage: Any) -> None:
        if not isinstance(usage, dict):
            return
        # ResultMessage 的最终数字是权威总量，整体覆盖可修正分片缺失/重复造成的偏差。
        self.usage = {
            "total_input_tokens": int(usage.get("input_tokens") or 0),
            "total_output_tokens": int(usage.get("output_tokens") or 0),
            "total_cache_creation_input_tokens": int(
                usage.get("cache_creation_input_tokens") or 0
            ),
            "total_cache_read_input_tokens": int(
                usage.get("cache_read_input_tokens") or 0
            ),
        }


# ---------------------------------------------------------------------------
# Tool bridge (RPent registry -> SDK MCP server)
# ---------------------------------------------------------------------------


def _build_rpent_server(sdk: Any, *, toolkit: Toolkit) -> Any:
    """把 RPent 工具注册表转换为 Claude SDK 的进程内 MCP server。

    这里不会启动额外 HTTP 服务。Claude Agent SDK 通过内存中的 MCP server 看到
    ``mcp__rpent__<tool>``；每次模型工具调用最终仍落到同一个
    ``toolkit.execute_tool``。因此 VLA、环境和状态记录都继续受 RPent 控制。
    """
    sdk_tools = []

    # Claude SDK 可能并发发起工具请求，但单个机器人环境只能串行推进。外层 MCP
    # 桥先串行化请求，Toolkit 内部还有第二层锁作为跨 Planner 的通用保护。
    tool_execution_lock = asyncio.Lock()
    for spec in toolkit.get_tools_spec():
        name = str(spec["name"])
        description = str(spec.get("description", ""))
        input_schema = spec.get("input_schema", {"type": "object"})

        async def run_tool(
            args: dict[str, Any],
            *,
            tool_name: str = name,
        ) -> dict[str, Any]:
            # ``tool_name=name`` 用默认参数冻结本轮循环变量，避免 Python 闭包晚绑定
            # 导致所有 MCP handler 最终都调用最后一个工具。
            async with tool_execution_lock:
                # primitive/RPC 是同步接口，放入工作线程，避免阻塞 Claude SDK 的
                # asyncio 消息流和 Dashboard 交互。
                result = await asyncio.to_thread(
                    toolkit.execute_tool,
                    tool_name,
                    # 防御 SDK 对无参数工具传 None，Toolkit 始终接收 dict。
                    args or {},
                )
            # MCP 边界统一转换文本/图片 block 和 is_error，不泄漏 ToolResult Python 对象。
            return _tool_result_to_mcp(result)

        # 名称仅用于 SDK/日志诊断；真正公开的工具名仍来自 spec。
        run_tool.__name__ = f"rpent_{name}"
        sdk_tools.append(sdk.tool(name, description, input_schema)(run_tool))

    return sdk.create_sdk_mcp_server(
        name="rpent",
        version="0.1.0",
        tools=sdk_tools,
    )


def _tool_result_to_mcp(tr: Any) -> dict[str, Any]:
    """将 RPent ``ToolResult`` 转换为 Claude SDK 所需的 MCP content。"""
    # Toolkit 已把环境结果组织为 Anthropic 风格的文本/图片 block。这里只改字段
    # 形状，不重新读取图像，也不丢弃动作后采集的多模态反馈。
    blocks = getattr(tr, "content_blocks", None)
    if blocks is None:
        # 兼容非标准 handler 返回值，至少把其字符串表示作为文本反馈给模型。
        return {"content": [{"type": "text", "text": str(tr)}]}

    content: list[dict[str, Any]] = []
    for block in blocks:
        block_type = _get(block, "type")
        if block_type == "text":
            content.append({"type": "text", "text": _get(block, "text", "")})
        elif block_type == "image":
            src = _get(block, "source", {})
            # Toolkit 使用 Anthropic 风格 source；SDK MCP 形状将 media_type 改名 mimeType。
            content.append(
                {
                    "type": "image",
                    "data": _get(src, "data", ""),
                    "mimeType": _get(src, "media_type", "image/png"),
                }
            )
        # 未知 block 不猜测语义；完整结果仍保存在 Toolkit 自身状态/artifact 中。

    response: dict[str, Any] = {"content": content}
    result_dict = getattr(tr, "result", None)
    if isinstance(result_dict, dict) and result_dict.get("error"):
        # MCP 的 is_error 让 Claude 知道该工具失败，但会话仍可继续并尝试恢复。
        response["is_error"] = True
    return response


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _kind(value: Any) -> str:
    if isinstance(value, dict):
        # 原始 JSON/dict 消息可能使用 type 或 kind；优先遵循 type。
        return str(value.get("type") or value.get("kind") or "")
    # SDK dataclass/对象以类名作为稳定分派键，如 AssistantMessage、TextBlock。
    return value.__class__.__name__


def _get(value: Any, key: str, default: Any = None) -> Any:
    # 统一兼容原始 dict 和 SDK 对象，Recorder 无需绑定某个 SDK 版本的具体类。
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _message_to_json(message: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(message):
        # asdict 递归展开 SDK dataclass，便于 JSONL 排障。
        data = dataclasses.asdict(message)
    elif hasattr(message, "__dict__"):
        data = vars(message)
    else:
        # 未知对象至少保存 repr，不能因日志序列化失败中断模型会话。
        data = {"value": repr(message)}
    return {"type": _kind(message), **_jsonable(data)}


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        # 原始流不内联二进制/图片，避免 JSONL 暴涨；保留大小用于诊断。
        return {"type": "bytes", "size": len(value)}
    return value


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    # 每条消息立即可见，便于观察长运行和异常退出后的部分日志。
    file_obj.flush()


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    # 只裁剪人类日志；实际 MCP 参数和 raw JSONL 不受影响。
    return text[:limit] + f"...(+{len(text) - limit})"


def _payload_size(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        # 字符串以字符数摘要；复杂结构使用 JSON 序列化后的字符数。
        return len(value)
    return len(json.dumps(value, ensure_ascii=False, default=str))


def _content_image_count(value: Any) -> int:
    if isinstance(value, list):
        # 标准 MCP content list 直接按结构化 image block 计数。
        count = 0
        for item in value:
            if isinstance(item, dict) and item.get("type") == "image":
                count += 1
        return count
    # 兼容 SDK 把 content 包成对象/字符串表示的旧形状；仅用于日志摘要，不参与解析。
    text = str(value)
    return text.count("'type': 'image'") + text.count('"type": "image"')
