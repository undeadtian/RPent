"""LIBERO 工具集：通用工具 + LIBERO 环境原语。

该类是 LLM 工具调用与物理执行之间的环境适配层：

``Planner/MCP -> Toolkit.execute_tool -> LiberoToolkit handler``
``-> LiberoPrimitives -> env/VLA/SAM3 RPC client``

基类 :class:`Toolkit` 提供通用文件工具、注册表、并发/取消控制和统一返回格式；
本类负责创建单次运行的 ``EnvState``、注册 LIBERO 专属工具，以及在每个会改变
环境的工具执行后保存新的 RGB-D/机器人状态。
"""
from __future__ import annotations

from functools import partial
from typing import Any

from robots.libero import tools as libero_tools
from rpent.dashboard.events import DashboardEventSink
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger, get_output_dir


class LiberoToolkit(Toolkit):
    """把 LIBERO primitive 暴露为 Planner 可调用工具的适配器。

    ``primitives_kwargs`` 由 ``robots.libero._init_runtime`` 构造，包含三个已经通过
    readiness 检查的客户端：``env``、``model``（VLAClient）和 ``sam3_client``。
    Toolkit 不负责启动这些服务，但在整个 Agent loop 中持有并使用它们。
    """

    # Dashboard 发布 StepRecord 时，用这些逻辑名称找到本步的主相机和腕部图像。
    _FRAME_ARTIFACTS = {
        "camera": "agentview.png",
        "wrist": "wrist.png",
    }

    def __init__(
        self,
        *,
        primitives_kwargs: dict[str, Any],
        dashboard_events: DashboardEventSink,
    ) -> None:
        """创建单次运行的状态仓库、primitive，并注册 Planner 工具。

        初始化顺序有意固定为：创建独立 ``EnvState`` → 初始化基类通用工具与锁 →
        reset LIBERO 并保存/发布 step 0 → 注册依赖 primitive 的专属 handler。这样
        Planner 首次拿到工具列表时，环境、初始 RGB-D 状态和 Dashboard 都已经同步。

        Args:
            primitives_kwargs: 构造 :class:`LiberoPrimitives` 所需的环境、VLA、SAM3
                客户端等依赖；取消回调由 Toolkit 自行注入。
            dashboard_events: 统一的 Dashboard 事件出口；禁用时仍正常保存状态，
                但跳过单动作视频开销。
        """
        # 每次运行使用独立 EnvState，根目录由 CLI 初始化的 output_dir 决定。
        state = EnvState(get_output_dir())

        # 基类先注册 read_text_file/finish 等通用工具，并建立工具调用串行锁。
        super().__init__(dashboard_events=dashboard_events, state=state)

        # 创建 primitive、reset 环境并保存 step 0；此后才注册依赖 primitive 的工具，
        # 保证 Planner 第一次看到工具时，环境和初始观测已经可用。
        self.init_primitives_clean(primitives_kwargs=primitives_kwargs)
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # LIBERO 专属工具注册
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        """把 ``TOOLS_SPEC`` 的公开 schema 逐项绑定到运行时 handler。

        schema 本身只保存 Planner 可见的工具名、描述和 JSON 参数约束。注册时，
        ``view_env_state``、``view_camera_meta``、``back_project`` 通过 ``partial``
        注入当前运行的 ``EnvState``，``segment`` 还绑定当前 primitive 的 SAM3 客户端；
        其余条目按名称查找 :class:`LiberoPrimitives` 方法。找不到实现的 schema 会
        跳过，避免暴露调用后必然失败的工具。

        :meth:`add_tool` 最终把同一份 schema/handler 对注册到 Planner API 或
        Claude SDK 进程内 MCP 层；内部 ``state`` 不会出现在公开参数中。
        """
        # 这些 handler 只读取已落盘的状态，不推进模拟器。partial 把当前运行的
        # EnvState 注入进去，使 LLM 的公开参数中不出现内部 state 对象。
        state_handlers = {
            "view_env_state": partial(
                libero_tools.view_env_state,
                state=self._state,
            ),
            "view_camera_meta": partial(
                libero_tools.view_camera_meta,
                state=self._state,
            ),
            "back_project": partial(
                libero_tools.back_project,
                state=self._state,
            ),
            "segment": partial(
                self._primitives.segment,
                state=self._state,
            ),
        }

        for spec in libero_tools.TOOLS_SPEC:
            name = spec["name"]
            if name in state_handlers:
                handler = state_handlers[name]
            else:
                # 动作 schema 与 LiberoPrimitives 的方法同名，例如 pi0_pick、
                # move_to、release。没有实现的方法不会暴露给 Planner。
                handler = getattr(self._primitives, name, None)
                if handler is None:
                    continue
            self.add_tool(name, spec, handler)

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        """在状态型工具结束后持久化观测，并返回 Planner 状态反馈。

        基类 :meth:`Toolkit.execute_tool` 会为所有非 ``@readonly`` handler 调用本
        方法，无论 primitive 成功还是报错。这里先以动作开始前的帧游标截取本次
        新增帧，再调用 :func:`libero_tools.dump_state` 创建 ``StepRecord``，把 RGB-D、
        世界图、机器人状态、命令、原始结果和耗时绑定到同一步。

        Dashboard 启用时，新增帧另存为 ``action_<tool>.mp4``；编码失败只记录警告。
        最后通过 ``view_env_state`` 生成 Planner 所需的文本/图像结果。方法返回后，
        基类会取得最新 ``StepRecord`` 并调用 ``_publish_step``，依据
        ``_FRAME_ARTIFACTS`` 发布主相机/腕部帧和状态事件，因此落盘始终先于发布。
        """
        # 游标在上次状态采集结束时指向缓冲区尾部；先保存旧值作为本动作起点，
        # 再推进到当前尾部，使后续动作不会重复包含这些帧。
        frame_start = self._action_frame_cursor
        self._action_frame_cursor = self._primitives.recorded_frame_count()

        # dump_state 创建新的 StepRecord，并把 RGB、深度、世界坐标图、相机标定、
        # 命令、原始结果和耗时写入该 step 的 artifacts。
        record = libero_tools.dump_state(
            self._primitives,
            self._state,
            log={
                "command": command,
                "result": result,
                "elapsed_s": elapsed_s,
            },
        )

        # Dashboard 启用时额外保存该工具期间产生的帧；普通 CLI 跳过这项开销。
        if self._dashboard_events.enabled:
            try:
                frames = self._primitives.frame_slice(frame_start)
                if frames:
                    candidate = f"action_{command['action']}.mp4"
                    self._state.save(
                        candidate,
                        frames,
                        step=record.step_idx,
                        fps=20,
                    )
            except Exception as e:
                # 单动作视频只是诊断产物，失败不应让机器人任务失败。
                get_logger("libero_toolkit").warning(
                    "failed to save action clip for step %s: %s",
                    record.step_idx,
                    e,
                )

        # view_env_state 把 StepRecord 转成 ToolResult 最终使用的文本和图像字段。
        out = libero_tools.view_env_state(record.step_idx, state=self._state)
        out["agent_elapsed_s"] = elapsed_s
        if result.get("interrupted"):
            # 保留取消原因，避免状态快照覆盖 tool_cancelled 诊断。
            out.update(result)
        return out

    def init_primitives_clean(
        self,
        *,
        primitives_kwargs: dict[str, Any],
    ) -> None:
        """重置运行产物与环境，并建立录制中的初始 step 0。

        清空 ``EnvState`` 后创建 primitive，注入 Toolkit 的取消检查回调；随后 reset
        远端环境、开启全 episode 录制并把动作片段游标设为当前帧数。初始观测以
        ``log=None`` 落盘，因此 step 0 没有关联工具命令。primitive 仅在完整记录
        写入后赋给实例，最后显式发布初始 Dashboard 状态。
        """
        self._state.reset()

        primitives = libero_tools.LiberoPrimitives(
            # Dashboard/交互模式可设置取消标记；primitive 在安全边界主动检查。
            check_cancelled=self.raise_if_cancelled,
            **primitives_kwargs,
        )
        primitives.reset()
        primitives.start_recording()
        self._action_frame_cursor = primitives.recorded_frame_count()

        # step 0 没有关联命令，代表 Planner 开始推理前的初始观测。
        record = libero_tools.dump_state(primitives, self._state, log=None)
        self._primitives = primitives
        self._publish_step(record)

    def close(self) -> None:
        """停止录制，并把共享帧时间线保存为完整 episode 视频。

        ``stop_recording`` 会一次性移交并清空 primitive 缓冲区；存在帧时，以固定
        20 FPS 保存到运行根级 ``episode.mp4``。该视频覆盖 VLA chunk 内每个环境步
        和脚本化 OSC 的逐步帧，而 Dashboard 单动作短片只是同一缓冲区的切片。
        关闭阶段属于尽力清理，编码异常只写日志，不能覆盖 Agent 的真实任务结果。
        """
        try:
            frames = self._primitives.stop_recording()
            if frames:
                self._state.save("episode.mp4", frames, step=None, fps=20)
        except Exception as e:
            # 清理阶段不能用视频编码错误掩盖前面的 Agent 执行结果。
            get_logger("libero_toolkit").warning(
                f"failed to save episode video: {e}"
            )

    def write_recipe(self, recipe_tag: str) -> str:
        """从当前 ``EnvState`` 轨迹导出可复用的 LIBERO primitive recipe。

        实际筛选、排序和 JSONL 保存委托给
        :func:`libero_tools.write_recipe_from_states`；返回生成的产物名，供上层展示
        或后续重放。该操作不推进环境。
        """
        return libero_tools.write_recipe_from_states(self._state, recipe_tag)
