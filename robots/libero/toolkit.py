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
        # 每次运行使用独立 EnvState，根目录由 CLI 初始化的 output_dir 决定。
        state = EnvState(get_output_dir())

        # 基类先注册 read_text_file/finish 等通用工具，并建立工具调用串行锁。
        super().__init__(dashboard_events=dashboard_events, state=state)

        # 创建 primitive、reset 环境并保存 step 0；此后才注册依赖 primitive 的工具，
        # 保证 Planner 第一次看到工具时，环境和初始观测已经可用。
        self.init_primitives_clean(primitives_kwargs=primitives_kwargs)
        self._register_libero_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------
    def _register_libero_tools(self) -> None:
        """把 ``TOOLS_SPEC`` 中的 schema 与实际 Python handler 绑定。

        只读感知工具需要额外绑定本次运行的 ``EnvState``；动作工具则按同名方法
        直接绑定到 ``LiberoPrimitives``。最终注册表由 Planner 转换为 API tool 或
        Claude SDK 的进程内 MCP tool。
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
        """在状态型工具结束后保存并返回新的可观测环境状态。

        基类 ``Toolkit.execute_tool`` 会为所有非 ``@readonly`` handler 调用本方法。
        因此 VLA 动作块执行完毕后，Planner 会收到刷新后的主相机/腕部图像、机器人
        状态、terminated/truncated 标志和工具结果，而不是只看到 primitive 的摘要。
        """
        # 记录本次工具开始前的视频帧游标，用于 Dashboard 生成单动作短视频。
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
        """清空旧产物、创建 primitive、reset 环境并保存初始 step 0。"""
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
        """结束录制并把整个 episode 的帧保存为视频。"""
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
        """从已保存的状态轨迹导出可复用 LIBERO primitive JSONL。"""
        return libero_tools.write_recipe_from_states(self._state, recipe_tag)
