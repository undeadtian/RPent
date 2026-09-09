"""Lightweight event boundary for optional Dashboard updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, TypeAlias


# =============================================================================
# Dashboard 结构化事件边界
# =============================================================================
#
# 本模块只定义“事件长什么样”和“接收器至少提供什么接口”，不依赖 DashboardState、
# HTTP/SSE 服务或具体机器人环境。这样 Planner、Toolkit 和 runtime 只需持有
# DashboardEventSink，就能发布事件而不必知道前端状态如何保存和广播。
#
# 主要数据流为：
#
#   Planner / Toolkit / environment runtime
#                    │ emit(event)
#                    ▼
#          DashboardEventSink 协议
#                    │
#                    ▼
#        DashboardState（启用 Dashboard）
#        或 NullDashboardEventSink（普通 CLI）
#
# DashboardState.emit() 是当前统一投影入口：它按事件实际类型更新 transcript、usage、
# runtime、帧、timeline 或任务状态。事件生产者不直接修改 DashboardState 内部字段。
#
# 所有事件 dataclass 都采用 frozen=True 和 slots=True：前者禁止重新绑定事件字段，后者
# 避免每个小事件创建 __dict__。注意 frozen 是“浅层”不可变：payload/result/dict 等字段
# 指向的对象仍可能可变，所以发布方应把事件视为所有权已经移交，emit 后不要继续修改。
#
# 并发责任位于 Sink 实现：事件可能来自 asyncio loop、Planner 工作线程或 runtime 启动
# 路径，发布方不围绕 emit() 额外加 Dashboard 锁；DashboardState 在投影可变状态时自行
# 同步。Null sink 则完全无状态，可安全复用。
# =============================================================================


@dataclass(frozen=True, slots=True)
class TranscriptEvent:
    """Append one existing frontend transcript payload."""

    # payload 已是前端理解的 JSON-like 事件，而不是 Pydantic AI/Codex/Claude 的原始
    # SDK 消息。常见 type 包括 initial_prompt、user、text、thinking、tool_call 和
    # tool_result；DashboardState 只按到达顺序追加，不在此边界重新解释后端协议。
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class UsageEvent:
    """Replace the cumulative planner usage counters."""

    # 三个数字都是“当前会话截至此刻的累计快照”，不是相对上一事件的增量。
    # DashboardState 收到后整体替换 usage，因此重复发送同一快照不会重复计数。
    inp: int
    out: int
    tool_calls: int


@dataclass(frozen=True, slots=True)
class RuntimeStatusEvent:
    """Update one environment-side runtime component."""

    # component 是 Dashboard spec 中声明的运行时名称，例如 env、vla、sam3；消费端会
    # 拒绝未知名称，防止拼写错误悄悄创建一条无法显示的新状态。
    component: str
    # status 当前由状态层校验为 pending/starting/ready/failed。生产者通常发布
    # starting -> ready，任一启动/连接/健康检查异常则发布 failed。
    status: str
    # error 保留异常对象或说明文字到投影边界；DashboardState 最终只公开 str(error)，
    # 避免异常实例进入 JSON 快照。正常状态使用 None。
    error: BaseException | str | None = None


@dataclass(frozen=True, slots=True)
class ToolResultEvent:
    """Publish one raw tool result for Dashboard projection."""

    # name 是 Toolkit 注册的短工具名，用于校验 result.log.command.action 并生成 timeline。
    name: str
    # result 有意保持 Any，使不同环境可发布自己的工具结果。目前 DashboardState 只处理
    # dict；非 dict 会被忽略。已识别字段包括 step、log、terminated、truncated、
    # action_video_path/action_video_artifact，以及 _image_cam_bytes、_image_wrist_bytes
    # （兼容旧字段 _image_bytes）。该事件只是类型契约；当前仓库尚无直接发布点。
    result: Any


@dataclass(frozen=True, slots=True)
class StepRecordEvent:
    """Publish one recorded environment step and its artifact context."""

    # record 通常是 Toolkit 在有状态工具完成后捕获的 StepRecord，包含 step_idx、command、
    # result、终止标志和 artifact 名称；使用 Any 可让事件边界不反向依赖 state 模块。
    record: Any
    # env_state 是产生该 record 的状态仓库。消费端用它解析/加载 record 中登记的 artifact，
    # 因而必须与 record 属于同一 TaskRun，不能混用其他任务的 EnvState。
    env_state: Any
    # Dashboard 帧槽位到 artifact 名的映射，例如 camera -> agentview.png。发布方复制映射，
    # 消费端也再次复制，避免类级配置在任务运行中被意外共享修改。
    frame_artifacts: dict[str, str]


@dataclass(frozen=True, slots=True)
class RunStartedEvent:
    """Mark startup complete and the agent run active."""

    # 这是没有数据字段的纯信号事件。它应在 runtime 已就绪且任务替换竞态检查通过后
    # 发布，把 Dashboard 任务从 task_starting 推进为 running；它不表示某个工具开始。


# DashboardState.emit() 当前支持的封闭事件集合。联合别名同时为生产者和 Sink 提供
# 静态检查；运行时消费端仍用 isinstance 分派，并对未知事件抛 TypeError，确保将来添加
# 新事件时不会因忘记实现投影逻辑而被静默丢弃。
DashboardEvent: TypeAlias = (
    TranscriptEvent
    | UsageEvent
    | RuntimeStatusEvent
    | ToolResultEvent
    | StepRecordEvent
    | RunStartedEvent
)


class DashboardEventSink(Protocol):
    """Consumer used by planners, toolkits, and environment runtimes."""

    # Protocol 使用结构化类型：具体类无需继承本协议，只要提供同签名的 enabled 与 emit
    # 即可注入 Planner/Toolkit/runtime。这避免业务层 import DashboardState 具体实现。
    @property
    def enabled(self) -> bool:
        """Whether Dashboard-only projections and artifacts are needed."""
        # enabled 不控制 emit 是否可调用；它是昂贵 Dashboard 专属工作的快速判断。
        # 例如 LIBERO Toolkit 仅在 True 时额外截取动作帧，普通 CLI 可跳过采集开销。
        ...

    def emit(self, event: DashboardEvent) -> None:
        """Consume one Dashboard event."""
        # 调用是同步的，没有返回 ACK 或 await。生产者只保证事件按本线程调用顺序提交；
        # 跨线程同步、状态锁和向 SSE 客户端广播快照均由具体 Sink 负责。
        ...


@dataclass(frozen=True, slots=True)
class NullDashboardEventSink:
    """No-op sink used when the Dashboard is disabled."""

    # 使用对象而非到处传 None，使所有业务路径都能无条件调用 emit，并通过 enabled
    # 统一跳过可选的帧采集等成本；该对象没有任何可变字段，可在运行间安全复用。
    @property
    def enabled(self) -> bool:
        # False 提醒生产者不要创建只供 Dashboard 使用的额外 artifact。
        return False

    def emit(self, event: DashboardEvent) -> None:
        # 有意接收并丢弃全部合法事件，让普通 CLI 与 Dashboard 共用同一发布代码。
        return None
