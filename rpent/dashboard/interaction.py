"""Shared types and the planner-facing Dashboard interaction port."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol


# =============================================================================
# Planner 与 Dashboard 状态层之间的交互协议
# =============================================================================
#
# 本模块只描述“Planner 控制器可以对 Dashboard 交互状态做什么”，不保存真实状态，也不
# 依赖 HTTP、SSE、具体模型 SDK 或机器人环境。当前角色分工为：
#
#   浏览器请求
#       │ 写入任务命令、消息、中断请求
#       ▼
#   DashboardState                  —— 本 Protocol 的主要实现者
#       ▲
#       │ DashboardInteractionPort  —— 本模块定义的窄接口
#       ▼
#   DashboardPlannerControl         —— 认领消息/中断并驱动 API、Codex 等后端
#
# 使用 Protocol 而非具体 DashboardState 类型，使 Planner bridge 只依赖交互能力；测试或
# 新状态后端只要结构上提供相同属性和方法，无需继承即可满足类型检查。
#
# 并发策略不写死在 Protocol 中：调用方只依赖“认领和状态迁移是原子的”这一语义，具体
# 实现负责 Lock/Condition。DashboardState 使用单调 interaction_version + Condition，
# DashboardPlannerControl 则把阻塞等待移到 asyncio worker thread，避免冻结事件循环。
# =============================================================================


# Planner 对当前 TaskRun 的活动状态。该值描述交互后端，不等同于机器人环境 task state：
# - starting：正在建立初始模型请求，尚不能安全接收后续消息；
# - idle：没有未完成后端请求，可立即提交 pending 消息；
# - busy：模型请求、turn 或工具链仍在执行，只能在安全边界 flush；
# - ended：当前 TaskRun 的交互已永久封存，不能重新开放。
PlannerActivity = Literal["starting", "idle", "busy", "ended"]

# 一条 Dashboard 用户消息的生命周期：
#
#   pending ── claim_next_pending_message ─> sending ─┬─> sent
#      │                                              ├─> failed
#      └─ 用户撤回 ─> withdrawn                       └─> unsent
#
# sending 是 bridge 的独占认领状态。sent 表示已真正提交到后端；failed 保存提交错误；
# unsent 保留已认领但最终未执行的文本；withdrawn 只适用于仍处于 pending 的消息。
DashboardMessageStatus = Literal[
    "pending",
    "sending",
    "sent",
    "withdrawn",
    "failed",
    "unsent",
]


@dataclass(slots=True)
class DashboardMessage:
    """One user message submitted through a Dashboard Session."""

    # 消息对象有意不是 frozen：DashboardState 在持锁临界区内原地推进 status/error，
    # 同时用 list 保留展示顺序、用 ID 索引定位同一对象。返回给外部时实现方应复制或转 dict。
    # ID 由 Session 内部生成并在该 TaskRun 内唯一，用于 HTTP 撤回和异步 ACK 对账。
    message_id: str
    # 保存用户原文；状态切换不会重写文本，unsent/failed 后仍可向前端展示。
    text: str
    # 当前六态之一；合法迁移由 DashboardState 校验，而 dataclass 自身不执行业务约束。
    status: DashboardMessageStatus
    # 通常只在 failed 时保存后端提交错误；其他状态应为 None。
    error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return the JSON-safe representation exposed to the frontend."""
        # 返回新 dict 而非暴露可变消息对象，使 SSE/HTTP 序列化发生在锁外也不会反向
        # 修改内部状态；字段名是 Dashboard 前端 interaction.messages 的稳定契约。
        return {
            "message_id": self.message_id,
            "text": self.text,
            "status": self.status,
            "error": self.error,
        }


class DashboardInteractionPort(Protocol):
    """Planner-facing access to one Dashboard interaction Session."""

    # 该接口刻意不暴露 timeline、frame、runtime 或 HTTP 细节；Planner 控制器只需要
    # 活动状态、消息 ACK、中断和任务替换。每个方法的线程安全由实现者负责。
    @property
    def planner_activity(self) -> PlannerActivity:
        """Return the current planner input activity."""
        # Control 用 ended 退出主循环，用 idle 决定能否立即 flush；busy 消息需等待
        # complete/tool_completed 安全边界，starting 表示初始提交尚未成功。
        ...

    @property
    def interaction_version(self) -> int:
        """Return the monotonic interaction-state version."""
        # 版本号是变化游标而非消息数量：消息、中断、activity 或任务替换任一变化都可递增。
        ...

    def wait_for_interaction_change(
        self,
        since: int,
        timeout: float | None = None,
    ) -> int:
        """Wait for an interaction change and return the latest version."""
        # 调用者传入先前观察的 since；若版本已经变化应立即返回，否则阻塞到变化或超时。
        # 该方法是同步阻塞接口，async Control 必须通过 asyncio.to_thread 调用。
        ...

    def claim_next_pending_message(self) -> DashboardMessage | None:
        """Claim the next pending message, if task replacement is not pending."""
        # 原子执行最早一条 pending -> sending，并返回供 driver 提交的快照。ended、任务
        # 替换或中断挂起时应返回 None，确保旧会话不会继续吸收新输入。
        ...

    def mark_message_sent(self, message_id: str) -> DashboardMessage:
        """Commit one successfully submitted message."""
        # 只允许 sending -> sent。非延迟 driver 在 submit 成功后调用；延迟 ACK driver
        # 则在内部队列中的消息真正开始执行时调用，避免前端过早显示已发送。
        ...

    def mark_message_failed(
        self,
        message_id: str,
        error: str,
    ) -> DashboardMessage:
        """Record one failed message submission."""
        # 只允许 sending -> failed，并保留稳定错误文本；失败消息不会自动回到 pending。
        ...

    def mark_message_unsent(self, message_id: str) -> DashboardMessage:
        """Restore one queued submission that never started."""
        # 只允许 sending -> unsent。用于 API driver 中断/关闭时恢复已 claim 但尚未开始的
        # 内部排队消息，保留用户文本且不把它误标为后端失败。
        ...

    def claim_interrupt_request(self) -> bool:
        """Claim a queued interrupt request."""
        # 两阶段握手的 claim 阶段：True 表示当前调用者取得唯一处理权，随后应取消 Toolkit
        # 和 model driver；False 表示无请求或已由其他处理者置为 in-flight。
        ...

    def complete_interrupt(self, error: str | None = None) -> None:
        """Complete the claimed interrupt request."""
        # 必须在成功 claim 后调用，无论实际取消成功还是失败都要结算；error 用于前端反馈，
        # 但中断失败本身不自动把 TaskRun 标记为 failed。
        ...

    def set_planner_activity(
        self,
        activity: PlannerActivity,
        *,
        accepting_input: bool | None = None,
    ) -> None:
        """Update planner activity and optionally input acceptance."""
        # activity 描述后端是否有安全提交窗口；accepting_input 单独控制浏览器能否创建
        # 新 pending 消息。Control 仅在初始 backend submission 成功后将其设为 True。
        # None 表示保持现有输入开关；ended 应封存交互且不可在同一 TaskRun 重新打开。
        ...

    def seal_interaction(self) -> None:
        """End the interaction Session."""
        # 关闭当前 TaskRun 输入、清除中断握手，并把 pending/sending 消息保存为 unsent。
        # 它结束的是 Planner 对话，不负责停止 Session 级共享服务。
        ...

    @property
    def task_replacement_requested(self) -> bool:
        """Whether this TaskRun must yield to a newer task command."""
        # 替换请求优先于普通中断和消息 flush。Control 观察到 True 后应先停止活动工具，
        # 再中断 driver，并且不能继续向旧 Planner 提交消息。
        ...

    def complete_task_replacement(self, error: str | None = None) -> None:
        """Seal the old conversation after reaching a safe boundary."""
        # 通知状态层旧 Planner 已在工具/SDK 安全边界完成收尾。无论中断成功还是失败都应
        # 调用，避免 SessionController 永久等待；error 仅记录替换过程中的失败原因。
        ...


# 交互异常统一继承 RuntimeError：它们表示当前状态下操作无效，而不是输入类型或程序
# 配置错误。HTTP 层可按具体子类映射“不可用、消息不存在、状态冲突”等响应。
class DashboardInteractionError(RuntimeError):
    """Base class for invalid Dashboard interaction operations."""


class InteractionUnavailableError(DashboardInteractionError):
    """The Session is not currently accepting Dashboard input."""

    # 常见于共享服务仍启动、Session fatal、Planner ended、任务已完成或输入开关未开放。


class UnknownDashboardMessageError(DashboardInteractionError):
    """The requested message does not belong to this Session."""

    # message_id 不在当前 TaskRun 索引中；新 TaskRun 会清空旧消息索引。


class DashboardMessageConflictError(DashboardInteractionError):
    """A message can no longer make the requested state transition."""

    # ID 存在但状态已变化，例如撤回 sending 消息或重复确认 sent；用于暴露 HTTP 与
    # Planner bridge 之间的并发竞态，而不是静默覆盖已有状态。
