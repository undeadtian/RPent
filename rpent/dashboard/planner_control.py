"""Backend-neutral control flow for interactive Dashboard planners."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from rpent.dashboard.interaction import DashboardInteractionPort


# =============================================================================
# 中文实现导读：后端无关的 Dashboard Planner 控制器
# =============================================================================
#
# 本类位于 Dashboard 交互状态与具体 Planner driver 之间：
#
#   浏览器 / HTTP API
#       -> DashboardState（实现 DashboardInteractionPort）
#       -> DashboardPlannerControl
#       -> API 或 Codex 的 Dashboard Session driver
#       -> Model / Toolkit
#
# Control 不知道 Pydantic AI 的 node，也不知道 Codex 的 thread/turn。它只依赖 driver
# 提供的最小异步协议：
#
# * ``submit(text) -> int``：提交普通消息，返回新增 completion 数；
# * ``submit_dashboard_message(message) -> int``：延迟 ACK 模式的提交入口；
# * ``interrupt() -> int``：中断并返回由中断直接消除的 completion 数。
#
# “completion”不是 token、tool call 或模型回答数，而是 Control 正在等待后端结算的
# 工作单元。API 后端每条独立 run 新增 1；Codex 若消息 steer 进当前活动 turn，新增
# 0，若创建新 turn 则新增 1。这个抽象让同一控制器适配两种完全不同的会话模型。
#
# ``_outstanding_completions`` 与 ``planner_activity`` 的关系：
#
# * 大于 0：至少有一个后端工作单元未完成，activity 应为 busy；
# * 等于 0：没有活动工作，activity 可转为 idle，并立即 flush 等待消息；
# * interaction ended：会话已封存，不再提交或确认消息。
#
# 消息状态大致按以下路径转换：
#
#   pending -> sending -> sent
#                    \-> failed
#                    \-> unsent（延迟 ACK 的消息尚未开始便被中断）
#
# ``claim_next_pending_message`` 完成 pending -> sending。立即 ACK 模式在 driver.submit
# 成功后马上调用 ``message_started``；延迟 ACK 模式由 API driver 真正从内部队列取出
# prompt 时回调 ``message_started``。因此“已进入 driver 队列”与“已开始执行”不会在
# UI 上混为一谈。
#
# 外部控制的处理优先级固定为：
#
#   ended -> task replacement -> Esc interrupt -> idle message flush
#
# 新任务替换比普通 Esc 更高，因为旧 TaskRun 必须尽快让出环境，且替换完成后不能再
# flush 旧会话消息。二者都先取消 Toolkit，再中断 Planner，防止模型会话已结束但同步
# 机器人动作仍在后台推进环境。
#
# ``_lock`` 只保护 Control 自己的异步状态转换，避免 run 完成回调、工具完成回调与
# 主控制循环同时 claim/flush 消息。它不能保护阻塞 Toolkit，所以后者必须通过
# ``asyncio.to_thread`` 运行，避免占住 event loop。
#
# ``interaction_version`` 是单调版本号。主循环每次先处理当前快照，再在线程中等待版本
# 变化；即使状态通知发生在进入 wait 之前，版本差也能防止丢失唤醒。
# =============================================================================


class DashboardPlannerControl:
    """Coordinate queued input, interrupts, and task replacement."""

    def __init__(
        self,
        *,
        interaction: DashboardInteractionPort,
        cancel_active_and_wait: Callable[[], None],
        emit_user: Callable[[str], None],
        emit_initial_user: Callable[[], None],
        defer_message_ack: bool = False,
    ) -> None:
        # DashboardState 暴露的窄接口：消息状态、控制请求、activity 和版本等待。
        self._interaction = interaction
        # Toolkit 的同步取消入口；可能等待 VLA/环境动作到达安全停止边界。
        self._cancel_active_and_wait = cancel_active_and_wait
        # 把已经真正提交的后续用户消息写入 transcript / Dashboard event stream。
        self._emit_user = emit_user
        # 初始 prompt 的展示方式由各 Planner 决定，避免 Control 了解后端 transcript。
        self._emit_initial_user = emit_initial_user
        # True：driver 真正开始消息时再 ACK；False：submit 成功后立即 ACK。
        self._defer_message_ack = defer_message_ack
        # 串行化 _process、complete、tool_completed 中的 claim/submit/计数转换。
        self._lock = asyncio.Lock()
        # 已提交但尚未由 complete 或 interrupt 结算的后端工作单元数量。
        self._outstanding_completions = 0

    async def start(self) -> None:
        """Open Dashboard input after the initial backend submission succeeds."""
        # 调用 start 前，driver 已成功提交初始 prompt，因此预先登记一个 completion。
        self._outstanding_completions = 1
        # 初始请求正在执行，状态进入 busy；若任务已被替换，则不要再开放输入。
        self._interaction.set_planner_activity(
            "busy",
            accepting_input=not self._interaction.task_replacement_requested,
        )
        # 只有确认初始后端提交成功后才展示初始消息，避免前端出现假启动记录。
        self._emit_initial_user()

    async def run(self, driver: Any) -> None:
        """Forward Dashboard commands until the interaction ends."""
        # 记录已经观察到的状态版本；后续 wait 只等待比它更新的交互变化。
        version = self._interaction.interaction_version
        while self._interaction.planner_activity != "ended":
            # 先消费当前快照中的替换、中断或 pending message，避免不必要地先睡眠。
            await self._process(driver)
            # wait_for_interaction_change 使用 Condition 等阻塞原语，必须移出 event loop。
            version = await asyncio.to_thread(
                self._interaction.wait_for_interaction_change,
                version,
            )

    async def complete(self, driver: Any) -> None:
        """Record one completed backend request and flush queued input."""
        # 后端 run/turn 自然完成时调用；与主循环和工具完成回调互斥更新计数。
        async with self._lock:
            if self._interaction.planner_activity == "ended":
                # 会话已封存时，迟到的完成通知不能重新打开输入或提交旧消息。
                return
            # 防御性 max(0)：重复/迟到回调不能让 completion 计数变成负数。
            self._outstanding_completions = max(0, self._outstanding_completions - 1)
            self._interaction.set_planner_activity(
                "busy" if self._outstanding_completions else "idle"
            )
            # 完成边界也是安全提交点；若有排队消息，立即交给 driver。
            await self._flush(driver)

    async def tool_completed(self, driver: Any) -> None:
        """Flush input queued while the backend was running a tool."""
        # 工具结果已经形成，call/result 协议完整，此时可以安全注入或开启下一 run。
        async with self._lock:
            await self._flush(driver)

    def end(self) -> None:
        """Seal the interaction and preserve unfinished messages as unsent."""
        # seal_interaction 负责关闭输入，并把尚未真正执行的消息保留为可观察状态。
        self._interaction.seal_interaction()

    def message_started(self, message_id: str, text: str) -> None:
        """Acknowledge one deferred submission when execution starts."""
        # sending -> sent；先提交状态，再发布 transcript，保持前端状态与事件顺序一致。
        self._interaction.mark_message_sent(message_id)
        self._emit_user(text)

    def message_discarded(self, message_id: str) -> None:
        """Restore one deferred submission that never started."""
        # API driver 在中断时用它恢复已 claim、但尚未真正开始的内部排队消息。
        if self._interaction.planner_activity != "ended":
            self._interaction.mark_message_unsent(message_id)

    async def cancel_active_toolkit(self) -> None:
        """Cancel and drain the active toolkit operation off the event loop."""
        # cancel_active_and_wait 是同步阻塞函数；放到线程中避免 Dashboard 控制循环冻结。
        await asyncio.to_thread(self._cancel_active_and_wait)

    async def _process(self, driver: Any) -> None:
        # 同一时刻只允许一个控制路径 claim 请求、修改计数或 flush 消息。
        async with self._lock:
            if self._interaction.planner_activity == "ended":
                return

            # 新任务替换优先于普通 Esc 和消息提交。旧会话必须停止，不能恢复为 idle。
            if self._interaction.task_replacement_requested:
                try:
                    # 先停止可能正在执行的物理动作，再中断模型 run/turn。
                    await self.cancel_active_toolkit()
                    await driver.interrupt()
                except Exception as exc:
                    # 替换请求必须得到完成通知；错误作为状态保存，避免 Controller 永久等候。
                    self._interaction.complete_task_replacement(
                        error=f"planner interrupt failed: {_exception_text(exc)}"
                    )
                else:
                    self._interaction.complete_task_replacement()
                # 无论替换成功或失败，都不能继续处理旧会话的 Esc/pending message。
                return

            # claim 是原子“认领”操作，确保同一个 Esc 请求只由一次循环处理。
            if self._interaction.claim_interrupt_request():
                try:
                    await self.cancel_active_toolkit()
                    # driver 返回中断直接消除的 completion 数，而不是布尔成功标志。
                    completed = await driver.interrupt()
                    self._outstanding_completions = max(
                        0, self._outstanding_completions - completed
                    )
                except Exception as exc:
                    # 记录错误并结束该次中断请求；不让前端永远停留在 interrupting。
                    self._interaction.complete_interrupt(error=_exception_text(exc))
                else:
                    self._interaction.complete_interrupt()
                    self._interaction.set_planner_activity(
                        "busy" if self._outstanding_completions else "idle"
                    )
                    # 中断后若会话仍可用，尝试发送未被丢弃或新到达的排队消息。
                    await self._flush(driver)
                return

            # busy 时不能无条件提交：应等待后端 complete/tool_completed 给出安全边界。
            if self._interaction.planner_activity == "idle":
                await self._flush(driver)

    async def _flush(self, driver: Any) -> None:
        # claim 完成 pending -> sending；无可用消息或替换已挂起时返回 None。
        message = self._interaction.claim_next_pending_message()
        while message is not None and not self._interaction.task_replacement_requested:
            try:
                if self._defer_message_ack:
                    # API driver 先将完整 DashboardMessage 放入内部队列，真正开始时再 ACK。
                    added_completions = await driver.submit_dashboard_message(message)
                else:
                    # Codex 可直接 steer 当前 turn，故只需传递文本并立即确认提交。
                    added_completions = await driver.submit(message.text)
            except Exception as exc:
                # 单条消息提交失败不应阻止后续消息；记录 sending -> failed 后继续 claim。
                self._interaction.mark_message_failed(
                    message.message_id,
                    _exception_text(exc),
                )
            else:
                if not self._defer_message_ack:
                    # 非延迟模式中，driver.submit 成功即表示消息已进入后端会话。
                    self.message_started(message.message_id, message.text)
                # API 通常增加 1；Codex steer 增加 0、新 turn 增加 1。
                self._outstanding_completions += added_completions
                # 即使 steer 返回 0，当前 Codex turn 本身仍通常有既存 completion；
                # 将状态设为 busy 也能阻止 idle 分支在无安全边界时重复 flush。
                self._interaction.set_planner_activity("busy")

            # 循环批量处理当前可认领消息；InteractionPort 负责顺序和替换门禁。
            message = self._interaction.claim_next_pending_message()


# 把任意异常压缩成适合写入 Dashboard 状态的稳定单行文本。
# 状态层只保存类型名和消息，不保存异常对象或 traceback，确保后续 JSON/SSE
# 序列化安全；完整堆栈应由调用边界的日志负责记录。
def _exception_text(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"
