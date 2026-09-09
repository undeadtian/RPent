"""Thread-safe in-memory state for dashboard live runs."""

from __future__ import annotations

import re
import threading
import uuid
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from rpent.dashboard.events import (
    DashboardEvent,
    RunStartedEvent,
    RuntimeStatusEvent,
    StepRecordEvent,
    ToolResultEvent,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import (
    DashboardInteractionError,
    DashboardMessage,
    DashboardMessageConflictError,
    InteractionUnavailableError,
    PlannerActivity,
    UnknownDashboardMessageError,
)

if TYPE_CHECKING:
    # 仅供静态类型检查；运行时不导入环境状态模块，避免把可选环境依赖带入 Dashboard。
    from rpent.tools.state import EnvState, StepRecord

# =============================================================================
# DashboardState 中文导读
# =============================================================================
#
# 本模块是 Dashboard 后端的“线程安全内存投影”。它既不执行机器人任务，也不直接
# 驱动 Planner，而是把 SessionController、Planner bridge、Toolkit、Env runtime
# 产生的状态汇总为浏览器可以读取的 JSON、图片和视频索引。主要数据流如下：
#
#   浏览器 HTTP 请求 ──> 任务命令 / 普通消息 / 撤回 / 中断
#                                │
#                                ▼
#                       DashboardState（本模块）
#                         ▲        ▲        │
#                         │        │        ▼
#                   Controller  Planner   snapshot / SSE / media API
#                         ▲        ▲
#                         └── DashboardEvent（运行事件）
#
# 一、并发与锁
# ----------------
# DashboardState 会同时被 HTTP、Session Controller 和 Planner bridge 等线程访问。
# `_lock` 保护全部共享可变字段；`_condition` 复用同一把锁，并负责唤醒等待任务或
# 等待交互变化的线程。名称以 `_locked` 结尾的方法要求调用者已经持锁，不能在其
# 内部再次获取这把非重入 Lock。
#
# 会影响 Controller / Planner 等待条件的操作最终调用
# `_interaction_changed_locked()`：递增 `_interaction_version` 并 `notify_all()`。
# bridge 可以记录旧版本并调用 `wait_for_interaction_change()`，因此即使通知先于等待
# 发生，也能通过版本差检测到变化，避免单纯 Condition 通知可能产生的丢失唤醒。
#
# 返回消息时使用 `dataclasses.replace()`，快照中的 list / dict 也复制后再返回，避免
# 调用方在锁外直接修改内部对象。`env_state` 和文件读取部分只保留必要锁区间，避免
# 慢速磁盘 I/O 长时间阻塞状态查询。
#
# 二、Session 与 TaskRun 两层生命周期
# ----------------
# 一个 DashboardState 对应一个长生命周期 Session；Session 中串行运行多个 TaskRun。
# Session 级内容包括 run_id、共享 runtime 状态、任务代次和待执行命令；TaskRun 级
# 内容包括输出目录、timeline、frame、usage、消息、env_state 和终止信息。
# `_begin_task_locked()` 会重置后者，但保留 Session 级信息。
#
# 主要 Session 状态迁移：
#
#   starting_shared_services
#       ├─ shared_services_ready() ─> ready
#       └─ fail_session() ─────────> fatal
#
#   ready ── request_task() ─> task_starting ── RunStartedEvent ─> running
#     ▲                              │                              │
#     └──────── complete_task() <────┴──────────────────────────────┘
#
# 运行中提交新任务不会强杀当前线程，而是进入 `switch_pending` 并设置
# `_task_replacement_requested`。旧 Planner 在安全边界收尾，Controller 再认领新任务。
# `_pending_task` 只保留最后一次请求，即 last-write-wins：用户快速修改任务时，不会
# 依次执行已经过时的中间选择。
#
# `wait_for_task()` 真正认领任务时才增加 `_task_generation`，并生成
# `tasks/0001_<slug>` 形式的目录。slug 中的不安全字符会替换为下划线，空 slug 被拒绝。
# `complete_task()` 只结束当前 TaskRun，不结束 Session；若仍有 pending task，则回到
# task_starting，否则回到 ready。只把 scope="task" 的 runtime 组件复位，VLA/SAM3
# 等共享服务状态跨 TaskRun 保留。
#
# 三、命令解析与输入模式
# ----------------
# `submit_input()` 先用 dashboard_spec 中的 task schema 尝试 `_parse_task()`：命中
# 配置命令时校验字段数量、枚举 suggestions、整数格式和 minimum；其他普通文本进入
# Planner 对话。未知 `/rpent-*` 命令明确报错，并把原因写入 `_control_error`，供后续
# 前端快照展示。`_format_task()` 使用同一 schema 生成显示名称与输出目录 slug。
#
# `_input_mode_locked()` 将内部状态压缩为前端三态：
#
# - disabled：共享服务尚未启动完成，或 Session 已 fatal；
# - conversation：任务运行中、Planner 接受输入、且没有任务替换请求；
# - command_only：可以选择任务，但当前不能向 Planner 发送普通消息。
#
# 因而 ready 只表示“可提交任务命令”。Planner 初始 query 成功并调用
# `set_planner_activity(..., accepting_input=True)` 后，才开放 conversation。
#
# 四、Planner 活动和用户消息状态机
# ----------------
# Planner 活动为 starting / idle / busy / ended。ended 是当前 TaskRun 交互的封闭状态，
# 不能被重新打开；新 TaskRun 会重新初始化为 starting。
#
# 消息典型状态迁移：
#
#   pending ── claim_next_pending_message() ─> sending ─┬─> sent
#      │                                                ├─> failed
#      └─ withdraw_message() ─> withdrawn               └─> unsent
#
# 只有 pending 可撤回，只有 sending 可转为 sent / failed / unsent。前置状态不满足时抛
# DashboardMessageConflictError，防止 HTTP 与 bridge 的竞态导致重复提交。任务替换、
# 中断或 Planner ended 时停止认领新消息；封存交互时，未完成消息统一标为 unsent，
# 保留用户输入而不是静默丢弃。
#
# 五、中断是两阶段握手，不是强杀线程
# ----------------
# `request_interrupt()` 仅在 Planner busy 且任务未结束时接受请求，返回 accepted、
# duplicate 或 noop。bridge 再通过 `claim_interrupt_request()` 将请求标为 in-flight，
# 调用 SDK / Toolkit 的安全中断能力，最后用 `complete_interrupt()` 清除请求并记录可能
# 的错误。这样前端 Esc、后端实际处理和结果反馈都可见，同时避免重复发送中断。
#
# 六、结构化事件投影
# ----------------
# `emit()` 是统一 DashboardEventSink：
#
# - TranscriptEvent：追加前端 transcript payload；
# - UsageEvent：替换累计输入/输出 token 和工具调用数；
# - RuntimeStatusEvent：校验组件与 pending/starting/ready/failed 后更新状态；
# - ToolResultEvent：提取即时帧、工具日志、终止标志和动作视频；
# - StepRecordEvent：绑定 EnvState / artifact 上下文并投影规范环境步骤；
# - RunStartedEvent：把任务由 starting 标记为 running。
#
# 未知事件抛 TypeError，防止新增事件被静默忽略。runtime 和 frame 名称来自
# dashboard_spec，使通用 Dashboard 不需要硬编码具体机器人环境的组件和相机名称。
#
# 七、timeline、帧与视频
# ----------------
# `_apply_tool_result()` 可从即时工具结果构造 timeline；`on_step()` 从持久化 StepRecord
# 构造规范条目。条目保存 action、参数、结果、耗时、terminated / truncated 和动作
# 视频信息。`_terminated` / `_truncated` 是 timeline 的累计 OR，任务结束时还会重新
# 聚合，确保即使事件顺序不同也得到一致终态。
#
# 帧优先来自 StepRecord 的规范 artifact，也兼容工具返回的内存图片和旧路径字段。
# `_update_frames()` 拒绝比当前 `_frame_idx` 更旧的帧，避免并发结果晚到后让 Dashboard
# 画面倒退。每次更新替换整个 frame 字典，因此 `frame_available` 表示当前 step 实际
# 可用频道，而不是历史上曾出现过的频道。
#
# action 视频优先通过 EnvState 的 artifact 路径解析；否则使用工具结果记录的实际路径。
# episode 总视频只有在 TaskRun 到达终态且文件确实存在时才向前端报告 has_video。
#
# 八、前端快照边界
# ----------------
# `snapshot()` 用于 SSE 高频状态推送，只包含摘要、步骤数和媒体可用性；`run_detail()`
# 用于详情 API，额外返回完整 timeline；transcript 通过 `events_since(since)` 独立增量
# 获取，避免每个 SSE 快照重复传输不断增长的对话历史。`_visible_state_locked()` 将复杂
# Session 状态映射为旧前端兼容的 starting / running / succeeded / failed 等可见状态。
#
# 注意：本类只保存内存投影和受控输出路径，不负责持久化恢复；进程重启后的历史由日志、
# transcript、StepRecord 和 artifact 文件承担。

RUNTIME_STATUSES = {"pending", "starting", "ready", "failed"}
TERMINAL_RUN_STATES = {"succeeded", "failed", "cancelled"}
_PLANNER_ACTIVITIES = {"starting", "idle", "busy", "ended"}
InterruptRequestResult = Literal["accepted", "duplicate", "noop"]
InputMode = Literal["command_only", "conversation", "disabled"]
TaskRequest = dict[str, Any]
_INTEGER = re.compile(r"-?[0-9]+")
_UNSAFE_SLUG = re.compile(r"[^A-Za-z0-9_.-]+")


def _parse_task(task_spec: dict[str, Any], text: str) -> TaskRequest | None:
    # split 同时压缩连续空白；submit_input 已单独处理空字符串，因此 tokens[0] 安全。
    tokens = text.split()
    command = task_spec["command"]
    # `/rpent-*` 被保留为 Dashboard 本地命令命名空间。拼错命令时明确报错，不能把它
    # 当作自然语言误发给 Planner；普通非命令文本则通过下方 None 分支进入对话。
    if tokens[0].startswith("/rpent-") and tokens[0] != command:
        raise ValueError(f"unknown Dashboard command: {tokens[0]}")
    if tokens[0] != command:
        return None

    fields = task_spec["fields"]
    # 命令采用严格定长位置参数，不能静默忽略多余参数或用缺失值启动错误任务。
    if len(tokens) != len(fields) + 1:
        raise ValueError(f"expected {task_spec['usage']}")

    request: TaskRequest = {}
    for field, raw in zip(fields, tokens[1:], strict=True):
        name = field["name"]
        suggestions = field.get("suggestions", ())
        # suggestions 同时是前端补全数据和后端白名单，不能只依赖浏览器校验。
        if suggestions and raw not in suggestions:
            raise ValueError(f"unsupported {name}: {raw}")
        if field.get("kind") != "integer":
            # 非 integer 字段保留原始字符串，供环境插件自行解释。
            request[name] = raw
            continue
        # 先用完整正则拒绝 1.0、1x 等 int() 可能产生含糊错误信息的输入。
        if _INTEGER.fullmatch(raw) is None:
            raise ValueError(f"{name} must be an integer, got {raw!r}")
        value = int(raw)
        minimum = field.get("minimum")
        if minimum is not None and value < minimum:
            raise ValueError(f"{name} must be at least {minimum}, got {value}")
        request[name] = value
    # 返回值只包含 task_spec 声明字段，是后续目录模板和环境参数的统一来源。
    return request


def _format_task(task_spec: dict[str, Any], key: str, request: TaskRequest) -> str:
    # display 与 output_slug 共用 request 展开逻辑，避免 UI 标签和实际任务参数不一致。
    return task_spec[key].format(**request)


@dataclass(frozen=True, slots=True)
class ClaimedTask:
    """One last-write-wins task command claimed by the Session controller."""

    # number 在真正 claim 时单调递增，用于日志、UI 与磁盘目录稳定关联。
    number: int
    # request 是已通过 task_spec 校验的参数快照；新命令不会再改写已认领请求。
    request: TaskRequest
    # 每个 TaskRun 独占目录，形如 <session>/tasks/0001_<safe-slug>。
    output_dir: Path


class DashboardState:
    """Thread-safe projection for one sequential Dashboard Session."""

    @property
    def enabled(self) -> bool:
        # DashboardState 是真实事件接收器；Toolkit 可据此启用额外帧采集。
        return True

    def __init__(
        self,
        *,
        run_id: str,
        output_dir: str | Path,
        dashboard_spec: dict[str, Any],
    ) -> None:
        # root 是 Session 根目录；认领任务前 output_dir/video_path 暂指向该根目录。
        root = Path(output_dir)
        self.run_id = run_id
        self.output_dir = root
        self.video_path = root / "episode.mp4"
        self._session_root = root
        # dashboard_spec 是环境插件提供的声明式契约：任务命令、runtime 和相机通道均
        # 从这里派生，使通用状态层不硬编码 LIBERO 等具体环境名称。
        self._task_spec = dashboard_spec["task"]
        self._runtime_components = dashboard_spec["runtime_components"]
        self._frame_channels = dashboard_spec["frame_channels"]
        # 集合用于事件和媒体 API 的 O(1) 白名单校验。
        self._runtime_names = {
            component["name"] for component in self._runtime_components
        }
        self._frame_names = {channel["name"] for channel in self._frame_channels}

        # Condition 复用这把非重入锁。所有 `_locked` helper 均假定调用者已持锁，
        # 不能在其内部再次 `with self._lock`，否则会自锁。
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        # 以下从 TaskRun 终态到 frame_idx 都是当前任务的内存投影，新任务会统一重置。
        self._task_state: str | None = None
        self._terminated = False
        self._truncated = False
        self._error: str | None = None
        # UsageEvent 提供累计快照，因此这里保存最新值而不是事件增量。
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        # runtime 初始均 pending；scope=shared 的状态跨 TaskRun 保留。
        self._runtime = {
            component["name"]: {"status": "pending", "error": None}
            for component in self._runtime_components
        }
        # transcript 与 timeline 分开：前者按游标增量获取，后者由详情 API 整体读取。
        self._events: list[dict[str, Any]] = []
        self._timeline: list[dict[str, Any]] = []
        # _frames 只保存最新一步各通道 bytes；_frame_idx 用于拒绝乱序旧帧。
        self._frames: dict[str, bytes] = {}
        self._frame_idx = -1
        # EnvState 与 artifact 映射由 StepRecordEvent 成对绑定到当前 TaskRun。
        self.env_state: EnvState | None = None
        self.frame_artifacts: dict[str, str] = {}
        # Planner 活动、输入开关、中断和消息共同构成交互子状态机。
        self._accepting_input = False
        self._planner_activity: PlannerActivity = "starting"
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        # list 保留展示顺序，dict 支持按 ID 原子更新；二者引用同一 Message 对象。
        self._messages: list[DashboardMessage] = []
        self._messages_by_id: dict[str, DashboardMessage] = {}
        self._last_interaction_error: str | None = None
        # 单调版本配合 Condition 避免 bridge 等待时发生丢失唤醒。
        self._interaction_version = 0
        # 以下是跨 TaskRun 的 Session 调度状态；pending 任务采用 last-write-wins。
        self._session_state = "starting_shared_services"
        self._task_generation = 0
        self._pending_task: TaskRequest | None = None
        self._current_task: TaskRequest | None = None
        self._task_replacement_requested = False
        self._control_feedback: list[str] = []
        self._control_error: str | None = None
        # shutdown 只唤醒 SessionController，由控制器负责真实资源清理。
        self._shutdown_requested = False

    @property
    def session_state(self) -> str:
        with self._lock:
            return self._session_state

    @property
    def task_replacement_requested(self) -> bool:
        with self._lock:
            return self._task_replacement_requested

    def shared_services_ready(self) -> None:
        """Open the command channel after shared services have started."""
        with self._condition:
            if self._session_state == "fatal":
                # 启动失败可能与迟到的 ready 回调竞态；fatal 是不可恢复终态，不能覆盖。
                return
            # ready 仅开放任务命令；尚无 Planner，所以对话输入保持关闭且 activity=ended。
            self._session_state = "ready"
            self._planner_activity = "ended"
            self._accepting_input = False
            self._control_error = None
            self._interaction_changed_locked()

    def fail_session(self, error: BaseException | str) -> None:
        """Make a shared-service failure fatal for this in-memory Session."""
        with self._condition:
            # shared runtime 失败意味着整个 Session 无法再执行任务，清除尚未认领命令。
            self._session_state = "fatal"
            self._error = str(error)
            self._control_error = str(error)
            self._pending_task = None
            self._task_replacement_requested = False
            # 封存消息和中断状态，但资源停止仍由 SessionController 的 finally 负责。
            self._seal_interaction_locked()
            self._interaction_changed_locked()

    def request_shutdown(self) -> None:
        """Wake the controller so process-level cleanup can proceed."""
        with self._condition:
            # 标志本身不停止线程/daemon；wait_for_task 被唤醒后返回 None，控制器自然清理。
            self._shutdown_requested = True
            self._interaction_changed_locked()

    def submit_input(self, text: str) -> DashboardMessage | TaskRequest:
        """Route a local task command or a normal conversation message."""
        if not isinstance(text, str) or not text.strip():
            # 空输入沿用普通消息校验路径，由 _submit_message 产生统一 ValueError。
            return self._submit_message(text)
        try:
            # task 命令只在本地更新调度状态，绝不会发送给模型。
            request = _parse_task(self._task_spec, text)
        except ValueError as exc:
            with self._condition:
                # 解析错误同时反馈给 HTTP 调用方和 snapshot 中的 control_error。
                self._control_error = str(exc)
                self._interaction_changed_locked()
            raise
        if request is not None:
            self.request_task(request)
            return request
        # 不属于本地命令的文本才进入 Planner 消息状态机。
        return self._submit_message(text)

    def request_task(self, request: TaskRequest) -> None:
        """Atomically record the latest desired TaskRun and close old input."""
        with self._condition:
            if self._session_state == "fatal":
                raise InteractionUnavailableError("Dashboard Session is fatal")
            if self._session_state == "starting_shared_services":
                raise InteractionUnavailableError(
                    "Dashboard Session is still starting shared services"
                )

            # 每次覆盖而非追加 pending task，实现用户快速切换选择时 last-write-wins。
            self._pending_task = dict(request)
            self._control_error = None
            self._control_feedback = [
                f"Task selected: {_format_task(self._task_spec, 'display', request)}"
            ]

            active = self._task_state in {"starting", "running"}
            if active:
                # 不在 HTTP 线程强杀旧任务：标记 switch_pending，由 Planner 在工具安全
                # 边界收尾，再由 SessionController 完成旧 TaskRun 并认领最新请求。
                self._session_state = "switch_pending"
                self._task_replacement_requested = True
                self._accepting_input = False
                for message in self._messages:
                    if message.status == "pending":
                        # 尚未被 bridge claim 的旧任务消息保留文本但改为 unsent，防止误发。
                        message.status = "unsent"
                        message.error = None
            else:
                # 当前无活动 TaskRun，控制器可立即从 wait_for_task 认领。
                self._session_state = "task_starting"
            self._interaction_changed_locked()

    def wait_for_task(self, timeout: float | None = None) -> ClaimedTask | None:
        """Block until the controller can claim the latest pending task."""
        with self._condition:
            # wait_for 会在持锁下反复检查谓词；即使通知先于线程进入等待也不会漏掉。
            self._condition.wait_for(
                lambda: (
                    self._pending_task is not None
                    or self._shutdown_requested
                    or self._session_state == "fatal"
                ),
                timeout=timeout,
            )
            if self._shutdown_requested or self._session_state == "fatal":
                # None 是 SessionController 退出任务循环并清理共享资源的哨兵。
                return None
            if self._pending_task is None:
                # 超时且没有任务同样返回 None；调用方可选择结束或再次等待。
                return None

            # 在同一临界区完成“取出 + 清空”，确保一个请求最多被认领一次。
            request = self._pending_task
            self._pending_task = None
            # generation 在认领时而非提交时递增，被覆盖的中间请求不会消耗编号。
            self._task_generation += 1
            number = self._task_generation
            output_dir = (
                self._session_root
                / "tasks"
                / f"{number:04d}_{self._task_output_slug(request)}"
            )
            # 切换所有 TaskRun 级内存投影后，才把不可变 claim 交给控制器。
            self._begin_task_locked(request, number=number, output_dir=output_dir)
            return ClaimedTask(
                number=number,
                request=request,
                output_dir=output_dir,
            )

    def complete_task_replacement(self, error: str | None = None) -> None:
        """Seal the old planner at the current scheduling boundary."""
        with self._condition:
            if not self._task_replacement_requested:
                # 允许控制器幂等调用；没有替换请求时不得误封正常会话。
                return
            if error is not None:
                self._last_interaction_error = str(error)
            # 这里只封闭旧 Planner 输入；TaskRun 终态由后续 complete_task(cancelled) 设置。
            self._seal_interaction_locked()
            self._interaction_changed_locked()

    def complete_task(
        self,
        *,
        state: Literal["succeeded", "failed", "cancelled"],
        error: BaseException | str | None = None,
    ) -> None:
        """Finish only the current TaskRun and reopen the command channel."""
        if state not in TERMINAL_RUN_STATES:
            raise ValueError(f"invalid terminal run state: {state!r}")
        with self._condition:
            self._task_state = state
            # 从完整 timeline 重新聚合，而非只信任最后事件，兼容不同事件到达顺序。
            self._terminated = any(
                item.get("terminated") for item in self._timeline
            )
            self._truncated = any(
                item.get("truncated") for item in self._timeline
            )
            self._error = None if error is None else str(error)
            self._task_replacement_requested = False
            # TaskRun 完成后 conversation 永久关闭；下一任务会重新初始化 Planner 状态。
            self._seal_interaction_locked()
            # 只把 task scope runtime 复位，VLA/SAM3 等共享服务保持 ready。
            self._reset_task_runtime_locked()
            # 如果运行期间已有新选择，立即暴露 task_starting；否则回到命令 ready。
            self._session_state = (
                "task_starting" if self._pending_task is not None else "ready"
            )
            if error is not None:
                self._control_error = str(error)
            self._interaction_changed_locked()

    def _begin_task_locked(
        self,
        request: TaskRequest,
        *,
        number: int,
        output_dir: Path,
    ) -> None:
        # 本方法由 wait_for_task 持 Condition 锁调用：保留 Session 级 run_id、generation、
        # shared runtime 与 feedback，只重置下面的 TaskRun 级投影。
        self._current_task = request
        self.output_dir = output_dir
        self.video_path = output_dir / "episode.mp4"
        self._session_state = "task_starting"
        self._task_state = "starting"
        self._task_replacement_requested = False
        # 清空上一任务终态、错误和累计用量。
        self._terminated = False
        self._truncated = False
        self._error = None
        self._usage = {"in": 0, "out": 0, "tool_calls": 0}
        self._reset_task_runtime_locked()
        # transcript、timeline 和最新帧均按 TaskRun 隔离，避免前端混入上一任务数据。
        self._events = []
        self._timeline = []
        self._frames = {}
        self._frame_idx = -1
        # 新环境尚未发布 StepRecordEvent，先解除旧 EnvState/artifact 绑定。
        self.env_state = None
        self.frame_artifacts = {}
        # Planner bridge 尚未完成首个 query，暂不接受 conversation 消息。
        self._accepting_input = False
        self._planner_activity = "starting"
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        self._messages = []
        self._messages_by_id = {}
        self._last_interaction_error = None
        self._control_error = None
        self._control_feedback.append(f"TaskRun {number:04d} starting…")
        self._interaction_changed_locked()

    def _task_output_slug(self, request: TaskRequest) -> str:
        raw = _format_task(self._task_spec, "output_slug", request)
        # 目录名仅保留跨平台安全字符；连续不安全字符折叠成单个下划线。
        slug = _UNSAFE_SLUG.sub("_", raw).strip("._")
        if not slug:
            # 防止模板只含符号时退化为 tasks/<number>_ 这类无意义目录。
            raise ValueError("Dashboard task output slug must not be empty")
        return slug

    @property
    def planner_activity(self) -> PlannerActivity:
        """Return the current planner input activity."""
        with self._lock:
            return self._planner_activity

    @property
    def interaction_version(self) -> int:
        """Monotonic version for bridges waiting on interaction changes."""
        with self._lock:
            return self._interaction_version

    def set_planner_activity(
        self,
        activity: PlannerActivity,
        *,
        accepting_input: bool | None = None,
    ) -> None:
        """Update planner activity from the owning planner bridge.

        The bridge sets ``accepting_input=True`` only after the initial
        ``query()`` succeeds. Once ended, a Session cannot be reopened.
        """
        if activity not in _PLANNER_ACTIVITIES:
            # 即使类型标注为 Literal，运行时仍需保护来自插件/动态调用的非法字符串。
            raise ValueError(f"unknown planner activity: {activity!r}")
        with self._condition:
            if self._planner_activity == "ended" and activity != "ended":
                # ended 是当前 TaskRun 的单向终态；只有 _begin_task_locked 能创建新状态机。
                raise InteractionUnavailableError("Dashboard Session has ended")
            if activity == "ended":
                # 统一走 seal，确保输入、中断和未完成消息同时收口。
                self._seal_interaction_locked()
                self._interaction_changed_locked()
                return
            changed = self._planner_activity != activity
            self._planner_activity = activity
            if accepting_input is not None:
                requested_accepting = bool(accepting_input)
                if self._task_state in TERMINAL_RUN_STATES:
                    # 迟到的 bridge 回调不能在 TaskRun 已完成后重新开放输入。
                    requested_accepting = False
                changed = changed or self._accepting_input != requested_accepting
                self._accepting_input = requested_accepting
            if changed:
                # 无实际字段变化时不递增版本，避免无意义唤醒 bridge。
                self._interaction_changed_locked()

    def _submit_message(self, text: str) -> DashboardMessage:
        """Create one pending user message and notify the owning bridge."""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("message text must not be blank")
        with self._condition:
            if (
                not self._accepting_input
                or self._planner_activity == "ended"
                or self._task_state in TERMINAL_RUN_STATES
            ):
                # 三重检查防止启动阶段、封存后或迟到 HTTP 请求进入旧 Planner。
                raise InteractionUnavailableError(
                    "Dashboard Session is not accepting input"
                )
            message = DashboardMessage(
                # UUID 使并发 HTTP 请求拥有稳定独立 ID；状态初始为等待 bridge 的 pending。
                message_id=f"msg_{uuid.uuid4().hex}",
                text=text,
                status="pending",
            )
            # list 用于有序 UI 快照，dict 用于后续 ACK 按 ID 常数时间定位。
            self._messages.append(message)
            self._messages_by_id[message.message_id] = message
            self._interaction_changed_locked()
            # 返回浅拷贝，禁止调用者在锁外修改内部消息对象。
            return replace(message)

    def withdraw_message(self, message_id: str) -> DashboardMessage:
        """Atomically withdraw a message that is still pending."""
        with self._condition:
            message = self._message_locked(message_id)
            if message.status != "pending":
                # sending 表示 bridge 已占有提交权，此时撤回会与模型请求产生竞态。
                raise DashboardMessageConflictError(
                    f"message is {message.status}, not pending"
                )
            message.status = "withdrawn"
            message.error = None
            self._interaction_changed_locked()
            return replace(message)

    def claim_next_pending_message(self) -> DashboardMessage | None:
        """Claim one message so task replacement can stop later sends."""
        with self._condition:
            if (
                self._planner_activity == "ended"
                or self._task_replacement_requested
                or self._interrupt_requested
            ):
                # 封存、任务替换或中断期间均暂停认领；已发送消息不受影响。
                return None
            # 按提交顺序寻找首条 pending；withdrawn/failed/unsent 不会自动重试。
            message = next(
                (item for item in self._messages if item.status == "pending"),
                None,
            )
            if message is None:
                return None
            # sending 是 bridge 的独占 claim，后续只能转 sent/failed/unsent。
            message.status = "sending"
            message.error = None
            self._interaction_changed_locked()
            return replace(message)

    def mark_message_sent(self, message_id: str) -> DashboardMessage:
        """Commit one successfully queried message."""
        with self._condition:
            # 公共 transition helper 强制前置状态必须为 sending，阻止重复 ACK。
            message = self._transition_sending_message_locked(
                message_id,
                status="sent",
                error=None,
            )
            self._interaction_changed_locked()
            return replace(message)

    def mark_message_failed(
        self,
        message_id: str,
        error: str,
    ) -> DashboardMessage:
        """Record one failed query without retrying it."""
        error_text = str(error)
        with self._condition:
            # failed 记录提交根因但不自动重新排队，由用户决定是否重发。
            message = self._transition_sending_message_locked(
                message_id,
                status="failed",
                error=error_text,
            )
            self._interaction_changed_locked()
            return replace(message)

    def mark_message_unsent(self, message_id: str) -> DashboardMessage:
        """Restore one queued backend submission that never started."""
        with self._condition:
            # unsent 表示已 claim 但实际 run 未开始，文本保留且错误清空。
            message = self._transition_sending_message_locked(
                message_id,
                status="unsent",
                error=None,
            )
            self._interaction_changed_locked()
            return replace(message)

    def request_interrupt(self) -> InterruptRequestResult:
        """Record an Esc request without waiting for the planner backend."""
        with self._condition:
            if self._interrupt_requested:
                # pending 和 in-flight 都算同一未完成请求，前端重复 Esc 不重复触发取消。
                return "duplicate"
            if (
                self._planner_activity != "busy"
                or self._task_state in TERMINAL_RUN_STATES
            ):
                # idle/starting 没有正在执行的后端工作；终态任务也无需中断。
                return "noop"
            # HTTP 请求只登记意图并立即返回；Planner bridge 稍后 claim 后实际调用 SDK。
            self._interrupt_requested = True
            self._interrupt_in_flight = False
            self._interaction_changed_locked()
            return "accepted"

    def claim_interrupt_request(self) -> bool:
        """Claim a queued interrupt while keeping it visibly requested."""
        with self._condition:
            if not self._interrupt_requested or self._interrupt_in_flight:
                # False 区分“无请求”或“已有处理者”，两种情况都不能再次执行取消。
                return False
            # requested 保持 True 供 UI 显示；in_flight 防止多个 bridge 重复处理。
            self._interrupt_in_flight = True
            self._interaction_changed_locked()
            return True

    def complete_interrupt(self, error: str | None = None) -> None:
        """Complete the claimed SDK interrupt, successfully or with an error."""
        with self._condition:
            if not self._interrupt_requested or not self._interrupt_in_flight:
                # 强制 request -> claim -> complete 次序，暴露错误的 bridge 调用。
                raise DashboardInteractionError("no interrupt request is in flight")
            self._interrupt_requested = False
            self._interrupt_in_flight = False
            if error is not None:
                # 中断失败不直接终止 TaskRun，仅在 interaction.last_error 中反馈。
                self._last_interaction_error = str(error)
            else:
                self._last_interaction_error = None
            self._interaction_changed_locked()

    def seal_interaction(self) -> None:
        """End input and preserve every unfinished message as ``unsent``."""
        with self._condition:
            self._seal_interaction_locked()
            self._interaction_changed_locked()

    def wait_for_interaction_change(
        self,
        since: int,
        timeout: float | None = None,
    ) -> int:
        """Wait for any interaction state change and return its latest version."""
        with self._condition:
            # bridge 先记录 since。wait_for 在每次唤醒后重新检查版本；若变化早于进入
            # wait，谓词会立即为真，因此不会出现只依赖 notify 的“丢失唤醒”。
            self._condition.wait_for(
                lambda: self._interaction_version != since,
                timeout=timeout,
            )
            # 超时可能返回原版本，调用方据此判断是否真的发生交互变化。
            return self._interaction_version

    def emit(self, event: DashboardEvent) -> None:
        """Project one structured event into the existing frontend state."""
        if isinstance(event, TranscriptEvent):
            with self._lock:
                # payload 已由 Planner 转成前端格式；这里只按到达顺序 append。
                self._events.append(event.payload)
            return
        if isinstance(event, UsageEvent):
            with self._lock:
                # UsageEvent 是累计快照，因此整体替换而不是 +=，重复事件也保持幂等。
                self._usage = {
                    "in": int(event.inp),
                    "out": int(event.out),
                    "tool_calls": int(event.tool_calls),
                }
            return
        if isinstance(event, RuntimeStatusEvent):
            # runtime helper 负责白名单校验和错误字符串化。
            self._apply_runtime_status(event)
            return
        if isinstance(event, ToolResultEvent):
            # 兼容直接工具结果投影，可提取内存帧和 legacy timeline。
            self._apply_tool_result(event)
            return
        if isinstance(event, StepRecordEvent):
            # record、EnvState 与 artifact 映射必须作为同一事件切换，避免跨 TaskRun 误读。
            self.env_state = event.env_state
            self.frame_artifacts = dict(event.frame_artifacts)
            self.on_step(event.record)
            return
        if isinstance(event, RunStartedEvent):
            # 纯信号：runtime 初始化及启动前竞态检查完成，TaskRun 正式 running。
            self._start()
            return
        # 新增事件必须显式实现投影，不能被静默忽略导致前后端状态漂移。
        raise TypeError(f"unsupported dashboard event: {type(event).__name__}")

    def _apply_runtime_status(self, event: RuntimeStatusEvent) -> None:
        if event.component not in self._runtime_names:
            # 拒绝 spec 外组件，避免拼写错误在快照中生成前端不会渲染的幽灵条目。
            raise ValueError(f"unknown runtime component: {event.component!r}")
        if event.status not in RUNTIME_STATUSES:
            raise ValueError(f"unknown runtime status: {event.status!r}")
        with self._lock:
            self._runtime[event.component] = {
                "status": event.status,
                # 异常对象只在进程内传输，到 JSON 投影边界统一字符串化。
                "error": None if event.error is None else str(event.error),
            }

    def _runtime_snapshot(self) -> dict[str, dict[str, str | None]]:
        """Return a detached copy of runtime status for a locked caller."""
        # 外层与每个状态内层 dict 都复制，防止响应序列化方改动内部 runtime。
        return {component: dict(status) for component, status in self._runtime.items()}

    def _reset_task_runtime_locked(self) -> None:
        for component in self._runtime_components:
            if component.get("scope", "shared") != "task":
                # shared 组件由 SessionController 启动一次，跨任务保留 ready/failed 状态。
                continue
            # env 等 task scope 服务会为下一 TaskRun 重建，先恢复 pending。
            self._runtime[component["name"]] = {
                "status": "pending",
                "error": None,
            }

    def _apply_tool_result(self, event: ToolResultEvent) -> None:
        name = event.name
        result = event.result
        if not isinstance(result, dict):
            # ToolResultEvent 允许环境自定义 Any；通用投影只理解 dict，其他类型安全忽略。
            return
        frames = {
            # 主相机优先新字段，_image_bytes 仅用于兼容旧工具返回协议。
            "camera": result.get("_image_cam_bytes") or result.get("_image_bytes"),
            "wrist": result.get("_image_wrist_bytes"),
        }
        # 即使缺少 log，内存帧仍可先独立更新。
        self._update_frames(
            step=result.get("step"),
            frames={kind: data for kind, data in frames.items() if data},
        )
        log = result.get("log")
        if not isinstance(log, dict):
            # 无规范 log 就无法可靠构造 action/args/result timeline，但帧已保留。
            return
        command = log.get("command")
        if not isinstance(command, dict) or command.get("action") != name:
            # 校验事件工具名与嵌入 command 一致，防止错配结果污染 timeline。
            return
        try:
            step = int(result["step"])
        except Exception:
            # timeline 要求稳定整数 step；缺失或非法时只保留前面的帧投影。
            return
        action = str(command.get("action", name))
        terminated = bool(result.get("terminated"))
        truncated = bool(result.get("truncated"))
        action_video_path = self._action_video_from_result(
            result,
            step=step,
            action=action,
        )
        action_video = str(action_video_path) if action_video_path is not None else None
        action_video_artifact = result.get("action_video_artifact")
        # timeline 只存 JSON-friendly 摘要和媒体索引，不复制原始图片 bytes。
        item = {
            "step": step,
            "action": action,
            "args": {k: v for k, v in command.items() if k != "action"},
            "result": log.get("result"),
            "elapsed_s": log.get("elapsed_s"),
            "terminated": terminated,
            "truncated": truncated,
            "action_video_path": action_video,
            "action_video_artifact": action_video_artifact,
            "has_action_video": bool(action_video_artifact or action_video_path),
        }
        with self._lock:
            self._timeline.append(item)
            # 终止标志采用累计 OR，后续普通步骤不能把已发生终态改回 False。
            self._terminated = self._terminated or terminated
            self._truncated = self._truncated or truncated

    def _action_video_from_result(
        self,
        result: dict[str, Any],
        *,
        step: int,
        action: str,
    ) -> Path | None:
        raw_path = result.get("action_video_path")
        if not raw_path:
            # 旧工具可能不显式返回路径，按历史目录约定推导默认文件名。
            raw_path = f"action_videos/step_{step:02d}_{action}.mp4"
        try:
            path = self._resolve_output_path(raw_path)
        except TypeError:
            # Path 不接受的工具返回值视为无视频，而不是让 Dashboard 投影失败。
            return None
        # 只向 timeline 声明当前真实存在的文件，避免前端请求必然 404。
        return path if path.exists() else None

    def on_step(self, record: StepRecord) -> None:
        """Project one recorded environment step into frames and timeline."""
        # 先从 record 的规范 artifact 更新帧；即使该步骤没有 action，也可刷新画面。
        self._update_step_frames(record)
        command = record.command
        if not isinstance(command, dict) or not command.get("action"):
            # 无动作命令的状态记录不进入动作 timeline。
            return
        # artifact 名排序后取首个 mp4，保证集合/字典来源下选择结果稳定。
        action_video = next(
            (name for name in sorted(record.artifacts) if name.endswith(".mp4")),
            None,
        )
        item = {
            "step": record.step_idx,
            "action": str(command.get("action")),
            "args": {key: value for key, value in command.items() if key != "action"},
            "result": record.result,
            "elapsed_s": record.elapsed_s,
            "terminated": record.terminated,
            "truncated": record.truncated,
            "action_video_artifact": action_video,
            "has_action_video": action_video is not None,
        }
        with self._lock:
            self._timeline.append(item)
            # 与即时 ToolResult 投影相同，终止信息在 TaskRun 内单调累积。
            self._terminated = self._terminated or record.terminated
            self._truncated = self._truncated or record.truncated

    def _update_step_frames(self, record: StepRecord) -> None:
        """Load dashboard frame bytes from the step's canonical artifacts."""
        env_state = self.env_state
        if env_state is None:
            # 尚未通过 StepRecordEvent 绑定状态仓库时无法解析 artifact。
            return
        frames: dict[str, bytes] = {}
        for kind, artifact in self.frame_artifacts.items():
            if kind not in self._frame_names or artifact not in record.artifacts:
                # 同时要求合法前端通道和当前 step 已登记 artifact，禁止任意文件读取。
                continue
            try:
                frames[kind] = env_state.load_bytes(artifact, step=record.step_idx)
            except FileNotFoundError:
                # artifact 元数据与落盘短暂不同步时跳过该通道，不影响其余状态投影。
                continue
        self._update_frames(step=record.step_idx, frames=frames)

    def _resolve_output_path(self, value: Any) -> Path:
        path = Path(value)
        if path.is_absolute() or path.is_relative_to(self.output_dir):
            # 绝对路径和已经以当前 output_dir 为前缀的路径不重复拼接。
            return path
        # 普通相对路径解释为当前 TaskRun 输出目录内的 artifact。
        return self.output_dir / path

    def _apply_frame_paths(self, result: dict[str, Any]) -> None:
        projected = result.get("frames")
        # 新协议直接给 channel -> path；非 dict 时从空映射开始尝试 legacy 字段。
        frame_paths = dict(projected) if isinstance(projected, dict) else {}
        for channel in self._frame_channels:
            name = channel["name"]
            path_key = channel.get("legacy_path_key")
            if name in frame_paths or path_key is None:
                continue
            if path_key in result:
                frame_paths[name] = result[path_key]
        if not frame_paths:
            return
        frames: dict[str, bytes] = {}
        for kind, path in frame_paths.items():
            if kind not in self._frame_names:
                # 工具不能通过 result 动态注入 spec 未声明的媒体通道。
                continue
            if not path:
                continue
            try:
                # 文件 I/O 在 _update_frames 加锁前完成，避免阻塞 snapshot 和 HTTP 请求。
                frames[kind] = self._resolve_output_path(path).read_bytes()
            except (OSError, TypeError):
                # 单个路径损坏只丢弃该帧，不让工具 timeline 整体失败。
                continue
        self._update_frames(step=result.get("step"), frames=frames)

    def _update_frames(
        self,
        *,
        step: Any,
        frames: dict[str, bytes],
    ) -> None:
        try:
            frame_idx = int(step)
        except (TypeError, ValueError):
            # 无有效 step 的 legacy 结果仍可更新图像，但不能推进乱序游标。
            frame_idx = None
        with self._lock:
            if frame_idx is not None and frame_idx < self._frame_idx:
                # 并发工具结果可能晚到；拒绝旧 step，防止浏览器画面倒退。
                return
            # 整体替换使 frame_available 精确表示当前 step，而非历史通道并集。
            self._frames = dict(frames)
            if frame_idx is not None:
                self._frame_idx = frame_idx

    def _start(self) -> None:
        with self._condition:
            # RunStartedEvent 标志真正进入 Agent 执行，而非仅开始创建 runtime。
            self._task_state = "running"
            if not self._task_replacement_requested:
                self._session_state = "running"
            # 若启动窗口已收到替换请求，则保留 switch_pending，不能覆盖为 running。
            self._interaction_changed_locked()

    def _transition_sending_message_locked(
        self,
        message_id: str,
        *,
        status: Literal["sent", "failed", "unsent"],
        error: str | None,
    ) -> DashboardMessage:
        # 调用者必须持有 Condition 锁；先按 ID 定位，再强制唯一合法前置状态。
        message = self._message_locked(message_id)
        if message.status != "sending":
            raise DashboardMessageConflictError(
                f"message is {message.status}, not sending"
            )
        message.status = status
        message.error = error
        return message

    def _message_locked(self, message_id: str) -> DashboardMessage:
        try:
            return self._messages_by_id[message_id]
        except KeyError as exc:
            # 使用领域异常而不是泄漏 KeyError，便于 HTTP 层映射明确响应。
            raise UnknownDashboardMessageError(
                f"unknown Dashboard message: {message_id}"
            ) from exc

    def _seal_interaction_locked(self) -> None:
        # 封存是当前 TaskRun 的不可逆交互终点；新任务会在 _begin_task_locked 重建状态。
        self._planner_activity = "ended"
        self._accepting_input = False
        self._interrupt_requested = False
        self._interrupt_in_flight = False
        for message in self._messages:
            if message.status not in {"pending", "sending"}:
                # 已 sent/failed/withdrawn/unsent 的历史状态保持原样。
                continue
            # 未完成消息保留文本并标为 unsent，而不是在任务结束/替换时静默丢弃。
            message.status = "unsent"
            message.error = None

    def _interaction_changed_locked(self) -> None:
        # 版本先递增再广播；等待者醒来后通过版本差判断变化，抵抗虚假/提前唤醒。
        self._interaction_version += 1
        self._condition.notify_all()

    def _interaction_snapshot_locked(self) -> dict[str, Any]:
        return {
            "session_id": self.run_id,
            "input_mode": self._input_mode_locked(),
            "planner_activity": self._planner_activity,
            "interrupt_requested": self._interrupt_requested,
            # as_dict 为每条消息创建新 dict，前端序列化不会持有内部对象引用。
            "messages": [message.as_dict() for message in self._messages],
            "last_error": self._last_interaction_error,
        }

    def _input_mode_locked(self) -> InputMode:
        if self._session_state in {"starting_shared_services", "fatal"}:
            # runtime 未就绪或 fatal 时连任务命令也不可接受。
            return "disabled"
        if (
            self._session_state == "running"
            and self._accepting_input
            and not self._task_replacement_requested
        ):
            # conversation 需要运行中、bridge 显式开放输入且不存在任务切换。
            return "conversation"
        # ready/task_starting/switch_pending 等状态只允许本地任务命令。
        return "command_only"

    def _command_snapshot(self, request: TaskRequest | None) -> dict[str, Any] | None:
        if request is None:
            return None
        return {
            # 保留扁平字段兼容现有前端，同时提供 parameters 显式嵌套副本。
            **request,
            "parameters": dict(request),
            "label": _format_task(self._task_spec, "display", request),
        }

    def _session_fields_locked(self) -> dict[str, Any]:
        return {
            "session_state": self._session_state,
            "task_generation": self._task_generation,
            "current_task": self._command_snapshot(self._current_task),
            "pending_task": self._command_snapshot(self._pending_task),
            "control_feedback": list(self._control_feedback),
            "control_error": self._control_error,
        }

    def _visible_state_locked(self) -> str:
        if self._session_state == "fatal":
            # 前端旧状态协议没有 fatal，统一映射为 failed。
            return "failed"
        if self._session_state in {"starting_shared_services", "task_starting"}:
            # 共享服务和单任务 runtime 启动阶段都对前端显示 starting。
            return "starting"
        # running/终态优先使用 TaskRun 状态；无任务时回退到 Session 状态 ready 等。
        return self._task_state or self._session_state

    def events_since(self, since: int) -> list[dict[str, Any]]:
        with self._lock:
            # since 是前端已消费事件数；切片返回新 list，实现 transcript 增量传输。
            return list(self._events[since:])

    def frame(self, kind: str) -> bytes | None:
        if kind not in self._frame_names:
            # 媒体路由只允许 dashboard_spec 声明通道，避免任意键访问。
            raise ValueError(f"unknown frame kind: {kind!r}")
        with self._lock:
            # bytes 不可变，可直接返回；None 表示当前 step 不含该通道。
            return self._frames.get(kind)

    def action_video_path(self, step: int) -> Path | None:
        # EnvState 引用在锁外用于磁盘解析，避免文件操作长时间占用状态锁。
        env_state = self.env_state
        with self._lock:
            artifact = None
            raw_path = None
            for item in self._timeline:
                if int(item.get("step", -1)) != int(step):
                    continue
                # 只复制匹配项中的路径信息，退出锁后再检查文件系统。
                artifact = item.get("action_video_artifact")
                raw_path = item.get("action_video_path")
                break
        if artifact and env_state is not None:
            try:
                # 规范 StepRecord artifact 优先，由 EnvState 限定到对应 step 目录。
                path = env_state.artifact_path(artifact, step=int(step))
            except (LookupError, ValueError):
                return None
            return path if path.exists() else None
        if raw_path:
            # legacy ToolResult 已在投影时解析为路径；仍需确认文件尚存在。
            video_path = Path(raw_path)
            return video_path if video_path.exists() else None
        return None

    def has_video(self) -> bool:
        with self._lock:
            # 避免任务运行中读取仍在写入的 episode.mp4；终态且文件存在才公开。
            return self._task_state in TERMINAL_RUN_STATES and self.video_path.exists()

    def _frame_snapshot(self) -> tuple[int, dict[str, bool]]:
        # 调用者已持锁；按 spec 顺序生成所有通道布尔值，缺帧也显式返回 False。
        available = {
            channel["name"]: channel["name"] in self._frames
            for channel in self._frame_channels
        }
        return self._frame_idx, available

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            # snapshot 面向高频 SSE，仅返回 timeline 长度，不重复传输完整历史。
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._visible_state_locked(),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "error": self._error,
                # 对可变嵌套结构做副本，响应序列化不会修改内部投影。
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "has_video": (
                    self._task_state in TERMINAL_RUN_STATES and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "n_steps": len(self._timeline),
                "interaction": self._interaction_snapshot_locked(),
                **self._session_fields_locked(),
            }

    def run_info(self) -> dict[str, Any]:
        # 列表接口只需稳定 Session ID，详情由 snapshot/run_detail 分离提供。
        return {"id": self.run_id}

    def run_detail(self) -> dict[str, Any]:
        with self._lock:
            # 详情接口与 snapshot 使用同一状态字段，但额外复制完整 timeline。
            frame_idx, frame_available = self._frame_snapshot()
            return {
                "state": self._visible_state_locked(),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "error": self._error,
                "usage": dict(self._usage),
                "runtime": self._runtime_snapshot(),
                "timeline": list(self._timeline),
                "has_video": (
                    self._task_state in TERMINAL_RUN_STATES and self.video_path.exists()
                ),
                "frame_idx": frame_idx,
                "frame_available": frame_available,
                "interaction": self._interaction_snapshot_locked(),
                **self._session_fields_locked(),
            }
