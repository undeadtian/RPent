"""一个长生命周期 Dashboard Session 的串行 TaskRun 调度器。

Dashboard 的资源分为两个生命周期层级：

- **Session 级共享资源**：例如 VLA、SAM3 等加载昂贵、可被多个任务复用的服务，
  在控制器开始时启动一次，在整个 Session 退出时统一关闭；
- **TaskRun 级独占资源**：例如 env_server、Toolkit 和 Planner，由传入的
  ``run_task`` 回调为每个任务分别创建和清理。

本模块只负责两层生命周期之间的编排，不了解具体机器人环境，也不直接创建进程。
环境差异通过 ``start_shared`` 和 ``run_task`` 两个回调注入。任务始终串行执行，
避免多个智能体同时控制环境或竞争共享推理服务。
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from rpent.dashboard.state import ClaimedTask, DashboardState
from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    # 仅用于类型标注，运行时不导入进程管理实现，从而保持该调度模块轻量并避免
    # 可能的循环依赖。
    from rpent.utils.daemon import ProcessDaemon

# 使用独立日志命名空间，便于把 Session 调度和共享服务清理问题与单个 Agent
# TaskRun 的日志区分开来。
logger = get_logger("dashboard_session")


class DashboardSessionController:
    """管理共享服务，并且一次只执行一个已认领的 TaskRun。

    控制器本身是同步、单线程的。浏览器请求和 Planner 交互可以在其他线程中更新
    :class:`DashboardState`；控制器通过 ``wait_for_task`` 等状态方法进行同步，
    不直接操作锁、事件或任务队列。

    ``DashboardState`` 负责“最后写入优先”（last-write-wins）的任务命令语义：
    当前任务执行期间若用户选择了新任务，状态对象会标记替换请求；当前任务在安全
    边界结束后记为 ``cancelled``，下一次 ``wait_for_task`` 再认领最新任务。
    """

    def __init__(
        self,
        *,
        state: DashboardState,
        start_shared: Callable[
            [],
            tuple[list[ProcessDaemon], dict[str, Any]],
        ],
        run_task: Callable[
            [ClaimedTask, dict[str, Any]],
            BaseException | str | None,
        ],
    ) -> None:
        """保存 Session 状态和两个环境相关生命周期回调。

        Args:
            state: Dashboard 前后端共享状态。它提供 Session 状态迁移、阻塞式任务
                认领、任务替换标志和完成事件发布。
            start_shared: 无参数回调，在 Session 开始时调用一次。返回值包含：

                1. 需要在 Session 结束时停止的共享 daemon 列表；
                2. 传给每个 TaskRun 的共享 primitive 参数，例如 VLA/SAM3 客户端。

            run_task: 每认领一个任务调用一次。输入为任务描述和共享 primitive
                参数；成功返回 ``None``，失败返回异常对象或错误字符串。

        Note:
            回调由上层 CLI 注入，使本控制器不依赖 LIBERO、RoboCasa 等具体环境。
            ``run_task`` 应自行管理 TaskRun 级资源，但不应关闭 Session 共享 daemon。
        """
        self._state = state
        self._start_shared = start_shared
        self._run_task = run_task

    def run(self) -> None:
        """启动共享服务，串行消费任务命令，并在退出时释放共享资源。

        执行阶段如下：

        1. 调用 ``start_shared`` 创建 Session 级服务；
        2. 发布共享服务就绪状态，使前端允许提交任务；
        3. 阻塞等待并逐个执行最新认领的 TaskRun；
        4. 根据替换请求、错误或正常返回确定任务最终状态；
        5. Session 关闭后按创建顺序的逆序停止所有共享 daemon。

        该方法通常由 CLI 主线程调用，并持续运行到
        :meth:`DashboardState.wait_for_task` 返回 ``None``。
        """
        # 即使 start_shared 在返回前抛出异常，finally 也需要一个可安全遍历的列表。
        # 若回调成功返回，该列表就代表控制器拥有、必须负责停止的全部共享进程。
        shared_daemons: list[ProcessDaemon] = []
        try:
            # 内层 try 只处理共享运行时启动失败。此类失败意味着整个 Session 无法
            # 执行任何任务，因此应转为 fatal Session 状态并立即结束任务循环。
            try:
                shared_daemons, shared_primitives_kwargs = self._start_shared() # 调用init_shared_runtime
            except Exception as exc:
                # fail_session 会保存错误并通知前端；return 仍会经过外层 finally。
                # 注意：回调在成功返回前创建的部分资源应由回调自身回滚，因为控制器
                # 只有在完整返回后才能取得 shared_daemons 的所有权。
                self._state.fail_session(exc)
                return

            # 只有所有共享服务成功启动后才发布 ready。前端据此区分“Dashboard Web
            # 页面可访问”和“机器人推理运行时已可执行任务”这两个不同阶段。
            self._state.shared_services_ready()

            # wait_for_task 是阻塞式同步边界：有待执行任务时返回 ClaimedTask；
            # Session 收到关闭请求时返回 None，从而自然退出循环并进入共享资源清理。
            while True:
                claimed = self._state.wait_for_task()
                if claimed is None:
                    break

                # run_task 理论上把失败编码为返回值，但控制器仍捕获异常，防止环境
                # 插件的意外错误穿透并终止整个 Session，导致后续任务无法执行。
                try:
                    error = self._run_task(claimed, shared_primitives_kwargs)
                except Exception as exc:
                    error = exc

                # 最终状态的优先级有意设为：替换请求 > 执行错误 > 成功。
                # 用户已切换任务时，即使旧任务清理阶段同时报错，前端仍应把旧任务
                # 表示为 cancelled，而不是误导为用户主动请求之外的普通失败。
                if self._state.task_replacement_requested:
                    state = "cancelled"
                elif error:
                    state = "failed"
                else:
                    state = "succeeded"

                # complete_task 原子地发布最终状态、保存错误，并解除下一次任务认领的
                # 状态约束。控制器随后回到 wait_for_task，继续消费最新命令。
                self._state.complete_task(
                    state=state,
                    error=error,
                )
        finally:
            # 无论正常关闭、共享服务启动失败，还是循环中出现未预期异常，都必须回收
            # Session 级进程。倒序停止遵循“后创建先销毁”，适用于存在依赖关系的服务。
            cleanup_errors: list[str] = []
            for daemon in reversed(shared_daemons):
                try:
                    daemon.stop()
                except Exception as exc:
                    # 一个 daemon 停止失败不能阻止其余服务继续释放。错误先聚合，待
                    # 整个清理循环结束后统一记录，最大限度减少孤儿进程。
                    cleanup_errors.append(str(exc))

            # 清理发生在 Session 已退出的路径上，无法再作为某个 TaskRun 的错误；
            # 因此这里只写 warning，并保留所有失败信息供运维排查。
            if cleanup_errors:
                logger.warning(
                    "shared runtime cleanup failed: %s",
                    "; ".join(cleanup_errors),
                )
