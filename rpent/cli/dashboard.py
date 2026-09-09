"""``--dashboard`` 分支的长生命周期 Session 编排器。

普通 ``cli.main`` 一次只运行一个任务；Dashboard 则提前接管控制流，并把生命周期
拆成两层：

- Session 级：启动一次可复用的 VLA/SAM3 服务和 Web Dashboard；
- TaskRun 级：每次用户提交任务都创建全新的 env_server、Toolkit、Planner 和输出目录。

任务按顺序执行，避免多个 Agent 同时控制共享资源。每个 TaskRun 仍沿用普通 CLI 的
核心步骤：``parse_config -> get_toolkit -> build_planner -> solve -> cleanup``。
"""

from __future__ import annotations

import argparse
import copy
import json
import shlex
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.cli.main import _serialize_messages
from rpent.dashboard.events import RunStartedEvent
from rpent.envs import get_toolkit
from rpent.planner.base import build_planner
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

if TYPE_CHECKING:
    from rpent.dashboard.state import ClaimedTask, DashboardState
    from rpent.envs.env_spec import EnvSpec
    from rpent.utils.daemon import ProcessDaemon

# 使用与普通物理智能体相同的日志命名空间，使终端和文件日志可按 agent 统一过滤。
logger = get_logger("agent")


def run_dashboard_session(
    args: argparse.Namespace,
    env_spec: EnvSpec,
    *,
    parser: argparse.ArgumentParser,
) -> int:
    """运行一个长生命周期 Dashboard Session。

    该函数负责 Session 级资源，而不直接执行某个具体机器人任务。完整控制流为：

    1. 启动 Web Dashboard，并等待用户在启动页确认配置；
    2. 创建 Session 输出目录和前后端共享状态；
    3. 通过 :class:`DashboardSessionController` 启动共享服务；
    4. 串行接收和执行多个 TaskRun；
    5. 收到关闭请求或 ``Ctrl+C`` 后退出。

    Args:
        args: CLI 解析得到的参数。启动页确认后，部分字段会被原地更新。
        env_spec: 当前机器人环境的插件描述，提供 Dashboard 规格、提示词和
            Session/TaskRun 两级运行时初始化方法。
        parser: 顶层参数解析器。这里使用 ``parser.error`` 输出一致的 CLI
            错误格式并终止启动。

    Returns:
        正常结束时返回进程退出码 ``0``。
    """
    # 延迟导入 Dashboard 专用模块：普通非 Dashboard CLI 路径不需要加载
    # FastAPI、Web 状态机及 Session 控制器，也可避免不必要的导入副作用。
    from rpent.dashboard.launcher import apply_to_args, defaults_from_args
    from rpent.dashboard.server import DashboardServer
    from rpent.dashboard.session import DashboardSessionController
    from rpent.dashboard.state import DashboardState
    from rpent.utils.config import get_repo_root

    # 环境插件必须显式提供 DashboardSpec。它定义前端可选任务参数、运行时
    # 状态组件、相机频道等；没有该规格就无法构建对应控制界面。
    dashboard_spec = env_spec.dashboard
    if dashboard_spec is None:
        parser.error(
            f"environment {env_spec.name!r} does not support Dashboard control"
        )

    # 阶段 1：先启动 Web 服务，再阻塞等待启动页提交。
    #
    # 此时机器人共享服务尚未启动，因此用户可以先在浏览器修改模型、任务步数、
    # CUDA 设备等 Session 配置，避免每改一次配置都加载/卸载大模型。
    dashboard_server = DashboardServer(
        host=args.dashboard_host,
        port=args.dashboard_port,
        language=args.dashboard_language,
        dashboard_spec=dashboard_spec,
    )
    dashboard_url = dashboard_server.start()
    print(
        f"Dashboard: {dashboard_url}. Open it, adjust the Session config, "
        "and click Start Session.",
        flush=True,
    )

    # defaults_from_args 将 CLI 参数投影成前端表单默认值；wait_for_launch 一直
    # 等到用户点击启动；apply_to_args 再把用户确认的值写回同一个 Namespace。
    # 后续共享服务和所有 TaskRun 因而使用完全一致的最终 Session 配置。
    launch_config = dashboard_server.wait_for_launch(defaults=defaults_from_args(args))
    apply_to_args(args, launch_config)

    # Dashboard 要保证每个 TaskRun 都拥有全新的 env_server，以实现任务隔离和
    # 可控清理。外接 --env-endpoint 的生命周期不归 Session 管理，无法满足约束。
    if args.env_endpoint is not None:
        parser.error(
            "Dashboard task control cannot use --env-endpoint because each "
            "TaskRun requires a fresh owned env_server"
        )

    # 阶段 2：确定整个 Session 的根目录。每个 TaskRun 会在其下拥有独立子目录；
    # 未显式传 --output-dir 时，时间戳同时确保目录可读且不会覆盖旧 Session。
    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        session_root = get_repo_root() / "logs" / f"{timestamp}_dashboard_session"
    else:
        session_root = Path(args.output_dir)
    session_root = init_output_dir(session_root, verbose=args.verbose)

    # 保存最终配置和原始启动命令，便于从日志精确复现实验。shlex.join 会正确
    # 转义包含空格或特殊字符的命令行参数。
    logger.info("Dashboard: %s", dashboard_url)
    logger.info("launcher Session config applied: %s", launch_config)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    # 在加载 VLA/SAM3 等重资源前同步环境需要的本地资源。若资源缺失，这里尽早
    # 失败，而不是等到用户提交 TaskRun 后才出现难以定位的后台异常。
    ensure_resources(args.env_name)

    # DashboardState 是后端各线程之间的事件边界：Controller 写入 Session、任务、
    # transcript 和媒体状态；HTTP/SSE 路由读取这些状态并推送给浏览器。
    state = DashboardState(
        run_id=f"dashboard-session/{session_root.name}",
        output_dir=session_root,
        dashboard_spec=dashboard_spec,
    )
    dashboard_server.register(state)

    # 阶段 3：把环境相关创建逻辑以回调交给通用 Session 控制器。
    #
    # start_shared 只执行一次，典型返回 VLA/SAM3 客户端参数；run_task 每收到一个
    # 已认领任务执行一次。Controller 负责串行调度，避免多个 Agent 同时操作硬件
    # 或争抢同一个共享推理服务。
    controller = DashboardSessionController(
        state=state,
        start_shared=lambda: env_spec.init_shared_runtime(
            args,
            session_root,
            state,
        ),
        run_task=lambda claimed, shared: _run_dashboard_task(
            args=args,
            env_spec=env_spec,
            state=state,
            claimed=claimed,
            shared_primitives_kwargs=shared,
            session_root=session_root,
        ),
    )
    try:
        # run() 内部启动共享运行时，并持续处理前端提交的 TaskRun，直到 Session
        # 收到关闭请求。所有具体任务都由 _run_dashboard_task 串行完成。
        controller.run()

        # fatal 表示共享服务等 Session 级资源已不可恢复。仍保留 Web 服务，让用户
        # 能在浏览器查看错误和历史记录；此处等待 Ctrl+C，而不是立即关闭页面。
        if state.session_state == "fatal":
            logger.error(
                "Dashboard Session is fatal. Still serving at %s; "
                "press Ctrl+C to stop.",
                dashboard_url,
            )
            threading.Event().wait()
    except KeyboardInterrupt:
        # 将终端中断转换成统一状态请求，由 Controller/daemon 按既定生命周期清理，
        # 避免直接退出导致 env_server 或 GPU 模型服务成为孤儿进程。
        state.request_shutdown()
    return 0


def _run_dashboard_task(
    *,
    args: argparse.Namespace,
    env_spec: EnvSpec,
    state: DashboardState,
    claimed: ClaimedTask,
    shared_primitives_kwargs: dict[str, Any],
    session_root: Path,
) -> str | None:
    """在 Session 共享服务上执行一个全新的 Dashboard TaskRun。

    每次调用都创建并清理 TaskRun 级资源，包括 env_server、Toolkit 和 Planner；
    VLA、SAM3 等昂贵服务通过 ``shared_primitives_kwargs`` 复用。无论初始化、规划、
    工具调用还是收尾阶段是否抛出异常，函数都会尽力关闭资源并写入 transcript。

    Args:
        args: 已由启动页确认的 Session 级参数。函数不会直接修改它。
        env_spec: 环境插件，负责把前端任务参数解析成运行配置并创建 env_server。
        state: Session 共享状态，同时也是 Planner/Toolkit 的 Dashboard 事件接收器。
        claimed: Controller 已认领的任务，包含任务编号、参数覆盖和专属输出目录。
        shared_primitives_kwargs: Session 共享运行时提供给 Toolkit 的客户端或句柄。
        session_root: Session 根目录，用于维持父级目录及日志组织结构。

    Returns:
        成功时返回 ``None``；规划或清理失败时返回面向 Controller 的错误字符串。
    """
    # 阶段 1：构造本任务的参数快照。
    #
    # 浅复制 Namespace 可防止 task/seed/suite 等前端覆盖污染 Session 默认参数，
    # 从而确保下一个 TaskRun 仍从同一套启动配置开始。request 的键与 CLI 参数名
    # 一致，输出目录则强制使用 Controller 为该任务预留的唯一目录。
    task_args = copy.copy(args)
    for name, value in claimed.request.items():
        setattr(task_args, name, value)
    task_args.output_dir = str(claimed.output_dir)

    # parse_config 将通用 argparse 参数转换为环境专属配置，并生成 recipe_tag、
    # task_desc、prompt_vars 等后续 Planner 和记录系统需要的标准字段。
    run_config = env_spec.parse_config(task_args)
    output_dir = init_output_dir(run_config.output_dir, verbose=args.verbose)

    # 这些变量在 try 外初始化，确保任意阶段失败后 finally 都能安全读取并落盘。
    recipe_tag = run_config.recipe_tag
    finish_result = None
    messages: list[dict] = []
    stats: dict = {}
    agent_error: str | None = None
    task_daemons: list[ProcessDaemon] = []
    toolkit = None
    started = time.time()
    try:
        # 阶段 2：创建任务独占运行时。LIBERO 通常在这里启动新的 env_server，
        # 返回值分为“需要停止的 daemon”和“注入 Toolkit 的任务级参数”。
        task_daemons, task_primitives_kwargs = env_spec.init_task_runtime(
            task_args,
            output_dir,
            state,
        )

        # 用户可能在 env_server 启动期间从前端切换任务。若已请求替换，当前任务不再
        # 创建 Planner 或执行动作，而是直接进入 finally 清理刚启动的任务资源。
        if not state.task_replacement_requested:
            # 合并 TaskRun 独占能力和 Session 共享能力。共享参数后展开，因此同名键
            # 由 Session 级连接覆盖，避免误用任务初始化阶段产生的临时共享句柄。
            primitives_kwargs = {
                **task_primitives_kwargs,
                **shared_primitives_kwargs,
            }

            # Toolkit 将底层 primitive 封装为 Agent 可调用工具，并把执行事件写入
            # DashboardState，使前端能实时显示工具参数、结果、图像和动作视频。
            toolkit = get_toolkit(
                args.env_name,
                primitives_kwargs=primitives_kwargs,
                dashboard_events=state,
            )

            # Planner 也是每个 TaskRun 新建，防止不同任务共享对话历史、token 统计或
            # 中断状态；VLA/SAM3 模型服务仍由 Toolkit 通过共享客户端复用。
            planner = build_planner(
                args.planner,
                output_dir=output_dir,
                recipe_tag=recipe_tag,
                env_name=args.env_name,
                base_url=args.base_url,
                model=args.model,
                max_tokens=args.max_tokens,
                planner_timeout_s=args.planner_timeout_s,
                claude_code_max_budget_usd=args.claude_code_max_budget_usd,
                dashboard_events=state,
                no_images=args.no_images,
            )

            # 阶段 3：渲染环境提供的提示词。output_dir 额外注入变量表，允许系统提示
            # 和工具说明引用当前 TaskRun 的绝对输出位置，而不硬编码目录。
            prompt_vars = {**run_config.prompt_vars, "output_dir": output_dir}
            system_prompt = env_spec.prompts.render(
                "system",
                variables=prompt_vars,
            )
            user_message = env_spec.prompts.render(
                "user",
                variables=prompt_vars,
            )

            # 提示词渲染期间仍可能收到任务替换请求，因此在真正启动 Agent 前进行第二
            # 次竞态检查。RunStartedEvent 只在确定执行后发送，避免前端错误显示运行中。
            if not state.task_replacement_requested:
                state.emit(RunStartedEvent())
                result = planner.solve(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    toolkit=toolkit,
                    max_turns=args.max_turns,
                    # state 实现 DashboardInteractionPort：允许浏览器追加消息、排队
                    # 输入和请求中断，而不是使用终端 input_queue。
                    dashboard_interaction=state,
                )

                # PlannerResult 是不同 Planner 后端的统一结果。先保存到局部变量，
                # finally 即使后续清理失败，也能把已完成的结果写入 transcript。
                finish_result = result.finish_result
                messages = result.messages
                stats = result.stats
                agent_error = result.error
    except Exception as exc:
        # 捕获整个 TaskRun 边界，防止单个任务异常杀死长生命周期 Session。错误交给
        # Controller 更新前端状态，随后 finally 仍负责资源回收和审计记录。
        logger.error("EXCEPTION in Dashboard TaskRun %04d: %s", claimed.number, exc)
        agent_error = str(exc)
    finally:
        # 阶段 4：按依赖关系逆序清理。
        #
        # 先关闭 Toolkit，停止正在进行的工具/录像并写 recipe；再倒序停止 daemon，
        # 对应“后创建先销毁”，避免 env_server 先退出后 Toolkit 清理还尝试 RPC。
        cleanup_errors: list[str] = []
        if toolkit is not None:
            try:
                toolkit.close()
                recipe_path = toolkit.write_recipe(recipe_tag)
                logger.info("recipe: %s", recipe_path)
            except Exception as exc:
                # 单项清理失败不能阻止其他资源继续释放，因此只累计错误。
                cleanup_errors.append(f"Toolkit cleanup failed: {exc}")
        for daemon in reversed(task_daemons):
            try:
                daemon.stop()
            except Exception as exc:
                cleanup_errors.append(f"env cleanup failed: {exc}")

        # 若任务本身成功但清理失败，清理错误就是该 TaskRun 的最终错误；若已有主要
        # 错误，则保留主要错误给前端，只把附加清理问题写入日志。
        if cleanup_errors:
            cleanup_error = "; ".join(cleanup_errors)
            if agent_error is None:
                agent_error = cleanup_error
            else:
                logger.warning("%s", cleanup_error)

        # 无论任务成功、失败还是在启动阶段被替换，都写一份最小审计记录。消息先经
        # _serialize_messages 转成 JSON 兼容结构；default=str 为第三方统计对象兜底。
        transcript_path = output_dir / f"transcript_{run_config.recipe_tag}.json"
        record = {
            **run_config.task_desc,
            "model": args.model,
            "elapsed_s": round(time.time() - started, 1),
            "finish": finish_result,
            "stats": stats,
            "messages": _serialize_messages(messages),
        }
        try:
            # 当前 TaskRun 拥有独立输出目录，使用追加模式可保留同一路径中可能已有的
            # 诊断内容；该文件是审计产物，不会再作为 Planner 输入。
            with open(transcript_path, "a") as transcript_file:
                json.dump(record, transcript_file, indent=2, default=str)
        except Exception as exc:
            # transcript 写入失败不应覆盖更重要的 Agent/环境错误，也不能阻塞 Session
            # 执行后续任务，因此只记录 warning。
            logger.warning(
                "failed to write TaskRun transcript %s: %s", transcript_path, exc
            )

        # 重新确认 Session 根目录存在。某些外部清理器可能处理 TaskRun 子目录；这里
        # 维持父目录，保证后续 Controller 创建任务目录时有稳定落点。
        init_output_dir(session_root, verbose=args.verbose)

    return agent_error
