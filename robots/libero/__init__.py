"""LIBERO 对 RPent 环境插件协议的具体实现。

``rpent.envs.base`` 按环境名动态导入本模块，并通过 :func:`get_env_spec` 取得供
通用 CLI 编排的声明与 hook：

- ``_add_cli_args``：向共享解析器注册 suite/task/seed、服务 endpoint 和 CUDA 参数；
- ``_parse_config``：校验最终参数，派生 recipe tag、输出目录、Prompt 变量和任务元数据；
- ``_init_runtime``：普通单任务 CLI 一次性启动或连接 env/VLA/SAM3；
- ``init_shared_runtime`` + ``init_task_runtime``：将 Dashboard 生命周期拆成可跨任务
  复用的 Session 服务和每次重建的 TaskRun 环境；
- ``get_toolkit``：把各运行时返回的轻量客户端注入 ``LiberoToolkit``。

本模块是通用编排层与 LIBERO 实现的边界：``EnvSpec`` 本身不持有实时服务，hook
返回的 daemon 列表只表示当前进程拥有、需要在相应生命周期结束时关闭的子进程，
外部 endpoint 永不纳入所有权。重型仿真、RPC 和模型依赖均在 hook 内延迟导入，
因此单纯枚举环境、构建 argparse 或读取 Dashboard spec 时不会加载 MuJoCo、RLinf
或 CUDA 模型。
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from robots.libero.prompt_bundle import system_prompt, user_prompt
from robots.libero.spec import LIBERO_DASHBOARD_SPEC
from rpent.dashboard.events import DashboardEventSink, RuntimeStatusEvent
from rpent.envs.env_spec import EnvSpec, RunConfig
from rpent.envs.prompt_bundle import PromptBundle
from rpent.utils.config import get_repo_root

if TYPE_CHECKING:
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc import RpcClient


def get_env_spec() -> EnvSpec:
    """返回 LIBERO 的静态环境描述符及全部编排 hook。

    调用该函数只构造不可变的 :class:`EnvSpec`：Prompt 工厂、CLI 扩展、三种运行时
    初始化入口和 Dashboard 声明都以引用形式登记，不会在此刻启动服务。工具 schema、
    handler、MCP allowlist 与 Toolkit 自身的关闭逻辑不属于 ``EnvSpec``，由
    :func:`get_toolkit` 负责连接。

    Returns:
        名称为 ``libero``、可供普通 CLI 与 Dashboard 共用的环境描述符。
    """
    # 描述符只建立依赖关系；真正的运行时对象均由下列 hook 被调用时才创建。
    return EnvSpec(
        name="libero",
        prompts=PromptBundle(
            system=system_prompt,
            user=user_prompt,
        ),
        add_cli_args=_add_cli_args,
        parse_config=_parse_config,
        init_shared_runtime=init_shared_runtime,
        init_task_runtime=init_task_runtime,
        init_runtime=_init_runtime,
        dashboard=LIBERO_DASHBOARD_SPEC,
    )


def get_toolkit(
    *,
    primitives_kwargs: dict[str, Any],
    dashboard_events: DashboardEventSink,
):
    """用已初始化的客户端构造 LIBERO Toolkit。

    ``primitives_kwargs`` 在普通 CLI 中由 :func:`_init_runtime` 一次性产生；在
    Dashboard 中则由 TaskRun 的 ``env`` 与 Session 共享的 ``model``、
    ``sam3_client`` 合并而成。事件接收器继续注入 Toolkit，使工具执行结果能投影到
    Dashboard；此函数不取得或关闭任何 daemon 的所有权。

    Args:
        primitives_kwargs: 以名称注入 primitive 的环境、模型与分割客户端。
        dashboard_events: Toolkit 执行期间继续使用的 Dashboard 事件接收器。

    Returns:
        同时提供通用工具和 LIBERO 物理操作 primitive 的 ``LiberoToolkit`` 实例。
    """
    # Toolkit 依赖较重，延迟导入可保持环境发现和参数帮助路径轻量。
    from robots.libero.toolkit import LiberoToolkit

    return LiberoToolkit(
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
    )


def _add_cli_args(parser: argparse.ArgumentParser, use_dashboard: bool) -> None:
    """向共享 ``parser`` 注册 LIBERO 专属命令行参数。

    普通 CLI 在 argparse 阶段就要求 ``--suite`` 与 ``--task``；Dashboard 启动时
    尚未选择具体 TaskRun，因此先允许二者为空，待面板提交命令并覆盖 Namespace 后再由
    :func:`_parse_config` 校验。三个 endpoint 独立控制“连接外部服务”还是“启动本地
    子进程”，``--cuda-device`` 则原样传给需要本地启动的服务端。

    Args:
        parser: 已包含通用选项的共享参数解析器。
        use_dashboard: 是否采用延后选择任务的 Dashboard 模式。
    """
    # required 的差异只改变参数解析时机；两种入口最终都必须通过 _parse_config 校验。
    required = not use_dashboard
    parser.add_argument("--max-episode-steps", type=int, default=10000)
    parser.add_argument("--libero-type", default=None,
                        choices=["standard", "pro", "plus"],
                        help="LIBERO variant (auto-routed from suite suffix if not set).")
    parser.add_argument("--suite", default=None, required=required,
                        help="e.g. libero_object_task, libero_spatial_swap")
    parser.add_argument("--task", type=int, default=None, required=required)
    parser.add_argument("--seed", type=int, default=0)
    # endpoint 为空表示本次运行拥有本地 daemon；非空则只创建传输客户端而不管理服务寿命。
    parser.add_argument("--env-endpoint", default=None,
                        help="[protocol://]host:port of an existing env_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local env_server is spawned.")
    parser.add_argument("--vla-endpoint", default=None,
                        help="[protocol://]host:port of an existing vla_server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local vla_server is spawned.")
    parser.add_argument("--sam3-endpoint", default=None,
                        help="[protocol://]host:port of an existing SAM3 server "
                             "(protocol=http|socket, defaults to http). "
                             "If unset, a local SAM3 server is spawned.")
    parser.add_argument("--cuda-device", type=int, default=None,
                        help="GPU device to expose via CUDA_VISIBLE_DEVICES.")


def _parse_config(args: argparse.Namespace) -> RunConfig:
    """校验最终参数并派生单次运行的稳定标识与 Prompt 上下文。

    Dashboard 会先把用户选择的字段写回参数副本，再调用本 hook，因此这里是普通 CLI
    与 Dashboard 共同的最终校验点。``recipe_tag`` 用于审计、recipe 和 transcript
    命名；``prompt_vars`` 只包含模板所需的任务身份，实际 ``output_dir`` 由上层在目录
    初始化完成后合并。用户显式指定的输出目录保持不变，否则在仓库 ``logs`` 下按当前
    时间和任务身份生成默认目录。

    Args:
        args: 已应用通用参数、环境参数及可能的 Dashboard 覆盖项的 Namespace。

    Returns:
        包含文件标识、输出目录、Prompt 变量和 transcript 任务元数据的 ``RunConfig``。

    Raises:
        ValueError: suite 为空或 task 尚未设置。
    """
    # Dashboard 的 suite/task 在 argparse 阶段可为空，但进入 TaskRun 前必须已经补齐。
    if not args.suite:
        raise ValueError("--suite is required")
    if args.task is None:
        raise ValueError("--task is required")

    # recipe_tag 去掉常见环境前缀，形成同一 suite/task/seed 可复用的产物文件基名。
    recipe_tag = f"{args.suite.replace('libero_', '')}_t{args.task}_s{args.seed}"
    # 这些键对应 user Prompt 的 {{suite}}/{{task}}/{{seed}}/{{recipe_tag}} 占位符。
    prompt_vars = {
        "suite": args.suite,
        "task": args.task,
        "seed": args.seed,
        "recipe_tag": recipe_tag,
    }

    output_dir = args.output_dir
    if output_dir is None:
        # 时间戳隔离普通 CLI 的重复运行；Dashboard 会预先传入每个 TaskRun 的专属目录。
        timestamp = datetime.now().strftime("%Y%m%d-%H:%M:%S")
        output_dir = get_repo_root() / "logs" / f"{timestamp}_{args.suite}_t{args.task}_s{args.seed}"
    output_dir = Path(output_dir)

    return RunConfig(
        recipe_tag=recipe_tag,
        output_dir=output_dir,
        prompt_vars=prompt_vars,
        task_desc={"suite": args.suite, "task": args.task, "seed": args.seed},
    )


def _subprocess_env(**extra: str) -> dict[str, str]:
    """复制父进程环境，并以 ``extra`` 覆盖本地服务所需变量。

    每次返回独立字典，避免为某个 daemon 设置 LIBERO/MuJoCo 变量时污染当前 Python
    进程或其他服务。GPU 选择不在这里改写：``--cuda-device`` 会传入服务端，由服务端
    在首次接触 CUDA/EGL 前协调 ``CUDA_VISIBLE_DEVICES``。

    Args:
        **extra: 仅对子进程生效的环境变量覆盖项。

    Returns:
        可直接传给 ``ProcessDaemon`` 的完整环境变量字典。
    """
    env = os.environ.copy()
    env.update(extra)
    return env


def init_task_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """初始化一个 Dashboard TaskRun 独占的 LIBERO 环境。

    每次调用都对应新的任务代次。未提供 ``--env-endpoint`` 时，本 hook 选择空闲端口、
    启动配置了 suite/task/seed 的本地 ``env_server``，并把 daemon 作为 TaskRun 所有物
    返回；提供 endpoint 时只连接外部服务，返回的所有权列表为空，因而任务结束不会
    停止该服务。无论传输来自本地还是外部，都会等待 ready，并由 ``LiberoEnvClient``
    校验服务端元数据是否与当前任务一致。

    Dashboard 状态按 ``starting -> ready`` 发布；启动、endpoint 解析、就绪等待或
    元数据校验任一步失败，都会停止本次已拥有的 daemon、发布 ``failed`` 后重新抛出。
    VLA 与 SAM3 不在此处创建，它们属于 Session 级共享运行时。

    Args:
        args: 已写入当前 TaskRun suite/task/seed 等字段的参数副本。
        output_dir: 当前 TaskRun 的独立输出目录，也是环境服务日志目录。
        dashboard_events: 接收 ``env`` 组件状态变化的事件接收器。

    Returns:
        ``(owned_daemons, {"env": client})``；列表仅含本次启动的本地环境服务。
    """
    # 所有运行时实现都在 hook 调用后再导入，环境枚举路径无需安装仿真/RPC 依赖。
    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.socket_rpc import SocketRpcClient

    # “owned”是清理边界：外部 endpoint 对应的客户端绝不会把服务加入该列表。
    owned_daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon: ProcessDaemon | None = None
        # 本地分支创建并登记 daemon；外部 endpoint 分支只选择协议并构造轻量 RPC 客户端。
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            env_daemon = ProcessDaemon(
                name="env_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                    "--suite", args.suite,
                    "--task", str(args.task),
                    "--seed", str(args.seed),
                    "--max-episode-steps", str(args.max_episode_steps),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(
                    LIBERO_TYPE=libero_type,
                    MUJOCO_GL="egl",
                    ROBOT_PLATFORM="LIBERO",
                ),
                log_path=str(Path(output_dir) / "env_server.log"),
            )
            env_daemon.start()
            owned_daemons.append(env_daemon)
            env_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.env_endpoint)
            if protocol == "socket":
                env_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                env_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--env-endpoint protocol must be socket or http, got {protocol!r}"
                )
        # ready 探针同时监控本地 daemon；随后再以 expected_meta 防止连到错误任务实例。
        wait_for_ready(env_rpc, daemon=env_daemon)
        env = LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise
    dashboard_events.emit(RuntimeStatusEvent("env", "ready"))
    return owned_daemons, {"env": env}


def init_shared_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """初始化 Dashboard Session 级共享的 VLA 与 SAM3 服务。

    Dashboard 启动 Session 时只调用一次本 hook，随后多个顺序 TaskRun 复用返回的
    ``model`` 和 ``sam3_client``。VLA、SAM3 可分别采用本地 daemon 或外部 endpoint；
    只有本地启动项会进入返回的所有权列表，并在 Session 结束时由控制器关闭。

    所有需要本地启动的服务都会先调用 ``start``，待 VLA/SAM3 两个 RPC 客户端都已
    构造后再统一等待就绪，使耗时的模型初始化可以并行推进；状态检查按 SAM3、VLA 的
    确定顺序完成。任一启动、协议解析或 ready 检查失败时，本 hook 会逆序停止当前已
    拥有的全部共享 daemon，发布对应组件的 ``failed`` 状态并保留原异常。

    Args:
        args: Session 配置，其中 endpoint 与 CUDA 选择对后续所有 TaskRun 生效。
        output_dir: Session 根目录，本地共享服务在此写入各自日志。
        dashboard_events: 接收 ``vla``、``sam3`` 生命周期状态的事件接收器。

    Returns:
        本地共享 daemon 列表，以及包含 ``model``、``sam3_client`` 的 primitive 参数。
    """
    # 延迟导入确保普通环境发现不会加载传输层或模型客户端。
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    # 列表既是异常回滚集合，也是 Session 控制器最终清理时的所有权清单。
    owned_daemons: list[ProcessDaemon] = []
    cuda_args = (
        ["--cuda-device", str(args.cuda_device)]
        if args.cuda_device is not None
        else []
    )

    # --- vla_server --------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon: ProcessDaemon | None = None
        if args.vla_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
            vla_daemon.start()
            owned_daemons.append(vla_daemon)
            vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.vla_endpoint)
            if protocol == "socket":
                vla_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                vla_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise

    # --- sam3_server -------------------------------------------------------
    dashboard_events.emit(RuntimeStatusEvent("sam3", "starting"))
    try:
        sam3_daemon: ProcessDaemon | None = None
        if args.sam3_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            sam3_daemon = ProcessDaemon(
                name="sam3_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )
            sam3_daemon.start()
            owned_daemons.append(sam3_daemon)
            sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.sam3_endpoint)
            if protocol == "socket":
                sam3_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        _stop_owned_daemons(owned_daemons)
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    # 所有需本地启动的进程都已 start，两个 RPC 客户端也已构造，此时再等待可让重型
    # 初始化并行进行；固定检查顺序则让状态与故障归因稳定。外部服务只执行 RPC 探针。
    for component, client, daemon in (
        ("sam3", sam3_rpc, sam3_daemon),
        ("vla", vla_rpc, vla_daemon),
    ):
        try:
            wait_for_ready(client, daemon=daemon)
        except Exception as exc:
            _stop_owned_daemons(owned_daemons)
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    # ready 之后才包成领域客户端；底层模型仍驻留服务端，跨 TaskRun 复用同一 RPC 连接。
    model = VLAClient(vla_rpc)
    sam3_client = Sam3Client(sam3_rpc)

    return owned_daemons, {
        "model": model,
        "sam3_client": sam3_client,
    }


def _init_runtime(
    args: argparse.Namespace,
    output_dir: Path,
    dashboard_events: DashboardEventSink,
) -> tuple[list[ProcessDaemon], dict[str, Any]]:
    """为普通单任务 CLI 初始化完整的 LIBERO 运行时。

    与 Dashboard 的两层 hook 不同，此入口在一次调用中处理 env、VLA、SAM3，并把三类
    客户端一起返回。每项服务都可独立选择本地启动或连接 endpoint：本地 daemon 加入
    ``daemons`` 供本次运行结束时清理，外部服务只建立 HTTP/socket 客户端。所有需要
    本地启动的进程均 ``start``、三个 RPC 客户端均构造后，才按 env、SAM3、VLA 顺序
    等待 ready，以并行利用重型初始化时间；ready 失败会逆序停止已经登记的本地进程并
    发布组件失败事件。

    ``env`` 客户端额外校验 suite/task/seed/max steps 元数据，``model`` 与
    ``sam3_client`` 则封装各自 RPC。所有重型依赖继续延迟导入，使裸
    ``import robots.libero`` 只加载环境描述而不加载运行时实现。

    Args:
        args: 已通过 :func:`_parse_config` 校验的普通 CLI 参数。
        output_dir: 本次运行的输出目录，三个本地服务分别在此写日志。
        dashboard_events: 即使不使用交互面板也统一接收组件生命周期事件的 sink。

    Returns:
        当前运行拥有的 daemon 列表，以及 ``env``/``model``/``sam3_client`` 参数字典。
    """
    # 运行时依赖只在该 hook 真正执行时加载，保持 argparse 与环境注册阶段轻量。
    from robots.libero.env_client import LiberoEnvClient
    from rpent.utils.config import get_libero_type
    from rpent.utils.daemon import ProcessDaemon, pick_free_port
    from rpent.utils.http_rpc import HttpRpcClient
    from rpent.utils.rpc import parse_endpoint, wait_for_ready
    from rpent.utils.sam3_client import Sam3Client
    from rpent.utils.socket_rpc import SocketRpcClient
    from rpent.utils.vla_client import VLAClient

    # 只有本地创建的进程进入该列表；endpoint 背后的服务不属于当前运行。
    daemons: list[ProcessDaemon] = []
    libero_type = args.libero_type or get_libero_type()
    cuda_args = ["--cuda-device", str(args.cuda_device)] if args.cuda_device is not None else []

    # --- env_server --------------------------------------------------------
    # 环境服务绑定本次 suite/task/seed，并通过独立进程隔离 MuJoCo 状态；本地分支还设置
    # LIBERO 类型、EGL 后端和机器人平台环境变量，endpoint 分支则不改动远端配置。
    dashboard_events.emit(RuntimeStatusEvent("env", "starting"))
    try:
        env_daemon: ProcessDaemon | None = None
        if args.env_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            env_daemon = ProcessDaemon(
                name="env_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "env_server.py"),
                    "--suite", args.suite,
                    "--task", str(args.task),
                    "--seed", str(args.seed),
                    "--max-episode-steps", str(args.max_episode_steps),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(
                    LIBERO_TYPE=libero_type,
                    MUJOCO_GL="egl",
                    ROBOT_PLATFORM="LIBERO",
                ),
                log_path=str(Path(output_dir) / "env_server.log"),
            )
            env_daemon.start()
            daemons.append(env_daemon)
            env_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.env_endpoint)
            if protocol == "socket":
                env_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                env_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--env-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("env", "failed", error=exc))
        raise

    # --- vla_server --------------------------------------------------------
    # VLA 服务是独立的长生命周期推理进程，只负责“观测 -> Pi0.5 动作块”。
    # 环境执行仍在 env_server 中，因此模型服务可以远端部署或跨任务复用。
    dashboard_events.emit(RuntimeStatusEvent("vla", "starting"))
    try:
        vla_daemon: ProcessDaemon | None = None
        if args.vla_endpoint is None:
            # 未提供 endpoint：选择本机空闲端口并由当前运行拥有该子进程。最终清理
            # 时只有 daemons 中的本地服务会被停止，外部服务绝不会被误关闭。
            host, port = "127.0.0.1", pick_free_port()
            vla_daemon = ProcessDaemon(
                name="vla_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "vla_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    # 父进程异常退出、stdin pipe 关闭时，服务端自行终止，避免遗留
                    # 占用 GPU 显存的孤儿模型进程。
                    "--parent-watch",
                    # --cuda-device 在子进程第一次构造 CUDA 模型前设置可见 GPU。
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "vla_server.log"),
            )
            vla_daemon.start()
            daemons.append(vla_daemon)

            # 这里只创建轻量传输客户端；模型仍在 vla_server 子进程中加载。
            vla_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            # 提供 endpoint：不启动、不拥有远端服务，只按协议构造客户端。省略协议
            # 时 parse_endpoint 默认使用 HTTP。
            protocol, host, port = parse_endpoint(args.vla_endpoint)
            if protocol == "socket":
                vla_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                vla_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--vla-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        # 将启动/endpoint 解析错误同步给 Dashboard，然后让上层统一清理已启动服务。
        dashboard_events.emit(RuntimeStatusEvent("vla", "failed", error=exc))
        raise

    # --- sam3_server -------------------------------------------------------
    # SAM3 只提供无动作的视觉分割/反投影辅助，与 VLA 一样可本地常驻或连接外部部署。
    dashboard_events.emit(RuntimeStatusEvent("sam3", "starting"))
    try:
        sam3_daemon: ProcessDaemon | None = None
        if args.sam3_endpoint is None:
            host, port = "127.0.0.1", pick_free_port()
            sam3_daemon = ProcessDaemon(
                name="sam3_server",
                cmd=[
                    sys.executable,
                    str(get_repo_root() / "robots" / "libero" / "sam3_server.py"),
                    "--transport", "http",
                    "--host", host,
                    "--port", str(port),
                    "--parent-watch",
                    *cuda_args,
                ],
                env=_subprocess_env(),
                log_path=str(Path(output_dir) / "sam3_server.log"),
            )
            sam3_daemon.start()
            daemons.append(sam3_daemon)
            sam3_rpc: RpcClient = HttpRpcClient(f"http://{host}:{port}")
        else:
            protocol, host, port = parse_endpoint(args.sam3_endpoint)
            if protocol == "socket":
                sam3_rpc = SocketRpcClient(host, port)
            elif protocol == "http":
                sam3_rpc = HttpRpcClient(f"http://{host}:{port}")
            else:
                raise ValueError(
                    f"--sam3-endpoint protocol must be socket or http, got {protocol!r}"
                )
    except Exception as exc:
        dashboard_events.emit(RuntimeStatusEvent("sam3", "failed", error=exc))
        raise

    # 所有需本地启动的服务均已启动，三个 RPC 客户端也已构造；此时等待可让模型加载
    # 与仿真初始化重叠。检查固定为 env、SAM3、VLA；失败时逆序关闭已拥有进程。
    for component, client, daemon in (
        ("env", env_rpc, env_daemon),
        ("sam3", sam3_rpc, sam3_daemon),
        ("vla", vla_rpc, vla_daemon),
    ):
        try:
            wait_for_ready(client, daemon=daemon)
        except Exception as exc:
            for started_daemon in reversed(daemons):
                started_daemon.stop()
            dashboard_events.emit(RuntimeStatusEvent(component, "failed", error=exc))
            raise
        dashboard_events.emit(RuntimeStatusEvent(component, "ready"))

    # Toolkit 只接收领域客户端，不直接接触 daemon；环境元数据校验阻止误连其他 cell。
    primitives_kwargs = {
        "env": LiberoEnvClient(
            env_rpc,
            expected_meta={
                "suite": args.suite,
                "task": args.task,
                "seed": args.seed,
                "max_episode_steps": args.max_episode_steps,
            },
        ),
        "model": VLAClient(vla_rpc),
        "sam3_client": Sam3Client(sam3_rpc),
    }
    return daemons, primitives_kwargs


def _stop_owned_daemons(daemons: list[ProcessDaemon]) -> None:
    """逆序停止本次 hook 已取得所有权的 daemon。

    该辅助函数用于 Dashboard 分层运行时的启动失败回滚。逆序清理与登记顺序对称；某个
    ``stop`` 自身失败会被忽略，以免遮蔽真正的启动/就绪异常。调用方必须只传入本地
    owned 列表，不能把外部 endpoint 对应服务加入其中。

    Args:
        daemons: 按成功启动顺序登记的本地进程。
    """
    for daemon in reversed(daemons):
        try:
            daemon.stop()
        except Exception:
            # 回滚是尽力而为；继续处理更早启动的进程，并由调用方重抛原始异常。
            pass
