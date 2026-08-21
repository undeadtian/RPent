"""RPent 物理智能体的命令行总入口。

``pyproject.toml`` 把 console script ``rpent`` 映射到本模块的 ``main``：

.. code-block:: bash

   rpent --env libero --suite libero_object_task --task 0 --seed 0 [...]

本文件只做编排，不实现具体机器人、模型或工具：

1. 解析全局参数，并通过 ``EnvSpec`` 注入环境专属参数；
2. 构造 Planner，渲染环境提供的 system/user prompt；
3. 启动或连接环境、VLA、感知等运行时服务；
4. 创建环境 Toolkit，把 LLM 工具调用连接到物理 primitive；
5. 运行 Agent loop，最后保存 recipe、视频、日志和 transcript。

依赖方向必须保持为“CLI 依赖其他层”。其他 ``rpent`` 模块不要反向导入
``rpent.cli``，否则容易在 planner/env/tools/dashboard 之间形成循环依赖。
"""
from __future__ import annotations

import argparse
import json
import queue
import shlex
import sys
import time
from collections.abc import Callable
from pathlib import Path

# TUI 只负责终端输入；Planner 通过 Queue 接收首条任务和后续人工 steering。
from rpent.cli.tui import (
    start_first_prompt_resolver,
    start_interactive_reader,
)
from rpent.dashboard.events import (
    NullDashboardEventSink,
    RunStartedEvent,
)
# envs 是插件注册层：这里不直接 import robots.libero，从而保留扩展新环境的能力。
from rpent.envs import enumerate_envs, get_env_spec, get_toolkit
from rpent.planner.base import build_planner
from rpent.utils.logging import get_logger, init_output_dir
from rpent.utils.resources import ensure_resources

logger = get_logger("agent")


# ---------------------------------------------------------------------------
# Agent transcript serialization
# ---------------------------------------------------------------------------


def _strip_images(value):
    """递归复制消息结构，并移除体积很大的内联图像 payload。

    Planner 在运行时仍会收到真实图片；这里只处理任务结束后写入 transcript 的副本。
    文本、工具调用和普通字段保持不变，Anthropic ``image`` 与 OpenAI
    ``image_url`` 两种常见形态都替换为占位标记。

    非 list/dict 的 SDK 对象原样保留，最终由
    ``json.dump(..., default=str)`` 兜底序列化。
    """
    if isinstance(value, list):
        return [_strip_images(v) for v in value]
    if isinstance(value, dict):
        if value.get("type") == "image":
            return {
                "type": "image",
                "source": {"_omitted_for_transcript": True},
            }
        if value.get("type") == "image_url":
            return {
                "type": "image_url",
                "image_url": {"_omitted_for_transcript": True},
            }
        return {k: _strip_images(v) for k, v in value.items()}
    return value


def _serialize_messages(messages: list[dict]) -> list[dict]:
    """生成可落盘的消息副本，同时剥离每条消息中的内联图像。"""
    return [
        {
            **{k: v for k, v in m.items() if k != "content"},
            "content": _strip_images(m.get("content")),
        }
        for m in messages
    ]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    """创建只含“全局参数”的第一阶段 ArgumentParser。

    环境专属参数不能在这里硬编码。``main`` 首先用本 parser 读取 ``--env``，再从
    对应 ``EnvSpec.add_cli_args`` 注入 LIBERO 的 suite/task/endpoint 等参数，最后
    执行完整解析。
    """
    # 扫描 ``robots/`` 下可导入的环境包，并直接用作 argparse choices；错误环境名
    # 因此会在启动模型或服务之前被拒绝。
    known_envs = enumerate_envs()
    known_envs_text = ", ".join(known_envs) if known_envs else "none"
    ap = argparse.ArgumentParser(
        description="RPent: Agentic Infrastructure for the Physical World",
    )

    ap.add_argument(
        "--env",
        dest="env_name",
        required=True,
        choices=known_envs,
        help=f"Environment backend. Known environments: {known_envs_text}.",
    )

    # Planner/model 参数由所有环境共享。具体凭证和默认模型在 build_planner 中解析。
    ap.add_argument(
        "--planner",
        default="api",
        choices=["api", "claude_code", "codex"],
        help="LLM backend: api | claude_code | codex.",
    )
    ap.add_argument(
        "--model",
        default=None,
        help="Model id. For the 'api' planner, prefix the provider "
        "(e.g. anthropic:claude-opus-4-8, openai:gpt-5.5, "
        "openai-chat:glm-5.2). For claude_code/codex this "
        "overrides the backend default model.",
    )
    ap.add_argument(
        "--base-url",
        default=None,
        help="API base URL. Defaults to the selected backend's base URL env var.",
    )
    ap.add_argument("--max-turns", type=int, default=100)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument(
        "--no-images",
        action="store_true",
        help="Never send image bytes to the model (api planner only). "
        "Use for text-only models that reject image input "
        "(e.g. 400 \"message type 'image_url' is not supported\"); "
        "read_image then returns the image name instead, with a notice.",
    )
    ap.add_argument(
        "--planner-timeout-s",
        type=int,
        default=None,
        help="Wall-clock cap for api/claude_code/codex planner runs. "
        "Terminal interactive API/Claude sessions are exempt. "
        "Defaults to CODEX_TIMEOUT_S (codex only), "
        "CELL_TIMEOUT_S, or 1200.",
    )
    ap.add_argument(
        "--claude-code-max-budget-usd",
        type=float,
        default=None,
        help="Budget passed to claude -p --max-budget-usd. "
        "Defaults to MAX_BUDGET_USD env or 10.",
    )

    # 运行输出和 UI 模式也是环境无关配置。
    ap.add_argument("--output-dir", default=None)
    ap.add_argument(
        "--dashboard",
        action="store_true",
        help="Start a local dashboard server for this single run.",
    )
    ap.add_argument(
        "--dashboard-host",
        default="127.0.0.1",
        help="Dashboard bind host. Defaults to 127.0.0.1.",
    )
    ap.add_argument(
        "--dashboard-port",
        type=int,
        default=0,
        help="Dashboard port. 0 asks the OS for a free port.",
    )
    ap.add_argument(
        "--dashboard-language",
        choices=["en", "zh-cn"],
        default="en",
        help="Dashboard UI language. 'zh-cn' serves the Chinese "
        "translation; defaults to English.",
    )
    ap.add_argument(
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging for stdout and the run.log "
        "file. Defaults to INFO when not set.",
    )
    ap.add_argument(
        "--interactive",
        "-i",
        action="store_true",
        help="Interactive mode: opens an interactive cli session.",
    )

    return ap


def main() -> int:
    """执行一次 RPent CLI 会话，是 ``rpent`` console script 的总入口。

    普通模式、交互模式共享同一套环境和 Planner 生命周期；Dashboard 因为要管理
    长生命周期共享服务与多个 TaskRun，会在参数解析后立即委托给独立编排器。
    """
    # ------------------------------------------------------------------
    # 1. 两阶段参数解析与环境插件选择
    # ------------------------------------------------------------------
    parser = _build_argparser()

    # 第一阶段只关心 --env / --dashboard。parse_known_args 暂时保留尚未注册的
    # --suite、--task 等参数；确定环境后再由环境插件扩展同一个 parser。
    early, _ = parser.parse_known_args()

    # get_env_spec("libero") 会延迟 import robots.libero，并返回纯描述对象 EnvSpec。
    env_spec = get_env_spec(early.env_name)
    env_spec.add_cli_args(parser, use_dashboard=early.dashboard)

    # 第二阶段执行完整校验，此时全局参数和环境参数都已经注册。
    args = parser.parse_args()
    if args.dashboard and args.interactive:
        parser.error("--dashboard and --interactive cannot be used together")

    # Dashboard 拥有不同的生命周期：VLA/SAM3 可在 Session 内复用，每个 TaskRun
    # 使用全新的 env_server。因此这里直接委托并返回，不进入下方单任务流程。
    if args.dashboard:
        from rpent.cli.dashboard import run_dashboard_session

        return run_dashboard_session(args, env_spec, parser=parser)

    # ------------------------------------------------------------------
    # 2. 派生单次运行标识、输出目录和 Prompt 变量
    # ------------------------------------------------------------------
    # parse_config 是环境 hook。LIBERO 在这里校验 suite/task，并生成 recipe_tag、
    # 默认 logs 目录、Prompt 模板变量和最终 transcript 使用的任务描述字段。
    run_config = env_spec.parse_config(args)
    recipe_tag = run_config.recipe_tag
    output_dir = run_config.output_dir
    prompt_vars = run_config.prompt_vars
    task_desc = run_config.task_desc

    env_name = args.env_name

    # 创建输出目录，配置 stdout/run.log，并把目录保存为进程级当前 output_dir。
    # LiberoToolkit 后续通过 get_output_dir() 为 EnvState 找到相同位置。
    output_dir = init_output_dir(output_dir, verbose=args.verbose)
    logger.info("physical agent cmd: %s", shlex.join([sys.executable, *sys.argv]))

    # 同步该环境的跨运行 memory/reference 资源。网络失败是非致命的；resources
    # 模块会记录 warning 并继续使用本地副本。
    ensure_resources(env_name)

    # 普通 CLI 不需要向网页推送事件，但下层统一依赖 DashboardEventSink 接口。
    dashboard_events = NullDashboardEventSink()

    # ------------------------------------------------------------------
    # 3. 构造高层 Planner 并渲染环境 Prompt
    # ------------------------------------------------------------------
    # build_planner 只创建推理后端，不启动机器人服务：
    # - api         -> Pydantic AI agent loop；
    # - claude_code -> Claude Agent SDK + 进程内 MCP 工具桥；
    # - codex       -> Codex SDK planner。
    planner = build_planner(
        args.planner,
        output_dir=output_dir,
        recipe_tag=recipe_tag,
        env_name=env_name,
        base_url=args.base_url,
        model=args.model,
        max_tokens=args.max_tokens,
        planner_timeout_s=args.planner_timeout_s,
        claude_code_max_budget_usd=args.claude_code_max_budget_usd,
        dashboard_events=dashboard_events,
        no_images=args.no_images,
    )

    # Prompt 由环境贡献，CLI 只注入本次任务变量。这样 Planner 实现不需要知道
    # LIBERO suite/task 的语义。output_dir 也提供给 Prompt，供 Agent 读写产物。
    prompt_bundle = env_spec.prompts
    prompt_vars = {**prompt_vars, "output_dir": output_dir}
    system_prompt = prompt_bundle.render(
        "system",
        variables=prompt_vars,
    )
    user_msg = prompt_bundle.render(
        "user",
        variables=prompt_vars,
    )

    # ------------------------------------------------------------------
    # 4. 可选的终端交互输入
    # ------------------------------------------------------------------
    input_queue: "queue.Queue[str | None] | None" = None
    await_first_prompt: "Callable[[], str | None] | None" = None
    if args.interactive:
        input_queue = queue.Queue()

        # TUI 在线程中读取输入。首条默认值是环境渲染的任务，用户可以原样提交或
        # 编辑；之后输入的行作为 Planner steering 消息继续进入同一 Queue。
        start_interactive_reader(input_queue, first_prompt_default=user_msg)
        logger.info(
            "interactive mode on: the built-in task is pre-filled — "
            "edit it and press Enter, submit it as-is, or clear it to "
            "type your own. Once running, type to steer the agent. "
            "/help for commands."
        )

        # 单独线程提前等待首条输入；主线程同时启动较慢的 env/VLA/SAM3 服务，
        # 避免用户输入完成后才开始加载模型。
        await_first_prompt = start_first_prompt_resolver(input_queue)

    # ------------------------------------------------------------------
    # 5. 启动或连接环境运行时服务
    # ------------------------------------------------------------------
    # LIBERO 的 init_runtime 会并行启动/连接 env_server、vla_server、sam3_server，
    # 等待 healthz 后返回：
    # - daemons：仅包含本次运行自己启动、结束时需要 stop 的本地进程；
    # - primitives_kwargs：env/model/sam3_client 三个轻量 RPC client。
    daemons, primitives_kwargs = env_spec.init_runtime(
        args,
        output_dir,
        dashboard_events,
    )

    # ------------------------------------------------------------------
    # 6. 创建工具层并获取初始环境观测
    # ------------------------------------------------------------------
    # get_toolkit 再次通过环境插件工厂构造 LiberoToolkit。构造过程会 reset 环境、
    # 创建 LiberoPrimitives、保存 EnvState step 0，并注册全部 LLM 工具 schema。
    toolkit = get_toolkit(
        env_name,
        primitives_kwargs=primitives_kwargs,
        dashboard_events=dashboard_events,
    )

    # ------------------------------------------------------------------
    # 7. 运行 Planner <-> Toolkit <-> Environment 闭环
    # ------------------------------------------------------------------
    t0 = time.time()
    finish_result, messages, agent_error = None, [], None
    stats: dict = {}
    first_user_msg: str | None = user_msg

    if await_first_prompt is not None:
        # 到这里服务已经启动完成，再等待用户首条输入线程。None 表示用户在任务
        # 开始前退出；这种情况下仍执行下方 finally 清理运行时资源。
        first_user_msg = await_first_prompt()
        if first_user_msg is None:
            logger.info("no task entered; ending session before start.")

    try:
        if first_user_msg is not None:
            dashboard_events.emit(RunStartedEvent())

            # solve 是所有 Planner 的统一协议。Planner 从 toolkit.get_tools_spec()
            # 获得工具定义，并把每次模型工具调用交给 toolkit.execute_tool()；动作后
            # 返回的新图像/状态会进入下一轮模型推理，直到 finish、超时或轮次耗尽。
            result = planner.solve(
                system_prompt=system_prompt,
                user_message=first_user_msg,
                toolkit=toolkit,
                max_turns=args.max_turns,
                input_queue=input_queue,
            )
            finish_result = result.finish_result
            messages = result.messages
            stats = result.stats
            agent_error = result.error
    except Exception as exc:
        # Planner 外泄异常在这里转成运行记录；资源清理由 finally 保证执行。
        logger.error("EXCEPTION in agent loop: %s", exc)
        agent_error = str(exc)
    finally:
        # 先从已记录的 StepRecord 导出成功 primitive/segment 序列，再结束视频记录，
        # 最后停止本次运行拥有的服务。外部 endpoint 不在 daemons 中，不会被关闭。
        recipe_path = toolkit.write_recipe(recipe_tag)
        logger.info("recipe: %s", recipe_path)

        toolkit.close()
        for d in daemons:
            d.stop()

    elapsed = time.time() - t0

    # ------------------------------------------------------------------
    # 8. 保存精简 transcript 与使用统计
    # ------------------------------------------------------------------
    transcript_path = Path(output_dir) / f"transcript_{recipe_tag}.json"
    record = {
        **task_desc,
        "model": args.model,
        "elapsed_s": round(elapsed, 1),
        "finish": finish_result,
        "stats": stats,
        # 去除 base64 图像，避免 transcript 重复保存 EnvState 中已有的大文件。
        "messages": _serialize_messages(messages),
    }
    with open(transcript_path, "a") as f:
        json.dump(record, f, indent=2, default=str)

    logger.info("elapsed: %.1fs", elapsed)
    logger.info(
        "usage: in=%s out=%s tool_calls=%s",
        stats.get("total_input_tokens", "?"),
        stats.get("total_output_tokens", "?"),
        stats.get("tool_calls", "?"),
    )
    logger.info("transcript: %s", transcript_path)
    if agent_error:
        logger.error("error: %s", agent_error)

    # 当前 CLI 用 transcript/log 表达 Agent 失败；只要编排流程完成，进程返回 0。
    return 0


if __name__ == "__main__":
    # 支持 ``python -m rpent.cli.main``；安装后的 ``rpent`` console script 也调用
    # 同一个 main()。
    sys.exit(main())
