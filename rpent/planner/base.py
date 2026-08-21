"""高层推理后端的统一协议与 Planner 工厂。

CLI 不直接依赖 Pydantic AI、Claude Agent SDK 或 Codex SDK。它只通过 ``Planner``
协议调用 ``solve``，并把同一个 Toolkit 交给具体后端。每个后端负责：

1. 从 ``toolkit.get_tools_spec()`` 获取 LLM 可见的工具 schema；
2. 把模型工具调用转交给 ``toolkit.execute_tool()``；
3. 将文本与图像结果反馈给模型；
4. 最终统一返回 ``PlannerResult``。

具体 SDK 延迟导入，避免仅解析 CLI/环境配置时就加载所有模型后端及其依赖。
"""

from __future__ import annotations

import os
import queue
from pathlib import Path
from typing import Protocol

from rpent.dashboard.events import DashboardEventSink
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import (
    get_memory_dir,
    get_repo_root,
)

#: Claude/Codex 的 MCP 工具使用 ``mcp__<server>__<tool>`` 命名空间；Toolkit
#: 内部始终保存简短名称。前缀转换只发生在 Planner/SDK 边界。
MCP_TOOL_PREFIX = "mcp__rpent__"


def add_mcp_prefix(name: str) -> str:
    """为 Toolkit 工具名添加 RPent MCP 命名空间，并避免重复添加。"""
    if name.startswith(MCP_TOOL_PREFIX):
        return name
    return f"{MCP_TOOL_PREFIX}{name}"


def strip_mcp_prefix(name: str) -> str:
    """从 SDK 返回的名称中移除 RPent MCP 前缀。"""
    return name.removeprefix(MCP_TOOL_PREFIX)


class PlannerResult:
    """一次 Planner 会话返回给 CLI 的统一、可序列化结果。

    不同 SDK 的原始事件格式差异很大；CLI 只依赖这里的 finish、messages、stats
    和 error 四个字段来写 transcript 和日志。
    """

    __slots__ = ("finish_result", "messages", "stats", "error")

    def __init__(
        self,
        *,
        finish_result: dict | None = None,
        messages: list[dict] | None = None,
        stats: dict | None = None,
        error: str | None = None,
    ):
        """保存 Planner 结束状态、对话、统计和可恢复错误。"""
        # finish 工具的约定结果：status 通常为 success/failure/stuck。
        self.finish_result = finish_result

        # 已转换为可写入 transcript 的后端消息；CLI 还会移除内联图片 payload。
        self.messages = messages or []

        # 后端可扩展统计字段，常见项包括 token、turn、tool_calls 和耗时。
        self.stats = stats or {}

        # Planner 捕获的错误字符串。正常完成时为 None。
        self.error = error


class Planner(Protocol):
    """所有高层推理后端必须满足的结构化协议。"""

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """运行多轮 LLM/工具闭环，直到 finish、错误或预算耗尽。

        Args:
            system_prompt: 环境渲染的角色、规则与工作流说明。
            user_message: 当前任务的初始用户消息。
            toolkit: 通用工具与环境工具的完整注册表；Planner 从中获取 schema，
                并通过 ``execute_tool`` 执行每个模型工具调用。
            max_turns: 模型最大轮次预算。
            input_queue: 可选终端 steering 队列，仅交互 CLI 使用。
            dashboard_interaction: 可选 Dashboard 双向交互通道。

        Returns:
            统一的 ``PlannerResult``。
        """
        ...


# ---------------------------------------------------------------------------
# Planner construction
# ---------------------------------------------------------------------------


def build_planner(
    planner_type: str,
    *,
    output_dir: str | Path,
    recipe_tag: str,
    env_name: str,
    base_url: str | None = None,
    model: str | None = None,
    max_tokens: int = 8192,
    planner_timeout_s: int | None = None,
    claude_code_max_budget_usd: float | None = None,
    dashboard_events: DashboardEventSink,
    no_images: bool = False,
):
    """根据 CLI ``--planner`` 构造具体后端，并解析其默认值。

    这里不启动环境/VLA/SAM3 服务；它只创建高层推理对象。三个后端最终都遵循
    ``Planner.solve``，因此 ``cli.main`` 后续流程不需要分支。
    """
    # 延迟 import 也避免循环依赖：api_loop/claude_code/codex 都要从本模块导入
    # PlannerResult 和 MCP 名称辅助函数。

    if planner_type == "api":
        # API 后端必须用 provider:model 形式显式指定模型，避免无法判断协议与凭证。
        if not model:
            raise ValueError(
                "the 'api' planner requires a model id; pass --model with a "
                "provider prefix (e.g. 'anthropic:claude-opus-4-8', "
                "'openai:gpt-5.5', 'openai-chat:glm-5.2')."
            )

        import inspect

        from pydantic_ai.models import infer_model
        from pydantic_ai.providers import infer_provider, infer_provider_class

        from rpent.planner.api_loop import ApiAgentLoop

        def _provider_factory(provider_name: str):
            """构造 Pydantic AI provider，并可覆盖其 API base URL。

            API key 始终由 provider 自身从环境变量读取。仅在用户提供 ``base_url``
            时检查构造函数是否支持该参数，从而兼容不同 provider 实现。
            """
            if not base_url:
                return infer_provider(provider_name)
            provider_cls = infer_provider_class(provider_name)
            params = inspect.signature(provider_cls.__init__).parameters
            kwargs = {}
            if "base_url" in params:
                kwargs["base_url"] = base_url
            return provider_cls(**kwargs)

        api_model = infer_model(model, provider_factory=_provider_factory)
        api_timeout_s = planner_timeout_s
        if api_timeout_s is None:
            api_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))
        return ApiAgentLoop(
            model=api_model,
            max_tokens=max_tokens,
            dashboard_events=dashboard_events,
            no_images=no_images,
            timeout_s=api_timeout_s,
        )

    if planner_type == "claude_code":
        from rpent.planner.claude_code import ClaudeCodePlanner

        # CLI 显式值优先，其次使用跨后端 CELL_TIMEOUT_S，最后默认 1200 秒。
        cc_timeout_s = planner_timeout_s
        if cc_timeout_s is None:
            cc_timeout_s = int(os.environ.get("CELL_TIMEOUT_S", "1200"))

        # Claude Agent SDK 额外支持会话美元预算。
        cc_budget = claude_code_max_budget_usd
        if cc_budget is None:
            cc_budget = float(os.environ.get("MAX_BUDGET_USD", "10"))

        return ClaudeCodePlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model or "sonnet",
            timeout_s=cc_timeout_s,
            max_budget_usd=cc_budget,
            # 允许 Agent 访问持久化环境记忆，但不扩大到整个用户目录。
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"claude_{recipe_tag}.txt",
            dashboard_events=dashboard_events,
        )

    if planner_type == "codex":
        from rpent.planner.codex import CodexPlanner

        # Codex 专属 CODEX_TIMEOUT_S 的优先级高于通用 CELL_TIMEOUT_S。
        cx_timeout_s = planner_timeout_s
        if cx_timeout_s is None:
            cx_timeout_s = int(
                os.environ.get(
                    "CODEX_TIMEOUT_S",
                    os.environ.get("CELL_TIMEOUT_S", "1200"),
                )
            )
        return CodexPlanner(
            output_dir=output_dir,
            repo_root=get_repo_root(),
            model=model,
            timeout_s=cx_timeout_s,
            extra_dirs=[str(get_memory_dir(env_name))],
            output_path=Path(output_dir) / f"codex_{recipe_tag}.txt",
            dashboard_events=dashboard_events,
        )

    # argparse 正常情况下已限制 choices；保留显式错误方便测试和程序化调用。
    raise ValueError(f"unknown planner_type: {planner_type}")
