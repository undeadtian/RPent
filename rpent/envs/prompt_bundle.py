"""环境贡献的 system/user Prompt 工厂及统一渲染入口。

Prompt 使用 Python 节点构建，而不是由 CLI 拼接字符串。环境包可以自由组织任务说明、
工具使用规则和记忆路径；``rpent.cli.main`` 只传入本次运行变量并请求渲染。
该模块位于 ``rpent.envs``，可避免环境 Prompt 反向依赖 RPC 或 Toolkit 实现。
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from rpent.context.prompt_utils import PromptNode, format_prompt

# 工厂先构造 PromptNode；format_prompt 再递归展开节点并替换变量。
PromptFactory = Callable[..., PromptNode]


@dataclass(frozen=True)
class PromptBundle:
    """某个环境提供的两类 Prompt 工厂。

    ``system`` 定义 Agent 角色、流程和约束；``user`` 描述当前 suite/task/seed 对应
    的初始任务。两者使用同一份 ``variables``，保证路径和任务标识一致。
    """

    system: PromptFactory
    user: PromptFactory

    def render(
        self,
        variant: str,
        *,
        variables: Mapping[str, object] | None = None,
    ) -> str:
        """构造并渲染 ``system`` 或 ``user`` Prompt。

        ``getattr`` 让调用端保持统一；未知 variant 会自然抛出 AttributeError，避免
        静默使用错误 Prompt。工厂本身不接收运行变量，变量替换集中由 format_prompt
        完成。
        """
        prompt = getattr(self, variant)()
        return format_prompt(prompt, variables=variables)
