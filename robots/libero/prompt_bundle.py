"""把 LIBERO Prompt 内容片段组装成通用渲染器可消费的节点树。

本模块只定义两个无参工厂，不在这里拼接最终文本。system/user 各章节的正文保存在
:mod:`robots.libero.prompts` 下；工厂以有序映射表达顶层章节，并用 :class:`Numbered`
显式标记需要编号的工作流。返回值随后由 ``PromptBundle.render`` 交给
``format_prompt``：渲染器按节点类型递归生成标题和列表，再以同一份运行变量严格替换
正文中的 ``{{name}}`` 占位符；缺失变量会直接报错，而不会把未展开模板交给 Agent。

这种分层让章节顺序、展示结构与大段 Prompt 文本彼此独立，也避免 Prompt 定义依赖
LIBERO 的 RPC、仿真或模型运行时。
"""

from __future__ import annotations

from robots.libero.prompts import system as system_parts
from robots.libero.prompts import user as user_parts
from rpent.context.prompt_utils import Numbered, PromptNode


def system_prompt() -> PromptNode:
    """返回 LIBERO system Prompt 的结构化章节树。

    顶层映射的插入顺序就是最终 Prompt 的章节顺序；键由渲染器输出为一级分隔标题，
    值则引用 ``prompts.system`` 中保持原样的正文。此工厂不接收运行参数，也不执行
    变量替换，因而同一棵静态结构可供普通 CLI 与 Dashboard 共用。

    Returns:
        由标题映射和编号工作流组成的 :class:`PromptNode`。
    """
    # Python 映射保留插入顺序；调整这里的排列会改变 Agent 阅读约束的先后次序。
    return {
        "ROLE AND EVALUATION": system_parts.ROLE_AND_EVALUATION,
        "PROVEN LEVERS & LESSONS — libero_10_task seed-0 sweep solved 9/10 (READ THIS)": (
            system_parts.PROVEN_LEVERS
        ),
        "RUNTIME": system_parts.RUNTIME,
        "YOUR GOAL": system_parts.GOAL,
        "RULES (NON-NEGOTIABLE)": system_parts.RULES,
        "LOCALIZATION — how to get an object's world xyz WITHOUT GT coords": (
            system_parts.LOCALIZATION
        ),
        "FIRST-STEP ALGORITHM — agentview = IDENTITY, wrist = GEOMETRY": (
            system_parts.PERCEPTION_ALGORITHM
        ),
        # 普通 tuple 会被渲染为项目符号；Numbered 保留“先感知、后执行、再审计”的步骤语义。
        "WORKFLOW": Numbered(system_parts.WORKFLOW_STEPS),
        "KEY HYPERPARAMETERS": system_parts.KEY_HYPERPARAMETERS,
        "OUTPUT DISCIPLINE": system_parts.OUTPUT_DISCIPLINE,
    }


def user_prompt() -> PromptNode:
    """返回描述当前 LIBERO evaluation cell 的 user Prompt 章节树。

    ``CELL`` 中的 suite/task/seed、输出目录和 recipe tag 仍是模板占位符；调用端会把
    ``_parse_config`` 产生的变量与实际 ``output_dir`` 合并后统一渲染。``MODE`` 与
    ``BEGIN`` 则把 system Prompt 的完整约束收束为本次任务的感知模式和起始动作。

    Returns:
        按 ``CELL``、``MODE``、``BEGIN`` 顺序组织的 :class:`PromptNode`。
    """
    return {
        "CELL": user_parts.CELL,
        "MODE": user_parts.MODE,
        "BEGIN": user_parts.BEGIN,
    }


# 仅暴露环境描述符需要的两个 Prompt 工厂，正文常量仍由 prompts 子包管理。
__all__ = ["system_prompt", "user_prompt"]
