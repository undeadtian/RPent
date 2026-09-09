"""Common physical agent tools."""
from __future__ import annotations

import os
from pathlib import Path

from rpent.tools.toolkit import readonly
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_output_dir

# =============================================================================
# 通用工具层的职责与注册协议
# =============================================================================
#
# 该模块提供与具体机器人、仿真器和模型后端无关的四个基础工具：读取文本、写入
# 文本、列目录，以及声明任务结束。每个 Toolkit 实例初始化时都会遍历本模块的
# TOOLS_SPEC，再通过下方同名 TOOL_HANDLERS 找到 Python 实现并注册。
#
# 两个表承担不同职责：
#
# * TOOLS_SPEC 是发给 LLM/Planner 的公开协议，描述工具名、用途和 JSON 参数约束；
# * TOOL_HANDLERS 是进程内执行映射，将协议名称绑定到真正的 Python callable。
#
# 因此新增或重命名工具时，两处名称必须保持一致。schema 只约束模型应生成什么，
# 真正调用仍会经过 Toolkit.execute_tool(name, input_dict)，由它统一处理参数错误、
# ToolResult 封装和工具执行的串行化。
#
# 本文件中的 @readonly 不是文件系统的“只读权限”。它只通知 Toolkit：该调用不会
# 改变机器人/仿真环境，所以执行结束后不必额外抓取 RGB-D、机器人位姿等环境快照。
# write_text_file 虽然会修改磁盘文件，但不改变物理环境，因此同样标记为 readonly。
# =============================================================================

# Planner 可见的 Anthropic/MCP 风格工具 schema。各后端会把这些结构转换成自身需要
# 的 function-tool 格式；description 也是引导模型正确选择工具的运行时提示。
TOOLS_SPEC: list[dict] = [
    {
        # 用于读取历史 recipe、审计结果和持久化 memory。实现只返回文本，不把文件
        # 作为 Python 对象暴露给模型；max_chars 防止大文件撑满模型上下文。
        "name": "read_text_file",
        "description": (
            "Read a UTF-8 text file. Use for past recipe JSONLs, audit JSONs, "
            "and memory files. Large files are truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                # 绝对路径保持原样；相对路径会由 _resolve 锚定到仓库根目录，而不是
                # 当前 shell 的工作目录，从而避免从不同入口启动时解析结果漂移。
                "path": {"type": "string", "description": "Absolute or repo-relative path"},
                # 这是字符数而非 UTF-8 字节数，默认值由 Python handler 的形参提供。
                "max_chars": {"type": "integer", "description": "Max chars (default 40000)"},
            },
            # max_chars 未列入 required，省略时使用 handler 的 40000 默认值。
            "required": ["path"],
        },
    },
    {
        # 保存 Planner 生成的 recipe、audit 或其他文本 artifact。父目录不存在时会
        # 自动递归创建；文件已存在时 Path.write_text 会整体覆盖而非追加。
        "name": "write_text_file",
        "description": (
            "Write a UTF-8 text file (creates parent dirs). Use this to save "
            "the working recipe JSONL and the final audit JSON at the end of "
            "a successful run."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            # 两项都必需，避免缺少 content 时意外把文件覆盖为空或在 handler 展开
            # kwargs 时产生缺参错误。
            "required": ["path", "content"],
        },
    },
    {
        # 只列出目标目录的直接子项，不递归遍历。空路径有特殊含义：使用当前任务的
        # output_dir，使并行 TaskRun 各自看到自己的 artifact 目录。
        "name": "list_dir",
        "description": (
            "List files in a directory (non-recursive). Default = {{output_dir}}. "
            "Use to inspect the working directory or to discover existing "
            "recipes in resources/libero/results_*_pert/."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Default: {{output_dir}}"},
            },
            # 没有 required 字段，因此模型可以传空对象 {}，触发 output_dir 默认值。
        },
    },
    {
        # finish 是 Planner 与外层任务编排之间的显式终止协议。它不是普通自然语言
        # 回答：只有成功执行该工具并携带合法参数，后端才会提升为 finish_result。
        "name": "finish",
        "description": (
            "Call when the task is complete or unrecoverable. Halts the agent "
            "loop. Save any artifacts (recipe, audit) BEFORE calling finish."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "description": "Outcome, e.g. 'success', 'failure', or 'stuck'.",
                },
                "summary": {
                    "type": "string",
                    "description": "Short natural-language summary of the run.",
                },
            },
            # status 供外层判断业务终态，summary 供日志、Dashboard 和审计展示。
            "required": ["status", "summary"],
        },
    },
]


# ---------------------------------------------------------------------------
# 路径与输出裁剪辅助函数
# ---------------------------------------------------------------------------


def _resolve(path: str) -> Path:
    # Path 只做语法层面的路径对象转换；这里不要求目标已经存在，因此读、写和列目录
    # 可以在各自 handler 中采用不同的存在性与创建策略。
    p = Path(path)
    if not p.is_absolute():
        # 所有相对路径统一相对于 RPent 仓库根目录。注意这里不做 resolve()，所以不会
        # 主动规范化符号链接或检查路径是否越过仓库；绝对路径也被有意允许。
        p = get_repo_root() / p
    return p


def _truncate(text: str, max_chars: int) -> str:
    # 未超过预算时原样返回，避免无意义地复制或追加提示。
    if len(text) <= max_chars:
        return text
    # 超限时保留前缀，并明确告诉模型原始字符数和实际展示字符数。这里按 Python
    # Unicode 字符切片，不按 UTF-8 字节截断，因此不会切坏多字节字符编码。
    return (
        text[:max_chars]
        + f"\n\n[TRUNCATED — file is {len(text)} chars, showed first {max_chars}]"
    )


# ---------------------------------------------------------------------------
# 通用工具 handler
# ---------------------------------------------------------------------------


@readonly
def read_text_file(path: str, max_chars: int = 40000) -> dict:
    # 先统一绝对/仓库相对路径，返回值中的 path 也使用这一解析后的表示。
    p = _resolve(path)
    # 将常见输入错误编码成普通工具结果，使模型能够修正路径并继续，而不是让 Planner
    # 会话因 FileNotFoundError 或 IsADirectoryError 直接终止。
    if not p.exists():
        return {"error": f"file not found: {p}"}
    if p.is_dir():
        return {"error": f"is a directory: {p}"}
    try:
        # errors="replace" 会把无法按默认文本编码解码的字节替换为 U+FFFD，尽可能
        # 返回其余可读内容；读取失败（权限、I/O 等）仍转换为结构化 error。
        text = p.read_text(errors="replace")
    except Exception as e:
        return {"error": str(e)}
    # size 是完整文本的字符数；content 才受 max_chars 限制。调用方因此能够判断结果
    # 是否只是文件前缀，而无需解析裁剪提示文字。
    return {"path": str(p), "size": len(text), "content": _truncate(text, max_chars)}


@readonly
def write_text_file(path: str, content: str) -> dict:
    # 相对目标落在仓库根目录下；显式绝对路径则按调用方要求保留。
    p = _resolve(path)
    # parents=True 递归创建缺失目录；exist_ok=True 允许目录已经存在。
    p.parent.mkdir(parents=True, exist_ok=True)
    # write_text 是整体覆盖写入。这里不捕获异常，让 Toolkit.execute_tool 的统一异常
    # 边界把权限、磁盘空间和 I/O 错误转换为模型可见的 ToolResult。
    p.write_text(content)
    # 报告 UTF-8 编码后的 payload 字节数，而不是字符数。它描述 content 的逻辑 UTF-8
    # 大小；实际 Path.write_text 使用的平台默认编码由 Python 运行环境决定。
    return {"path": str(p), "bytes_written": len(content.encode("utf-8"))}


@readonly
def list_dir(path: str = "") -> dict:
    # Default to the current output dir (so parallel agents see their own).
    # 非空 path 采用仓库相对/绝对解析；空字符串则读取上下文中的当前 output_dir，确保
    # 并行 agent 不会错误查看其他任务的输出目录。
    p = _resolve(path) if path else get_output_dir()
    if not p.exists():
        # 与 read_text_file 相同，常见的不存在错误直接反馈给模型以便恢复。
        return {"error": f"directory not found: {p}"}
    # os.listdir 只返回一层名称；sorted 提供稳定、可复现的字典序结果。这里不读取文件
    # 内容，也不区分文件、目录和符号链接。如 p 不是目录，异常由 Toolkit 统一捕获。
    files = sorted(os.listdir(p))
    return {"path": str(p), "count": len(files), "files": files}


@readonly
def finish(status: str, summary: str) -> dict:
    """Signal that the run is complete. Halts the agent loop.

    The ``_finish`` sentinel is what each planner detects to stop the
    tool-calling loop — see ``event.part.tool_name == "finish"`` in
    :meth:`rpent.planner.api_loop.ApiAgentLoop._solve` and the
    ``pending_finish`` bookkeeping in
    :class:`rpent.planner.claude_code._Recorder`.
    """
    # _finish 是跨 API、Claude Code 和 Codex 后端统一的机器可读 sentinel。status 与
    # summary 保留模型提交的业务结论；外层不能只靠普通 assistant 文本判断任务完成。
    return {"_finish": True, "status": status, "summary": summary}


# 执行注册表必须与 TOOLS_SPEC 一一对应。Toolkit 初始化时按 spec 顺序取出同名 handler，
# 因而缺失键会在启动注册阶段暴露，而不会等到模型真正调用时才静默失败。
TOOL_HANDLERS: dict = {
    "read_text_file": read_text_file,
    "write_text_file": write_text_file,
    "list_dir": list_dir,
    "finish": finish,
}
