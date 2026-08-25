"""具体 LIBERO evaluation cell 的 user Prompt 章节正文。

system Prompt 给出跨任务的完整运行规则；本文件只补充本次运行的身份、产物位置、
感知模式和起始动作。``CELL`` 中的双花括号是统一模板变量，不在导入或节点组装时
求值：普通 CLI 或 Dashboard 会把 ``_parse_config`` 返回的 suite/task/seed 与
recipe tag，再加上最终 ``output_dir``，交给 Prompt 渲染器做严格替换。
"""

from __future__ import annotations

# 运行单元清单同时告诉 Agent 审计 JSON 与 recipe JSONL 的约定落点；路径由渲染变量生成。
CELL = """- suite:      {{suite}}
- task:       {{task}}
- seed:       {{seed}}
- output_dir: {{output_dir}}
- audit:      {{output_dir}}/{{recipe_tag}}.json
- recipe:     {{output_dir}}/recipe_{{recipe_tag}}.jsonl"""


# 用一句话声明感知隔离模式：动作前必须从图像出发，经反投影或分割得到几何位置。
MODE = """Inspect `agentview_high.png` returned by `view_env_state`, then use
`back_project` or `segment` to localize objects before motion."""


# 启动指令规定首轮顺序，避免 Agent 跳过记忆/指南或在读取初始观测前直接操作。
BEGIN = """Read MEMORY.md and the guides, then call
`view_env_state({"step": 0})` and inspect `agentview_high.png`. Localize the
target, then plan and execute."""
