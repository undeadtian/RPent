"""Provider-independent tool-use agent loop built on pydantic-ai.

The loop wraps the agent's :class:`~rpent.tools.toolkit.Toolkit` as
pydantic-ai function tools and drives :class:`pydantic_ai.Agent` runs,
streaming each turn so progress is logged in real time. Task completion is
signalled by the env-provided ``finish`` tool, whose result carries ``_finish``.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import json
import queue
from collections import deque
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic_ai import Agent, BinaryContent, ModelSettings, Tool, ToolReturn
from pydantic_ai.capabilities import ProcessHistory, Thinking
from pydantic_ai.exceptions import ModelHTTPError, UsageLimitExceeded
from pydantic_ai.messages import (
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import Model
from pydantic_ai.usage import RunUsage, UsageLimits

from rpent.cli.tui import QUIT_TOKENS
from rpent.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort, DashboardMessage
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import PlannerResult
from rpent.tools.state import EnvState
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("api_loop")

#: Console-log truncation limits (characters).
_TEXT_LOG_LIMIT = 500
_ARGS_LOG_LIMIT = 250
_TOOL_LOG_LIMIT = 350
#: Cap on cumulative decoded image bytes kept in the resent request history.
_MAX_HISTORY_IMAGE_BYTES = 4 * 1024 * 1024

#: Always retain at least this many of the most recent images, even if a single
#: frame exceeds the byte budget, so the model never loses its current view.
_MIN_RECENT_IMAGES = 2


# =============================================================================
# api_loop.py 中文导读：Pydantic AI 模型、工具和交互控制闭环
# =============================================================================
#
# 一、本模块在 RPent 中的位置
# ----------------
# `build_planner(planner_type="api")` 会先通过 Pydantic AI 的 `infer_model()` 把
# `provider:model` 字符串解析为 Model 对象，再把它注入本类。`infer_model()` 只选择
# provider/model 实现并构造客户端对象；真正的模型请求发生在本文件进入
# `Agent.iter()` 并推进 run node 时。
#
#   CLI / Dashboard
#       │ system_prompt + user_message + Toolkit
#       ▼
#   ApiAgentLoop.solve()
#       │
#       ├─ 构造 Pydantic AI Agent
#       │    ├─ provider Model
#       │    ├─ RPent Toolkit -> function tools
#       │    ├─ read_image 专用工具
#       │    └─ history image pruning capability
#       │
#       ├─ Agent.iter() 发起模型请求并生成 node
#       │    ├─ CallToolsNode -> 执行工具并流式产生 tool events
#       │    └─ EndNode       -> 本次 run 正常结束
#       │
#       ├─ _ApiRunObserver -> 日志、transcript、usage、finish
#       └─ PlannerResult -> CLI 的统一后端结果
#
# 本模块不直接关心 OpenAI、Anthropic、Google 或兼容 endpoint。差异封装在注入的
# `Model` 及 provider 中。只有 AnthropicModel 在 `_build_model_settings()` 中额外开启
# instructions、工具定义和消息的 prompt cache；其他模型只设置 max_tokens。
#
# 二、`solve()` 的三种互斥运行模式
# ----------------
# 入口根据两个可选交互通道分成三种模式：
#
# 1. 非交互模式：input_queue=None，dashboard_interaction=None
#    - 只处理初始任务；
#    - 外层使用 `asyncio.wait_for(..., timeout=_timeout_s)`；
#    - 模型结束、finish、预算耗尽或错误后直接返回。
#
# 2. 终端交互模式：input_queue != None
#    - TUI 线程把用户输入放进 thread-safe queue.Queue；
#    - 当前 run 内的消息用 `run.enqueue(..., priority="asap")` 注入下一请求边界；
#    - run 结束后可通过 `asyncio.to_thread(queue.get)` 无限等待下一条输入；
#    - 不套整体 wait_for，因为用户停留在终端等待输入本来就可能没有时间上限。
#
# 3. Dashboard 模式：dashboard_interaction != None
#    - 使用 DashboardState 实现的交互端口；
#    - `_ApiDashboardSession` 把每条浏览器消息排成独立 Agent run；
#    - `DashboardPlannerControl` 协调消息 ACK、Esc 中断和任务替换；
#    - `_solve_dashboard()` 自己应用整体 timeout，并在所有退出路径取消活动 Toolkit。
#
# input_queue 与 dashboard_interaction 不能同时存在，因为两者对消息顺序、确认和中断
# 有不同所有权；若同时传入，solve() 在创建 event loop 前直接报错。
#
# 三、Pydantic AI Agent 与 node 生命周期
# ----------------
# `_build_agent()` 将以下对象组合成一个 Agent：
#
# - `self._model`：已解析的 provider Model；
# - `instructions`：RPent system prompt；空字符串转换为 None；
# - `tools`：read_image 加上 Toolkit 的所有 JSON Schema 工具；
# - `model_settings`：最大输出 token 及可选 Anthropic cache；
# - `Thinking(effort="high")`：请求 provider 支持的高强度 thinking；
# - `ProcessHistory(_prune_history_images)`：每次重发历史前裁剪旧图片。
#
# `Agent.iter(seed, message_history=...)` 返回异步 run。推进 run 时会产生不同 node：
#
# - CallToolsNode：模型响应中包含一个或多个工具调用。代码先观察整个
#   `node.model_response`，然后 `node.stream(run.ctx)` 执行工具并流式接收
#   FunctionToolCallEvent / FunctionToolResultEvent；
# - EndNode：本次模型 run 已正常结束，没有后续工具节点。
#
# 普通路径用 `async for node in run` 自动推进；Dashboard 路径用
# `node = await run.next(node)` 手动推进，目的是在每个工具边界检查新消息、中断和
# checkpoint。二者使用同一个 Agent、工具包装和 Observer，因此 transcript 语义一致。
#
# 四、turn、request、tool call 是三个不同计数
# ----------------
# 本文件中的 `observer.turns` 只在观察到 CallToolsNode 的模型响应时增加。因此它是
# “带工具调用的模型轮次”计数，不一定等于 provider HTTP 请求数，也不会为只产生
# EndNode 的最终文本自动增加。
#
# - `turns`：被 observe_response() 记录的工具调用轮次；
# - `tool_calls`：每个 FunctionToolCallEvent 增加一次，同一 turn 可包含多个；
# - `usage.requests`：Pydantic AI 记录的实际模型请求数。
#
# `UsageLimits(request_limit=max_turns + 1)` 覆盖 Pydantic AI 默认 request limit，避免
# SDK 的默认 50 次先于 RPent 手工预算生效。真正跨多个 run 的限制由
# `observer.turns >= max_turns` 执行。多留出的一个 request 允许 SDK 完成边界处理，
# 但不允许 Observer 超过 RPent 的工具轮次预算继续工作。
#
# 普通交互路径每个 `agent.iter()` 没有显式复用上一 run 的 RunUsage；最终局部变量
# `usage` 指向最后完成的 run。Dashboard 路径则把同一个 `session.usage` 显式传给每个
# 独立 run，因此 Dashboard stats 的 token/request 使用量会跨浏览器消息累计。
#
# 五、普通/终端 `_solve()` 流程
# ----------------
# 首轮 seed 是 `user_message`，transcript messages 先写入一条 user。非交互模式执行
# 一个 run；终端模式可执行多个 run，并把 `run.all_messages()` 作为下一 run 的完整历史。
#
# `_inject_pending(run)` 在每次 node 到达时非阻塞排空 queue：
#
# - None 或 QUIT_TOKENS 表示结束整个终端会话；
# - 空白行忽略；
# - 普通文本以 priority="asap" 放进 Pydantic AI pending queue；
# - 同时追加 RPent transcript messages 并写日志。
#
# 注入不会中断正在进行的 provider 请求或正在执行的工具，只保证文本进入下一个 SDK
# 允许的请求边界。queue.Queue 由 TUI 线程生产、event-loop 线程消费，因此
# get_nowait() 本身线程安全；对 run.enqueue 的修改只发生在 event-loop 线程。
#
# 若当前 run 结束而终端会话仍可继续，`_await_next()` 使用 asyncio.to_thread() 在
# worker thread 中阻塞 queue.get，避免阻塞 asyncio event loop。下一条输入成为新的
# seed，并结合前一 run 的完整 history 开启新 run。
#
# 非交互模式遇到 EndNode 立即停止。终端模式遇到 EndNode 只结束当前 run，然后等待
# 用户继续输入。finish 工具、/quit、累计 max_turns 或 queue EOF 则结束整个会话。
#
# 六、finish 工具是 RPent 的业务结束协议
# ----------------
# `finish` 不是 Pydantic AI 内建终止信号，而是 Toolkit 暴露的普通工具。Observer 在
# FunctionToolCallEvent 到达时识别：
#
#   part.tool_name == "finish"
#       -> finish_result = {"_finish": True, **工具参数}
#
# `_finish` 是 Planner 后端间统一的 sentinel。工具流仍由 node.stream() 正常处理；
# 工具节点结束后，普通路径停止 run，Dashboard 路径调用 control.end() 封存交互。
# `PlannerResult.finish_result` 最终交给 CLI 决定任务 success/failure/stuck 等业务状态。
#
# 七、Dashboard 为什么使用“独立 run + 完整历史 checkpoint”
# ----------------
# 浏览器可能在模型思考、工具执行或 idle 时追加消息，也可能随时请求 Esc 或切换任务。
# 为了让取消有明确边界，`_ApiDashboardSession` 不把所有浏览器消息塞进一个永久 run，
# 而是维护：
#
# - `_history`：上一 run 的完整 Pydantic AI ModelMessage checkpoint；
# - `_pending_prompts`：`deque[(message_id | None, text)]`；
# - `_run_task`：串行消费 pending prompts 的唯一 asyncio.Task；
# - `_active_prompt`：当前是否有 prompt 正在执行；
# - `_closing`：关闭后拒绝新提交；
# - `usage`：跨独立 run 累积的同一个 RunUsage。
#
# `_queue_prompt()` 只追加队列；若没有消费者，创建一个 `_run_pending_prompts()` task。
# 所有 prompt 仍在一个 event loop 中串行执行，不会并发调用同一 Agent/Toolkit。
#
# 初始 prompt 没有 Dashboard message_id；浏览器后续消息带 ID。由于 control 配置
# `defer_message_ack=True`，消息从 DashboardState 的 pending 变为 sending 后不会立即
# 标为 sent，只有消费者真正 pop 并开始该 prompt 时才调用 message_started()。这样
# 排队后又因中断被清除的消息可以恢复为 unsent，而不会错误显示为已发送给模型。
#
# 八、Dashboard 消息到工具边界的调度
# ----------------
# `_run_agent()` 手动推进 node。每处理完一个 CallToolsNode：
#
# 1. Observer 记录模型响应；
# 2. node.stream() 执行全部工具事件；
# 3. 每个工具结果完成后 `control.tool_completed()` 尝试 flush 浏览器 pending 消息；
# 4. 若 `_pending_prompts` 非空，先 `run.next(node)` 推进一步，再退出当前 run；
# 5. finally 从 `run.all_messages()` 保存完整历史；
# 6. 消费者用新消息作为 seed 开启下一独立 run。
#
# 这意味着 Dashboard 新消息不会粗暴插入正在执行的工具中间，而是在工具完成的安全
# 边界切换 run。Toolkit 自身的取消另有 cancel_active_and_wait 协议。
#
# 九、取消、Esc 与任务替换
# ----------------
# DashboardPlannerControl 区分两种外部控制：
#
# - Esc 中断：claim_interrupt_request() 成功后，先在线程中
#   cancel_active_and_wait()，再调用 session.interrupt()；完成后清除中断状态，交互可
#   回到 idle/busy 并继续接收消息；
# - 任务替换：同样先取消 Toolkit 和 run，但随后 complete_task_replacement() 封存旧
#   TaskRun 交互，由 SessionController 切换到最新任务，不再恢复当前对话。
#
# `session.interrupt()` 会：
#
# 1. 统计一个 active prompt 加队列长度；
# 2. 清空 pending deque；
# 3. 将尚未开始且带 message_id 的消息通过 message_discarded() 恢复为 unsent；
# 4. cancel 当前 `_run_task`；
# 5. 等待 CancelledError 收敛；
# 6. 仅在引用仍指向同一 task 时清空 `_run_task`，避免覆盖新消费者。
#
# `close()` 幂等地设置 `_closing=True` 后复用 interrupt。外层 finally 无论模型成功、
# 超时还是异常，都会先取消活动 Toolkit，再关闭 session，避免遗留环境动作线程。
#
# 十、取消后的 Pydantic AI 历史修复
# ----------------
# 工具调用响应和工具结果必须在模型历史中成对出现。取消可能恰好发生在模型已经写入
# ToolCallPart、但工具 request/result 尚未完整写入 history 的窗口。finally 中会检查
# history frontier：
#
# - 若最后一条是带 tool_calls 的 ModelResponse，并且当前 node 有 `request`，追加该
#   request，让 Pydantic AI 在下一 run 中修复/解释被中断的工具结果；
# - 旧版 SDK 若没有 node.request，则移除最后这条孤立 tool-call response；
# - 其他完整历史原样保存。
#
# 这里只修复最前沿的一条不完整记录，不重写较早历史。目的是避免下一次 provider
# 请求因“tool call 缺少对应 tool result”而被 400 拒绝。
#
# 十一、Observer 同时服务日志、transcript 和结束检测
# ----------------
# `_ApiRunObserver` 由普通路径与 Dashboard 路径共享：
#
# - observe_response()：增加 turn，序列化 Text/Thinking/ToolCall，追加 transcript，
#   输出控制台日志，并把 text/thinking 发布为 Dashboard TranscriptEvent；
# - observe_tool()：工具调用开始时增加 tool_calls、发布工具名和参数；工具结果到达时
#   序列化纯文本结果、发布 is_error 与字符数摘要；识别 finish；
# - emit_usage()：发布累计 input/output token 和工具调用数。
#
# Dashboard 的 tool_result 事件只放错误标志与 size，不把完整结果重复塞入实时状态；
# 完整文本仍在 `messages` 和日志中。图片也不会进入 `_serialize_tool_result()`，避免
# transcript JSON 膨胀；视觉 BinaryContent 只存在于 Pydantic AI 模型消息历史中。
#
# 十二、Toolkit 到 Pydantic AI Tool 的桥接
# ----------------
# `_build_tools()` 总是先注册 API 后端专用 `read_image`，再遍历
# `toolkit.get_tools_spec()`。每个工具使用 `Tool.from_schema()`，保留原始 name、
# description 和 JSON input_schema；缺失 schema 时使用空 object schema。
# `takes_ctx=False` 表示 RPent wrapper 不接收 Pydantic AI RunContext。
#
# `_make_tool_function()` 生成与工具同名的 Python callable：
#
#   模型 kwargs
#       -> toolkit.execute_tool(name, kwargs)
#       -> ToolResult.content_blocks（Anthropic 风格 text/image blocks）
#       -> _content_blocks_to_pydantic()
#       -> 纯文本，或 ToolReturn(return_value=text, content=BinaryContent[])
#
# 若 `--no-images`，即使工具结果含图片也只返回文本；否则 base64 图片解码成原始 bytes
# 和 media_type 交给 Pydantic AI。没有任何文本 block 时使用字符串 "{}"，保证工具
# 始终有可序列化的 return_value。
#
# 十三、`read_image` 的 artifact 安全边界
# ----------------
# 模型只能按 `(name, step)` 读取 EnvState 已登记的 artifact：
#
# 1. `state.get(step)` 解析真实 step（-1 通常代表最新）；
# 2. `state.artifact_path()` 在状态管理的输出目录中解析路径；
# 3. name 必须存在于该 StepRecord.artifacts；
# 4. 文件必须存在；
# 5. 扩展名只允许 png/jpg/jpeg。
#
# 失败被转成 `{"error": ...}` 工具结果，而不是终止 Agent。视觉模式返回
# ToolReturn + BinaryContent；文本模式只确认 artifact 存在，并提示模型改用
# view_env_state、back_project 和数值工具，不读取图片 bytes。
#
# 十四、历史图片预算
# ----------------
# 长任务若把每次相机图都随完整 history 重发，会迅速放大请求。ProcessHistory 在每次
# 请求前调用 `_prune_history_images()`：
#
# - 只扫描 UserPromptPart.content 列表中的 image/* BinaryContent；
# - 预算按解码后的 `len(item.data)` 计算，总额 4 MiB；
# - 从最新向最旧保留；
# - 最近 `_MIN_RECENT_IMAGES=2` 张无条件保留，即使单图已超预算；
# - 其余图片只有累计 bytes 不超预算才保留；
# - 被删除项替换为文字占位符，不删除整条用户消息。
#
# 函数通过 `dataclasses.replace()` 创建新的 part/message，只复制发生变化的层级，不
# 原地修改 Pydantic AI 历史。若没有图片或全部可保留，则直接返回原列表。
#
# 十五、错误和超时语义
# ----------------
# 普通 `_solve()` 捕获 UsageLimitExceeded 并正常结束；其他异常转换为
# `"ExceptionType: message"` 写入 PlannerResult.error。非交互外层 timeout 还会同步
# 调用 toolkit.cancel_active_and_wait()，并返回只含初始 user message 的超时结果。
#
# Dashboard 在 `_solve_dashboard()` 捕获 timeout/异常后先 control.end() 封存输入，
# finally 再取消 Toolkit 和关闭 session。若清理本身失败且此前没有主错误，清理错误
# 会成为 PlannerResult.error；session 内模型异常保存在 session.error，外层错误优先。
#
# `_is_image_rejection()` 只识别 ModelHTTPError、4xx 且错误文本包含 image 的情况。若
# 当前未启用 --no-images，错误文本会追加“模型可能不支持视觉，请重试 --no-images”
# 的可操作提示。它不会把所有 400 都误判成视觉问题。
#
# 超时取消无法保证远端 provider 没有完成请求；Toolkit 取消则通过项目协议等待活动
# 环境工具到达安全停止点。调用方收到 error 后由 TaskRun 生命周期决定是否继续或清理。
#
# 十六、PlannerResult 和统计
# ----------------
# 两条 solve 路径最终都返回：
#
# - finish_result：finish 工具参数加 `_finish=True`，否则 None；
# - messages：RPent 自己的 JSON-safe user/assistant/tool transcript；
# - stats：turns_used、tool_calls，以及可用时的 input/output/cache/request usage；
# - error：后端、超时或清理错误字符串。
#
# Dashboard stats 额外带 `backend="api"`。`_serialize_response()` 仅保留 TextPart、
# ThinkingPart 和 ToolCallPart；未知 Pydantic AI part 不进入 RPent transcript。
# `_serialize_tool_result()` 对非字符串内容用 `json.dumps(default=str)`，确保 transcript
# 可落盘。日志裁剪常量只影响控制台显示，不截断真正发给模型的内容或保存的 message。
#
# 十七、线程与 event loop 边界
# ----------------
# 核心 Agent、session deque 和 run task 都只在 `asyncio.run()` 创建的 event loop 中
# 操作，不需要 threading.Lock。会阻塞的外部同步操作通过明确边界处理：
#
# - 终端 queue.get -> asyncio.to_thread；
# - DashboardState Condition 等待 -> DashboardPlannerControl 内 asyncio.to_thread；
# - Toolkit cancel_active_and_wait -> asyncio.to_thread；
# - Toolkit 工具的具体执行调度由 Pydantic AI Tool/node.stream 管理。
#
# DashboardState 自己使用 threading.Condition 和单调 interaction_version；即使状态
# notify 先于异步控制器进入等待，版本变化仍可被发现，不依赖一次性通知时序。
#
# 阅读本文件时应始终区分三类“队列”：终端 thread-safe input_queue、DashboardState
# 的消息状态机，以及 `_ApiDashboardSession` event-loop 内部 pending deque。它们不能
# 互换，也不应由同一 TaskRun 同时启用。


# 十八、按代码顺序阅读各对象
# ----------------
# 以下说明把前面的架构概念映射到下面的具体实现，便于逐行调试：
#
# ``ApiAgentLoop.__init__``
#   只保存已经由 ``build_planner`` 构造好的 Model、输出 token 上限、Dashboard sink、
#   图像开关和总超时。这里不会创建 Agent，也不会向 provider 发网络请求；Toolkit
#   仍是每次 ``solve`` 的参数，因此同一个 Planner 实例不会提前绑定某个环境状态。
#
# ``ApiAgentLoop.solve``
#   是同步 Planner 协议到 asyncio 实现的边界。Dashboard 分支运行
#   ``_solve_dashboard``；终端分支运行不带总超时的 ``_solve``；普通分支用
#   ``wait_for`` 包裹 ``_solve``。普通分支超时后还要同步等待 Toolkit 活动操作退出，
#   否则 ``asyncio.run`` 虽然结束，机器人动作线程仍可能继续推进环境。
#
# ``ApiAgentLoop._solve``
#   为普通和终端模式维护 RPent transcript、Observer、最后一次 RunUsage 和退出标志。
#   内部 ``_inject_pending`` 只做非阻塞排队；``_await_next`` 只在一次 run 已结束时
#   阻塞等待。外层 while 表示可连续创建多个 Agent run，内层 ``agent.iter`` 表示
#   一个 seed 对应的 Pydantic AI run，最内层 node/stream 则表示模型和工具事件。
#
# ``ApiAgentLoop._solve_dashboard``
#   把事件投影、DashboardPlannerControl 和 _ApiDashboardSession 组装起来。initial
#   prompt 已经作为第一条 RPent message 保存，所以 ``emit_user(initial=True)`` 只发布
#   UI 占位事件，不重复追加 message；后续浏览器消息才会同时追加 transcript。
#
# ``ApiAgentLoop._build_agent``
#   每次 TaskRun 构造一个新 Agent。``Thinking`` 是 capability 请求，不保证每个模型
#   都产生 ThinkingPart；``ProcessHistory`` 在请求前处理历史副本，不删除 EnvState
#   中的图像文件，也不修改 RPent 自己保存的 transcript。
#
# ``_ApiRunObserver``
#   不驱动 Agent，只观察已发生的模型/工具事件。``observe_response`` 以模型响应为
#   turn 边界；``observe_tool`` 区分 call/result，并返回“是否完成一个工具”的布尔值，
#   供 Dashboard 在安全边界 flush 新消息。finish 在 call 事件就被捕获，因参数已完整
#   可用；结果事件仍继续消费，保持 SDK 的工具调用历史成对。
#
# ``_ApiDashboardSession``
#   是 DashboardPlannerControl 所需的 driver。``submit``/``submit_dashboard_message``
#   都返回 1，因为每条消息最终对应一个独立 run completion；后者额外携带 message_id，
#   直到消费者真正开始执行才 ACK。``_run_task`` 是唯一消费者，保证 pending deque、
#   Agent 和单环境 Toolkit 不被多个协程并发使用。
#
# ``_ApiDashboardSession.interrupt``
#   返回被中断的 completion 数：当前 active prompt 计 1，尚未开始的 pending prompt
#   各计 1。Control 用该值修正 outstanding completion。清队列时先收集 message_id，
#   再调用 ``message_discarded``，使 UI 能把 sending 消息恢复成 unsent。
#
# ``_ApiDashboardSession._run_pending_prompts``
#   每轮 pop 一个 seed，设置 active，按需执行 deferred ACK，再调用 ``_run_agent``。
#   若 run 因 finish、预算或错误返回 False，则清除余下 seed 并结束；正常完成则通知
#   control 结算一个 completion，并由 control 决定是否继续 flush Dashboard 消息。
#
# ``_ApiDashboardSession._run_agent``
#   使用共享 ``RunUsage`` 和历史快照创建独立 run。它手动调用 ``run.next``，因为必须
#   在每个工具节点后检查 finish、预算和 pending prompt。finally 始终保存
#   ``run.all_messages``；取消窗口产生的孤立 ToolCallPart 会追加 node.request 修复，
#   无可用 request 的旧 SDK 才删除这一条不可重放的前沿响应。
#
# ``_ApiDashboardSession._process_tool_node``
#   先记录导致该节点的模型响应，再消费工具流。只有收到 FunctionToolResultEvent、
#   尚未 finish 且仍有 turn 预算时才调用 ``control.tool_completed``；工具 call 刚开始
#   时不能 flush，因为此时切换 run 会破坏 call/result 配对和环境动作完整性。
#
# ``_build_model_settings``
#   延迟导入 Anthropic 类型，避免使用其他 provider 时提前加载其可选依赖。Anthropic
#   同时缓存 instructions、工具定义和消息；其他 Model 只获得统一 ``max_tokens``。
#
# ``_prune_history_images``
#   先收集每张图在 message/part/content 三层容器中的索引与 decoded byte 数，再从新到
#   旧决定保留集合，最后仅复制发生变化的 dataclass 层级。文字占位符保留原 content
#   顺序，使模型知道该位置曾经存在图像，同时显著缩小后续请求体。
#
# ``_build_tools`` / ``_make_tool_function``
#   前者把 API 专用 read_image 和 Toolkit schema 转成 Pydantic AI Tool；后者形成闭包，
#   将模型 kwargs 交给同步 ``Toolkit.execute_tool``。ToolResult 的 Anthropic-shaped
#   content blocks 在这里转成 Pydantic AI ToolReturn，因此 Toolkit 无需依赖具体 SDK。
#
# ``read_image`` / ``read_image_text_only``
#   两者共享 artifact 解析和白名单校验。视觉版本读取 bytes 并返回 BinaryContent；
#   text-only 版本只确认文件存在并引导模型使用数值感知工具，确保 ``--no-images``
#   模式不会因模型主动调用 read_image 而偷偷发送图像。
#
# ``_serialize_*`` / ``_build_stats`` / ``_log_*``
#   分别负责持久 transcript、统一统计和人类日志。三种输出刻意分离：日志可以裁剪，
#   transcript 必须 JSON-safe，模型上下文则保留完整文字和允许的 BinaryContent。
#   因此修改日志裁剪常量不会改变模型看到的工具结果，也不会改变 token 使用量。
#
# =============================================================================


class ApiAgentLoop:
    """Planner that runs the tool-calling loop via a pydantic-ai ``Agent``."""

    def __init__(
        self,
        model: Model,
        max_tokens: int = 8192,
        no_images: bool = False,
        *,
        dashboard_events: DashboardEventSink,
        timeout_s: int | None = None,
    ):
        """Store the pydantic-ai model and the output-token cap."""
        # Model 已由 build_planner/infer_model 构造完成；这里只保存引用，不发送请求。
        self._model = model
        # 限制单次 provider 响应的最大输出 token，而不是整个会话累计 token。
        self._max_tokens = max_tokens
        # Planner、Toolkit 和运行时统一通过该 sink 向 Dashboard 投影事件。
        self._dashboard_events = dashboard_events
        # 文本模型模式下，工具仍可返回状态文本，但不会向模型历史注入图片 bytes。
        self._no_images = no_images
        # 非交互及 Dashboard 会话的整体时间上限；终端交互模式有意不使用它。
        self._timeout_s = timeout_s

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
        """Run the tool-calling loop until finish, normal stop, or budget."""
        # 终端队列和 Dashboard 各自拥有消息确认/中断语义，不能同时驱动同一会话。
        if input_queue is not None and dashboard_interaction is not None:
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        # Dashboard 路径使用可取消、可连续提交消息的异步 Session 控制器。
        if dashboard_interaction is not None:
            return asyncio.run(
                self._solve_dashboard(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    interaction=dashboard_interaction,
                )
            )
        # 普通 CLI 与终端交互共享 _solve；是否有 input_queue 决定是否连续等待输入。
        solve = self._solve(
            system_prompt=system_prompt,
            user_message=user_message,
            toolkit=toolkit,
            max_turns=max_turns,
            input_queue=input_queue,
        )
        if input_queue is not None:
            # 用户停留在终端等待输入的时间不应被 planner timeout 截断。
            return asyncio.run(solve)
        try:
            # 普通非交互调用才对完整 Agent loop 应用总超时。
            return asyncio.run(asyncio.wait_for(solve, timeout=self._timeout_s))
        except asyncio.TimeoutError:
            # asyncio task 被取消后，同步 Toolkit 动作可能仍在线程中；等待其安全退出。
            toolkit.cancel_active_and_wait()
            return PlannerResult(
                finish_result=None,
                messages=[{"role": "user", "content": user_message}],
                stats={},
                error=f"API planner timed out after {self._timeout_s}s",
            )

    async def _solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
    ) -> PlannerResult:
        # 此处才把当前 TaskRun 的 Toolkit schema、system prompt 和 Model 组装成 Agent。
        agent = self._build_agent(system_prompt, toolkit)

        # input_queue 存在表示终端可在一次 run 内 steer，也可在 run 之间继续对话。
        interactive = input_queue is not None
        # 该列表是 RPent 自己保存的 JSON-safe transcript，不是 Pydantic AI 原始历史。
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]
        observer = _ApiRunObserver(
            dashboard_events=self._dashboard_events,
            messages=messages,
            max_turns=max_turns,
        )
        # 业务错误转成字符串返回，避免 provider 异常直接穿透 Planner 协议。
        last_error: str | None = None
        # 普通路径最终保留最后一次 run 的 usage；Dashboard 路径则显式跨 run 累计。
        usage: RunUsage | None = None
        # /quit、EOF 等通过该标志结束整个终端会话，而不只是当前 run。
        quit_requested = False

        def _inject_pending(run: Any) -> bool:
            """Drain queued user lines into the live run; True => end session.

            Each line is enqueued ``asap`` so it lands in the next model request
            (the next turn boundary). This runs on the event-loop thread, so
            mutating the run's pending-message queue here is race-free.
            """
            while True:
                try:
                    # 非阻塞排空线程安全队列，绝不让 node 推进停在等待用户输入上。
                    line = input_queue.get_nowait()  # type: ignore[union-attr]
                except queue.Empty:
                    # 队列暂时为空只表示“本轮没有 steering”，当前 Agent run 应继续推进。
                    return False
                if line is None:
                    # None 是生产者发送的 EOF/关闭哨兵，要求结束整个交互会话。
                    return True
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    # /quit 等控制词只在 Planner 边界消费，不能作为普通 prompt 发给模型。
                    return True
                if not line:
                    # 忽略纯空白输入，既不占模型轮次，也不污染 transcript。
                    continue
                # asap 表示在 SDK 允许的下一个请求边界注入，不会打断正在执行的工具。
                run.enqueue(line, priority="asap")
                # 同步记录到 RPent transcript，便于最终日志还原用户 steering。
                messages.append({"role": "user", "content": line})
                logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))

        async def _await_next() -> str | None:
            """Block off-loop for the next user line between runs (None => end)."""
            logger.info("awaiting input — type a message to continue, /quit to end")
            while True:
                # queue.get 是阻塞 API，转到 worker thread，避免冻结 asyncio event loop。
                line = await asyncio.to_thread(input_queue.get)  # type: ignore[union-attr]
                if line is None:
                    return None
                line = line.strip()
                if line.lower() in QUIT_TOKENS:
                    return None
                if line:
                    logger.info("[user] %s", _clip(line, _ARGS_LOG_LIMIT))
                    return line

        # seed 是当前独立 run 的新用户输入；history 是上一 run 的完整 SDK checkpoint。
        seed = user_message
        history: list[ModelMessage] | None = None
        try:
            # 终端交互可以围绕同一历史连续创建多个独立 Agent run。
            while True:
                # 只用于本次 run 的日志编号；Observer.turns 才是整个会话累计预算。
                run_turns = 0
                # request_limit overrides pydantic-ai's default (50) so the
                # manual max_turns break below is what bounds each run.
                # 进入 iter 本身构造 run；异步推进 node 时才会真正触发模型/工具工作。
                async with agent.iter(
                    seed,
                    message_history=history,
                    usage_limits=UsageLimits(request_limit=max_turns + 1),
                ) as run:
                    # Pydantic AI node 依次表示模型请求、工具调用和最终 EndNode 等阶段。
                    async for node in run:
                        if interactive and _inject_pending(run):
                            quit_requested = True
                            break
                        if Agent.is_call_tools_node(node):
                            # 该节点携带完整模型响应，并准备执行其中一个或多个工具调用。
                            run_turns += 1
                            observer.observe_response(
                                node.model_response,
                                run.usage,
                                log_turn=run_turns,
                            )

                            # stream 依次发出工具 call/result 事件；Toolkit handler 在此阶段执行。
                            async with node.stream(run.ctx) as stream:
                                async for event in stream:
                                    observer.observe_tool(event, run.usage)

                            if observer.finish_result is not None:
                                # finish 工具已经给出任务终态；后续模型节点和排队输入均不再执行。
                                logger.info("FINISH called: %s", observer.finish_result)
                                break
                            if observer.turns >= max_turns:
                                # turn 预算按整个会话累计，而不是按当前独立 run 重新计数。
                                logger.info(
                                    "reached max_turns=%d. Stopping.", max_turns
                                )
                                break
                        elif Agent.is_end_node(node):
                            # EndNode 表示模型自然结束当前 run，且本节点没有待执行工具。
                            if interactive:
                                logger.info(
                                    "model ended turn without a tool call "
                                    "— awaiting your input."
                                )
                            else:
                                logger.info(
                                    "model ended turn without a tool call. Stopping."
                                )
                            break

                    # run.usage 在节点推进过程中累计；退出 context 前保存最终快照。
                    usage = run.usage
                    if interactive:
                        # 下一条终端消息用完整 SDK 历史开启新 run，保持工具 call/result 配对。
                        history = run.all_messages()

                # finish, quit, non-interactive, or the cumulative turn budget is
                # spent => end the whole session so max_turns is enforced across
                # every run, not per run.
                if (
                    observer.finish_result is not None
                    or quit_requested
                    or not interactive
                    or observer.turns >= max_turns
                ):
                    # 任一终止条件成立都不再读取下一条终端输入；普通模式必定只跑一次。
                    break
                nxt = await _await_next()
                if nxt is None:
                    # EOF 或 quit token 在 run 间到达，同样正常结束，不记为 Planner 错误。
                    break
                seed = nxt
                messages.append({"role": "user", "content": seed})
        except UsageLimitExceeded as e:
            # SDK 自身的 request/token guard 属于预算耗尽，不是后端故障。
            logger.info("usage limit reached: %s", e)
        except Exception as e:  # noqa: BLE001 - surfaced via PlannerResult.error
            # Provider、schema 或工具桥接异常统一降为可序列化错误，保留已完成 transcript。
            last_error = _api_error_text(e, no_images=self._no_images)
            logger.error("agent run failed: %s", last_error)

        # 即使异常或预算终止，也返回已经观察到的 finish、消息和累计统计。
        return PlannerResult(
            # 只有模型实际调用 finish 工具时才非空；自然停机不伪造成功状态。
            finish_result=observer.finish_result,
            # RPent transcript 与 SDK history 分离，因此这里可直接交给 CLI 持久化。
            messages=messages,
            # usage 可能在首个请求前失败而为 None，_build_stats 会保留基础计数。
            stats=_build_stats(usage, observer.turns, observer.tool_calls),
            # 正常结束为 None；UsageLimitExceeded 当前按预算结束处理，也保持 None。
            error=last_error,
        )

    async def _solve_dashboard(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort,
    ) -> PlannerResult:
        """Drive cancellable PydanticAI runs from complete history checkpoints."""
        # Dashboard 同样复用统一 Agent，但由独立 Session 手动控制每个 run。
        agent = self._build_agent(system_prompt, toolkit)
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message}]

        def emit_user(text: str, *, initial: bool = False) -> None:
            # 初始 user_message 已在 messages 初始化时写入，避免重复追加；后续消息才补记。
            if not initial:
                messages.append({"role": "user", "content": text})
            # 前端以 initial_prompt 触发初始任务展示，后续输入则携带实际文本。
            self._dashboard_events.emit(
                TranscriptEvent(
                    {"type": "initial_prompt"}
                    if initial
                    else {"type": "user", "text": text}
                )
            )

        # Control 只管理交互状态；具体 Agent run 生命周期由 _ApiDashboardSession 实现。
        control = DashboardPlannerControl(
            interaction=interaction,
            cancel_active_and_wait=toolkit.cancel_active_and_wait,
            emit_user=emit_user,
            emit_initial_user=lambda: emit_user(user_message, initial=True),
            # 消息真正从 pending deque 取出并开始 run 时，才在前端标记为 sent。
            defer_message_ack=True,
        )
        observer = _ApiRunObserver(
            dashboard_events=self._dashboard_events,
            messages=messages,
            max_turns=max_turns,
        )
        session = _ApiDashboardSession(
            agent=agent,
            control=control,
            observer=observer,
            max_turns=max_turns,
            no_images=self._no_images,
        )
        error: str | None = None
        try:
            # timeout 覆盖初始请求及其后的整个 Dashboard 交互会话。
            await asyncio.wait_for(
                session.run(user_message),
                timeout=self._timeout_s,
            )
        except asyncio.TimeoutError:
            # wait_for 已取消 session.run；显式结束 Control，阻止 Dashboard 再接收输入。
            error = f"API planner timed out after {self._timeout_s}s"
            control.end()
        except Exception as exc:
            # 非超时故障同样封装成人类可读错误，并关闭交互状态机。
            error = _api_error_text(exc, no_images=self._no_images)
            control.end()
        finally:
            try:
                # 先让同步环境工具到达取消边界，再关闭负责推进 Agent 的异步 Session。
                await control.cancel_active_toolkit()
            except Exception as exc:
                cleanup_error = (
                    f"API toolkit cancellation failed: {type(exc).__name__}: {exc}"
                )
                logger.warning(cleanup_error)
                # 主运行错误优先；只有此前成功时才把清理失败暴露为最终错误。
                error = error or cleanup_error
            # close 具有幂等性，会取消消费者并恢复尚未开始消息的 ACK 状态。
            await session.close()

        return PlannerResult(
            finish_result=observer.finish_result,
            messages=messages,
            stats={
                "backend": "api",
                **_build_stats(session.usage, observer.turns, observer.tool_calls),
            },
            # 外层 timeout/cleanup 错误优先于 session 内部捕获的 provider 错误。
            error=error or session.error,
        )

    def _build_agent(self, system_prompt: str, toolkit: Toolkit) -> Agent:
        """Build an Agent for terminal or Dashboard execution."""
        return Agent(
            # Model 内部封装 provider/client；首次推进请求节点时才实际访问远端 API。
            self._model,
            instructions=system_prompt or None,
            # 把当前环境 Toolkit 的 JSON schema 和 handler 包装成 Pydantic AI Tool。
            tools=_build_tools(toolkit, no_images=self._no_images),
            model_settings=_build_model_settings(self._model, self._max_tokens),
            capabilities=[
                # 请求高强度 reasoning；不支持的 provider 由 Pydantic AI/profile 处理。
                Thinking(effort="high"),
                # 每次重发历史前裁剪旧图，不改变 EnvState 落盘 artifact。
                ProcessHistory(processor=_prune_history_images),
            ],
        )


@dataclasses.dataclass
class _ApiRunObserver:
    """Record model/tool events shared by terminal and Dashboard runs."""

    dashboard_events: DashboardEventSink
    messages: list[dict[str, Any]]
    max_turns: int
    turns: int = 0
    tool_calls: int = 0
    finish_result: dict[str, Any] | None = None

    def observe_response(
        self,
        response: ModelResponse,
        usage: RunUsage,
        *,
        log_turn: int | None = None,
    ) -> None:
        # 一个 CallToolsNode 的模型响应记作一个 RPent turn；纯 EndNode 不经过这里。
        self.turns += 1
        # 转成项目自己的 JSON-safe transcript，隔离 Pydantic AI 内部消息类型。
        message = _serialize_response(response)
        self.messages.append(message)
        _log_response(
            response,
            usage,
            self.turns if log_turn is None else log_turn,
            self.max_turns,
        )
        for block in message["content"]:
            # Dashboard 仅实时展示自然语言和 reasoning；工具调用由 observe_tool 单独投影。
            if block["type"] == "text":
                # transcript 的 text 字段直接映射为前端 text 事件。
                payload = {"type": "text", "text": block["text"]}
            elif block["type"] == "thinking":
                # 持久化键名是 thinking，Dashboard 统一使用 text 承载显示内容。
                payload = {"type": "thinking", "text": block["thinking"]}
            else:
                # tool_use 等 block 在这里跳过，避免与工具事件流重复展示。
                continue
            self.dashboard_events.emit(TranscriptEvent(payload))
        # 每个模型响应后立即刷新累计 token，使 UI 不必等到工具完成。
        self.emit_usage(usage)

    def observe_tool(self, event: Any, usage: RunUsage) -> bool:
        # 返回值只表示是否看到工具“结果”事件，Dashboard 以它作为安全 flush 边界。
        completed = False
        if isinstance(event, FunctionToolCallEvent):
            # 一次模型响应可以并列发出多个调用，因此工具数独立于 turn 数。
            self.tool_calls += 1
            part = event.part
            args = part.args_as_dict()
            # 将模型刚生成的工具调用投影到 Dashboard transcript。
            # 该事件只负责通知前端“准备调用哪个工具、使用什么参数”，并不会执行工具。
            # 实际工具执行发生在后续 node.stream(run.ctx) 推进过程中。
            self.dashboard_events.emit(
                TranscriptEvent(
                    {
                        "type": "tool_call",      # 前端事件类型：工具调用开始
                        "tool": part.tool_name,   # 模型请求调用的工具名称
                        "args": args,             # 模型为该工具生成的参数字典
                    }
                )
            )

            if part.tool_name == "finish":
                # finish 是普通 Toolkit 工具；_finish sentinel 由各 Planner 统一识别。
                self.finish_result = {"_finish": True, **args}
        elif isinstance(event, FunctionToolResultEvent):
            # 结果已形成，说明同步 Toolkit handler 已返回或已转成结构化错误。
            completed = True
            message = _serialize_tool_result(event)
            self.messages.append(message)
            _log_tool_result(message)
            part = event.part
            is_error = bool(getattr(part, "is_error", False))
            self.dashboard_events.emit(
                TranscriptEvent(
                    {
                        "type": "tool_result",
                        "tool": message.get("name") or "tool_result",
                        "result": {
                            # Dashboard 只需轻量摘要；完整结果已写入 messages/transcript。
                            "is_error": is_error,
                            # size 是序列化文本字符数，不代表图片 bytes 或 token 数。
                            "size": len(message["content"]),
                        },
                    }
                )
            )
        # call 和 result 都刷新 usage；tool_calls 使用 Observer 自己的累计计数。
        self.emit_usage(usage)
        return completed

    def emit_usage(self, usage: RunUsage) -> None:
        self.dashboard_events.emit(
            UsageEvent(
                # RunUsage 在 Dashboard 多个独立 run 间共享，因此这里发布的是会话累计值。
                inp=int(usage.input_tokens or 0),
                out=int(usage.output_tokens or 0),
                tool_calls=self.tool_calls,
            )
        )


class _ApiDashboardSession:
    """Own serial, independent PydanticAI runs for one Dashboard TaskRun."""

    def __init__(
        self,
        *,
        agent: Agent,
        control: DashboardPlannerControl,
        observer: _ApiRunObserver,
        max_turns: int,
        no_images: bool,
    ) -> None:
        # Agent 封装 model、instructions、tools 与 history processor，本 Session 只负责编排。
        self._agent = agent
        # Control 连接 Dashboard 消息状态机，并决定何时 submit/interrupt/结束会话。
        self._control = control
        # turn 上限按 Observer 的会话累计值检查，不会因开启新 run 而重置。
        self._max_turns = max_turns
        # 仅用于生成视觉拒绝错误提示，不改变此处的 Session 状态机。
        self._no_images = no_images
        # terminal 与 Dashboard 共用同一种事件记录逻辑和 finish sentinel。
        self._observer = observer
        # 仅保存上一 run 完成/中断后的完整 SDK 消息历史，供下一 run 重放。
        self._history: list[ModelMessage] = []
        # 同一个 RunUsage 注入每个独立 run，使 Dashboard token/request 跨消息累计。
        self.usage = RunUsage()
        # message_id=None 代表初始 prompt；浏览器消息携带 ID 以支持延迟 ACK/恢复。
        self._pending_prompts: deque[tuple[str | None, str]] = deque()
        # 唯一消费者 task 串行处理 deque，避免并发推进同一个 Agent/物理环境。
        self._run_task: asyncio.Task[Any] | None = None
        self._active_prompt = False
        # close 后拒绝任何新 prompt，且工具完成回调不能重新启动消费者。
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        # 初始 prompt 先成功入队，再开放 Dashboard 输入，避免假“可交互”窗口。
        await self.submit(prompt)
        await self._control.start()
        await self._control.run(self)

    async def submit(self, text: str) -> int:
        """Queue Dashboard input as a new independent API run."""
        return self._queue_prompt(text)

    async def submit_dashboard_message(self, message: DashboardMessage) -> int:
        """Queue Dashboard input and defer acknowledgement until it starts."""
        return self._queue_prompt(message.text, message_id=message.message_id)

    def _queue_prompt(self, text: str, *, message_id: str | None = None) -> int:
        if self._closing:
            # close 是不可逆边界，拒绝重启已关闭 TaskRun 的消费者。
            raise RuntimeError("API conversation is closed")
        if self.error is not None:
            # 首次 Agent 故障会封存于 session.error；后续提交直接复用该根因失败。
            raise RuntimeError(self.error)
        self._pending_prompts.append((message_id, text))
        if self._run_task is None:
            # 仅从“无消费者”跃迁到“有消费者”时创建 task；后续消息只追加队列。
            self._run_task = asyncio.create_task(self._run_pending_prompts())
        # DashboardPlannerControl 把返回值解释为新增 completion 数量。
        return 1

    async def interrupt(self) -> int:
        run_task = self._run_task
        # Control 用该数量扣减尚未自然 complete 的 outstanding completion。
        interrupted = int(self._active_prompt) + len(self._pending_prompts)
        discarded_message_ids = tuple(
            message_id
            for message_id, _ in self._pending_prompts
            if message_id is not None
        )
        self._pending_prompts.clear()
        for message_id in discarded_message_ids:
            # 已 claim 但未真正开始的消息恢复为 unsent，供用户编辑或重新发送。
            self._control.message_discarded(message_id)
        if run_task is None or run_task.done():
            # 无存活消费者时无需 cancel；仍返回刚清除的 completion 数供 Control 对账。
            return interrupted
        # 取消唯一消费者会把 CancelledError 传播到当前 agent.iter/node.stream。
        run_task.cancel()
        try:
            with contextlib.suppress(asyncio.CancelledError):
                await run_task
        finally:
            # identity guard 防止旧 task 清掉其退出期间新创建的消费者引用。
            if self._run_task is run_task:
                self._run_task = None
        return interrupted

    async def close(self) -> None:
        if self._closing:
            # close 可由外层 finally 与错误路径重复调用，第二次必须无副作用。
            return
        # 先封住新 submit，再 interrupt 当前/排队工作，避免取消窗口重新入队。
        self._closing = True
        await self.interrupt()

    async def _run_pending_prompts(self) -> None:
        task = asyncio.current_task()
        try:
            while self._pending_prompts and not self._closing:
                # pop 即“真正开始提交”；此前消息只处于 sending，尚未对模型生效。
                message_id, seed = self._pending_prompts.popleft()
                self._active_prompt = True
                try:
                    if message_id is not None:
                        # deferred ACK：此时才把 Dashboard 消息标记为 sent 并写 transcript。
                        self._control.message_started(message_id, seed)
                    if not await self._run_agent(seed):
                        # finish、预算、故障或 closing 均终止会话；丢弃尚未消费的 seed。
                        self._pending_prompts.clear()
                        return
                    # 一个独立 Agent run 正常结算一个 completion，并触发 Control flush。
                    await self._control.complete(self)
                finally:
                    self._active_prompt = False
        finally:
            # 只允许当前消费者清空槽位，防止 cancel/重启竞态覆盖新 task。
            if self._run_task is task:
                self._run_task = None

    async def _run_agent(self, seed: str) -> bool:
        run_completed = False
        run: Any | None = None
        node: Any | None = None
        try:
            # 传入历史浅拷贝，避免 Agent 对请求历史的处理直接改动 checkpoint 容器。
            async with self._agent.iter(
                seed,
                message_history=list(self._history),
                usage=self.usage,
                usage_limits=UsageLimits(request_limit=self._max_turns + 1),
            ) as run:
                # Dashboard 手动推进 node，以便在每个工具安全边界检查控制状态。
                node = run.next_node
                while not Agent.is_end_node(node):
                    if Agent.is_call_tools_node(node):
                        await self._process_tool_node(run, node)
                    if (
                        self._observer.finish_result is not None
                        or self._observer.turns >= self._max_turns
                    ):
                        # finish 或累计预算耗尽是整个 Dashboard 会话终点，不再接受新输入。
                        self._control.end()
                        return False
                    if self._pending_prompts:
                        # 当前 node 仍需推进一次，让 SDK 把刚完成的工具结果写入历史。
                        # Dashboard input accepted at this tool boundary starts
                        # a fresh run from the checkpoint captured below.
                        node = await run.next(node)
                        # 不把新 prompt 注入当前 run；退出后由串行消费者开启独立 run。
                        break
                    node = await run.next(node)

                # 到达 EndNode 或安全地为 pending prompt 截断，均可保存有效 checkpoint。
                run_completed = True
        except Exception as exc:
            # CancelledError 不属于 Exception，会继续向 interrupt/close 传播；普通故障才落盘。
            self.error = _api_error_text(exc, no_images=self._no_images)
            if not self._closing:
                # 主动 close 期间无需重复 end；运行故障则立即封住 Dashboard 输入。
                self._control.end()
        finally:
            # Preserve interrupted tool results for PydanticAI to repair on the
            # next run. Older supported releases can leave a bare tool-call
            # response when cancellation wins before any tool returns; remove
            # only that unusable frontier.
            if run is not None:
                # 无论正常、异常还是 cancel，都尽量保存 SDK 已形成的完整历史前沿。
                history = list(run.all_messages())
                if (
                    history
                    and isinstance(history[-1], ModelResponse)
                    and history[-1].tool_calls
                ):
                    if request := getattr(node, "request", None):
                        # 将中断工具对应的 request/result 补入历史，保持 provider 配对约束。
                        history.append(request)
                    else:
                        # 旧 SDK 无修复 request 时，只能删掉最后一条孤立 tool-call 响应。
                        history.pop()
                self._history = history
        # 仅“形成可重放 checkpoint 且 Session 尚未关闭”时允许消费者结算并取下一 prompt。
        return run_completed and not self._closing

    async def _process_tool_node(self, run: Any, node: Any) -> None:
        # 先发布模型产生的文字/thinking/tool calls，再执行工具事件流。
        self._observer.observe_response(node.model_response, run.usage)

        async with node.stream(run.ctx) as stream:
            async for event in stream:
                tool_completed = self._observer.observe_tool(event, run.usage)
                if (
                    # 仅工具结果完成后才允许 Control 把新消息加入后续 run。
                    tool_completed
                    and self._observer.finish_result is None
                    and self._observer.turns < self._max_turns
                ):
                    await self._control.tool_completed(self)


def _build_model_settings(model: Model, max_tokens: int) -> ModelSettings:
    """Build model settings, enabling prompt caching for Anthropic models."""
    # 延迟导入避免非 Anthropic 后端在模块加载时承担其可选依赖成本。
    from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings

    if isinstance(model, AnthropicModel):
        return AnthropicModelSettings(
            max_tokens=max_tokens,
            # 三个开关分别在稳定 prompt 区域放置 Anthropic cache breakpoint。
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
            anthropic_cache_messages=True,
        )
    # 非 Anthropic provider 只接收跨模型通用字段，避免透传不认识的缓存参数。
    return ModelSettings(max_tokens=max_tokens)


def _prune_history_images(messages: list[ModelMessage]) -> list[ModelMessage]:
    """Drop old camera images so the resent request body stays bounded."""
    # Every image in history, oldest -> newest: (msg_idx, part_idx, item_idx, nbytes).
    # 四元组保存图片在 message/part/content 三层容器中的位置及 decoded byte 数。
    located: list[tuple[int, int, int, int]] = []
    for mi, message in enumerate(messages):
        for pi, part in enumerate(getattr(message, "parts", ()) or ()):
            if not isinstance(part, UserPromptPart) or not isinstance(
                part.content, list
            ):
                continue
            for ii, item in enumerate(part.content):
                if isinstance(item, BinaryContent) and item.media_type.startswith(
                    "image/"
                ):
                    located.append((mi, pi, ii, len(item.data)))

    if not located:
        # 历史中没有视觉 payload 时保留原列表身份，避免无意义复制。
        return messages

    # Walk newest -> oldest, keeping images while under the byte budget.
    # keep 只保存位置索引；从最新图片开始分配预算，保证当前观察优先。
    keep: set[tuple[int, int, int]] = set()
    total = 0
    for rank, (mi, pi, ii, nbytes) in enumerate(reversed(located)):
        # 最近图片拥有最低保留数保障，其余图片必须满足累计 byte 预算。
        if rank < _MIN_RECENT_IMAGES or total + nbytes <= _MAX_HISTORY_IMAGE_BYTES:
            keep.add((mi, pi, ii))
            total += nbytes

    if len(keep) == len(located):
        # 所有图片均落在“最近数量 + byte 预算”内，无需重建任何 dataclass。
        return messages

    # 按所属 UserPromptPart 聚合待替换的 content 索引，后续每个 part 只复制一次。
    drop_items_by_part: dict[tuple[int, int], set[int]] = {}
    for mi, pi, ii, _ in located:
        if (mi, pi, ii) not in keep:
            drop_items_by_part.setdefault((mi, pi), set()).add(ii)

    # 复制容器和 dataclass，只替换含被裁剪图片的 part，不原地污染原历史。
    new_messages = list(messages)
    for (mi, pi), drop_items in drop_items_by_part.items():
        message = new_messages[mi]
        part = message.parts[pi]
        new_content = [
            "[earlier camera image omitted to bound request size]"
            if ci in drop_items
            else item
            for ci, item in enumerate(part.content)
        ]
        new_parts = list(message.parts)
        new_parts[pi] = dataclasses.replace(part, content=new_content)
        new_messages[mi] = dataclasses.replace(message, parts=new_parts)

    return new_messages


def _is_image_rejection(e: Exception) -> bool:
    """True when the provider returned a 4xx complaining about image input.

    Matches errors like ``400 {'code': 10007, 'msg': "Bad Request: [message
    type 'image_url' is not supported]"}`` from OpenAI-compatible endpoints
    serving text-only models.
    """
    # 只依据结构化 HTTP 异常判断，普通 ValueError/工具错误不能误报为视觉问题。
    if not isinstance(e, ModelHTTPError):
        return False
    # 只把客户端请求拒绝视为输入能力问题；5xx 更可能是服务端暂时故障。
    if not 400 <= e.status_code < 500:
        return False
    # OpenAI-compatible 服务错误结构不统一，因此在完整异常文本中宽松查找 image。
    return "image" in str(e).lower()


def _api_error_text(error: Exception, *, no_images: bool) -> str:
    # PlannerResult.error 采用“异常类型: 消息”，既方便人读，也保留根因类别用于日志。
    text = f"{type(error).__name__}: {error}"
    # 仅视觉确实启用时附加 --no-images 建议，避免给已禁用图片的用户错误提示。
    if not no_images and _is_image_rejection(error):
        text += (
            "\n\nThe model rejected image input — it is likely a text-only "
            "model (no vision support). Re-run with --no-images: RPent will "
            "then keep every visual observation as a file-path text notice "
            "instead of sending image bytes."
        )
    return text


def _build_tools(toolkit: Toolkit, *, no_images: bool = False) -> list[Tool]:
    """Build the API-only image reader plus pydantic-ai toolkit wrappers."""
    # read_image 不属于通用 Toolkit schema，只为 API 后端提供多模态 artifact 读取。
    image_reader = _make_image_reader(toolkit.state, no_images=no_images)
    tools: list[Tool] = [Tool(image_reader, name="read_image")]
    for spec in toolkit.get_tools_spec():
        name = spec["name"]
        # from_schema 原样使用 Toolkit 的 JSON input schema，不依赖 Python 函数签名推导。
        tools.append(
            Tool.from_schema(
                function=_make_tool_function(toolkit, name, no_images=no_images),
                name=name,
                description=spec.get("description", ""),
                # 极简/旧 Toolkit spec 缺 schema 时降级为空对象参数，而不是拒绝注册。
                json_schema=spec.get("input_schema")
                or {"type": "object", "properties": {}},
                # wrapper 只接收模型 kwargs，不要求 Pydantic AI 注入 RunContext。
                takes_ctx=False,
            )
        )
    return tools


def _make_image_reader(
    state: EnvState,
    *,
    no_images: bool,
) -> Callable[[str, int], ToolReturn | dict[str, str] | str]:
    if no_images:

        # 即便模型主动调用 read_image，纯文本模式也绝不读取或返回图像 bytes。
        def read_image_tool(name: str, step: int = -1) -> str:
            return read_image_text_only(name, step, state=state)

        read_image_tool.__name__ = "read_image"
        read_image_tool.__doc__ = read_image_text_only.__doc__
        return read_image_tool

    # 视觉模型路径返回 ToolReturn，其中 return_value 是文本元数据，content 是图片。
    def read_image_tool(
        name: str, step: int = -1
    ) -> ToolReturn | dict[str, str]:
        return read_image(name, step, state=state)

    read_image_tool.__name__ = "read_image"
    read_image_tool.__doc__ = read_image.__doc__
    return read_image_tool


def read_image(
    name: str, step: int = -1, *, state: EnvState
) -> ToolReturn | dict[str, str]:
    """Read a step-scoped image artifact as visual input.

    Artifact failures are returned as structured tool errors so a bad
    model-supplied name or step does not abort the agent run.
    """
    try:
        # 先验证 artifact 确实属于该 step，再从 EnvState 管理目录读取数据。
        resolved_step, path = _resolve_image_artifact(state, name, step)
        content = BinaryContent(
            data=state.load_bytes(name, step=resolved_step),
            media_type=_image_media_type(path),
        )
    except Exception as e:
        # 模型给出的名称/step 属于不可信输入；转成工具错误以允许 Agent 自行纠正。
        return {"error": str(e)}
    return ToolReturn(
        return_value={"artifact": name, "step": resolved_step},
        content=[content],
    )


def read_image_text_only(
    name: str, step: int = -1, *, state: EnvState
) -> str | dict[str, str]:
    """Acknowledge an image artifact without sending bytes to the model."""
    try:
        resolved_step, _ = _resolve_image_artifact(state, name, step)
    except Exception as e:
        # 与视觉版本保持相同错误协议，但成功路径只返回说明文字、不加载文件 bytes。
        return {"error": str(e)}
    return (
        f"Image artifact {name!r} exists at step {resolved_step}, but image "
        "input is disabled (--no-images, text-only model). Reason from textual "
        "state instead: view_env_state, back_project, and numeric tool results."
    )


def _resolve_image_artifact(
    state: EnvState,
    name: str,
    step: int,
) -> tuple[int, Path]:
    # state.get 统一解析 -1 等逻辑 step；后续始终使用记录中的真实 step_idx。
    record = state.get(step)
    path = state.artifact_path(name, step=record.step_idx)
    # 同时检查登记表和文件实体，禁止模型读取未登记的任意输出目录文件。
    if name not in record.artifacts or not path.is_file():
        raise FileNotFoundError(
            f"image artifact {name!r} is not available at step {step}"
        )
    # 登记为 artifact 并不等于可作为视觉输入；扩展名白名单拒绝 npz/json 等文件。
    if path.suffix.lower() not in {".png", ".jpg", ".jpeg"}:
        raise ValueError(f"artifact {name!r} is not an image")
    return record.step_idx, path


def _image_media_type(path: Path) -> str:
    return "image/jpeg" if path.suffix.lower() in {".jpg", ".jpeg"} else "image/png"


def _make_tool_function(toolkit: Toolkit, name: str, *, no_images: bool = False):
    """Return a callable that dispatches one tool call to the toolkit."""

    def _call(**kwargs: Any) -> Any:
        # Toolkit 统一完成串行化、取消、异常转结果和状态采集。
        result = toolkit.execute_tool(name, kwargs)
        # 将项目内部 Anthropic-shaped blocks 转为 provider 无关的 Pydantic AI 内容。
        text, images = _content_blocks_to_pydantic(result.content_blocks)
        if images and not no_images:
            # 视觉模式把文本作为 return_value、图片作为额外 content 一并反馈给模型。
            return ToolReturn(return_value=text, content=images)
        # no_images 时主动丢弃转换出的图片；无图结果也直接返回更轻量的纯文本。
        return text

    # Pydantic AI 用函数名辅助工具标识/诊断，必须与 schema 中公开名称一致。
    _call.__name__ = name
    return _call


def _content_blocks_to_pydantic(
    blocks: list[dict[str, Any]],
) -> tuple[str, list[BinaryContent]]:
    """Split Anthropic-shaped content blocks into text and image content."""
    text_parts: list[str] = []
    images: list[BinaryContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            text_parts.append(block.get("text", ""))
        elif block_type == "image":
            source = block.get("source") or {}
            data = source.get("data")
            if source.get("type") == "base64" and data:
                # Toolkit 为跨 SDK 兼容保存 base64；进入 BinaryContent 前恢复原始 bytes。
                images.append(
                    BinaryContent(
                        data=base64.b64decode(data),
                        media_type=source.get("media_type", "image/png"),
                    )
                )
        # 未知 block 类型留给其他后端处理；API bridge 不猜测其二进制/文本语义。
    # 空文本结果使用 {}，保证 tool return 始终具有可序列化的文本 return_value。
    text = "\n\n".join(part for part in text_parts if part) or "{}"
    return text, images


def _serialize_response(response: ModelResponse) -> dict[str, Any]:
    """Render one assistant turn as a serialisable transcript message."""
    content: list[dict[str, Any]] = []
    for part in response.parts:
        # 只投影 RPent transcript 需要的三类 part；二进制和供应商私有 part 不落盘。
        if isinstance(part, TextPart):
            # 空 TextPart 不生成空白 transcript block。
            if part.content:
                content.append({"type": "text", "text": part.content})
        elif isinstance(part, ThinkingPart):
            # reasoning 与最终文本分开保存，前端可采用不同样式展示。
            if part.content:
                content.append({"type": "thinking", "thinking": part.content})
        elif isinstance(part, ToolCallPart):
            # 工具调用保留 provider call id，便于 transcript 对齐后续 tool result。
            content.append(
                {
                    "type": "tool_use",
                    "id": part.tool_call_id,
                    "name": part.tool_name,
                    "input": part.args_as_dict(),
                }
            )
    return {"role": "assistant", "content": content}


def _serialize_tool_result(event: FunctionToolResultEvent) -> dict[str, Any]:
    """Render one tool result as a serialisable transcript message (no images)."""
    part = event.part
    # ToolReturn 可能是 dict/list 等对象；统一转换为可持久化文本，图片不在此重复保存。
    content = getattr(part, "content", None)
    if not isinstance(content, str):
        content = json.dumps(content, default=str)
    return {
        "role": "tool",
        "name": getattr(part, "tool_name", None),
        "tool_call_id": getattr(part, "tool_call_id", None),
        "content": content,
    }


def _build_stats(
    usage: RunUsage | None, turns: int, n_tool_calls: int
) -> dict[str, Any]:
    """Assemble the run stats dict from accumulated usage and counters."""
    stats: dict[str, Any] = {"turns_used": turns, "tool_calls": n_tool_calls}
    if usage is not None:
        # RunUsage 字段可能为 None；统一归零后转成普通 int，保证 JSON 可序列化。
        stats.update(
            {
                "total_input_tokens": int(usage.input_tokens or 0),
                "total_output_tokens": int(usage.output_tokens or 0),
                "cache_read_tokens": int(usage.cache_read_tokens or 0),
                "cache_write_tokens": int(usage.cache_write_tokens or 0),
                "requests": int(usage.requests or 0),
            }
        )
    # 首个请求前失败时 usage 为 None，此时仍返回 turn/tool 基础统计。
    return stats


def _log_response(
    response: ModelResponse, usage: RunUsage, turn: int, max_turns: int
) -> None:
    """Log model text, thinking, tool calls, and cumulative usage for a turn."""
    logger.info("=== turn %d/%d ===", turn, max_turns)
    for part in response.parts:
        if isinstance(part, TextPart):
            text = (part.content or "").strip()
            if text:
                # 最终回答通常需要完整排障信息，因此不裁剪普通 model text。
                logger.info("[model] %s", text)
        elif isinstance(part, ThinkingPart):
            text = (part.content or "").strip()
            if text:
                # reasoning 可能极长，只影响日志展示，不改 transcript 或模型历史。
                logger.info("[think] %s", _clip(text, _TEXT_LOG_LIMIT))
        elif isinstance(part, ToolCallPart):
            args = json.dumps(part.args_as_dict(), default=str)
            # 工具参数可能包含长路径/数组，单行日志按独立上限裁剪。
            logger.info("[tool>] %s(%s)", part.tool_name, _clip(args, _ARGS_LOG_LIMIT))
    # RunUsage 是截至当前节点的累计值，并非该 turn 的增量。
    logger.info(
        "[usage] in=%s out=%s cache_read=%s cache_write=%s requests=%s",
        usage.input_tokens,
        usage.output_tokens,
        usage.cache_read_tokens,
        usage.cache_write_tokens,
        usage.requests,
    )


def _log_tool_result(message: dict[str, Any]) -> None:
    """Log a one-line summary of a tool result."""
    content = " ".join((message.get("content") or "").split())
    logger.info("[tool<] %s: %s", message.get("name"), _clip(content, _TOOL_LOG_LIMIT))


def _clip(text: str, limit: int) -> str:
    """Truncate ``text`` to ``limit`` characters with an overflow marker."""
    if len(text) <= limit:
        # 未超限时返回原字符串，避免改变日志中的空白和内容。
        return text
    # marker 记录被省略字符数，便于判断原始 payload 的量级。
    return text[:limit] + "...(+%d)" % (len(text) - limit)
