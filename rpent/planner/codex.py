"""Codex SDK planner.

A thin, SDK-first backend. ``solve()`` prepares artifacts, drives one Codex
SDK turn, and assembles a ``PlannerResult``. RPent tools are exposed via an
in-process HTTP MCP server (``HttpMcpServer``) so the Codex binary can reach
the shared toolkit without spawning a subprocess; this backend does not
register tools in process. Event rendering and stats live in a single
``_Recorder``. Compare with ``claude_code.py`` which uses the in-process
``create_sdk_mcp_server`` instead of an HTTP transport.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import openai_codex

from rpent.cli.tui import next_user_line
from rpent.dashboard.events import (
    DashboardEventSink,
    TranscriptEvent,
    UsageEvent,
)
from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.dashboard.planner_control import DashboardPlannerControl
from rpent.planner.base import PlannerResult, strip_mcp_prefix
from rpent.planner.utils.http_mcp_server import HttpMcpServer
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("codex")

PROVIDER_ID = "rpent_proxy"
PROVIDER_ENV_KEY = "RPENT_CODEX_PROVIDER_KEY"

# =============================================================================
# 中文实现导读
# =============================================================================
#
# 本文件是 RPent 与 OpenAI Codex SDK 之间的适配层。这里的“Codex”不是简单的
# 单次 HTTP 文本补全，而是由 ``openai_codex`` Python SDK 驱动 Codex binary：
# binary 负责模型会话、thread/turn 和工具路由；RPent 负责机器人环境、Toolkit、
# Dashboard 状态，以及把最终执行结果统一包装成 ``PlannerResult``。
#
# 一、进程、线程和网络边界
# ------------------------
#
# 一次典型调用包含四个逻辑组件：
#
# 1. RPent 主进程
#    创建 ``CodexPlanner`` 和唯一的 ``Toolkit``。物理环境、VLA/SAM3 客户端及
#    Dashboard event sink 均由这个 Toolkit 所在进程持有。
# 2. Codex SDK / Codex binary
#    ``openai_codex.Codex`` 或 ``AsyncCodex`` 管理 thread、turn 及事件流；模型
#    请求和 MCP 调用由 Codex 一侧发起，本文件只消费 SDK 产生的结构化事件。
# 3. RPent HTTP MCP 服务
#    ``HttpMcpServer`` 在当前进程的 daemon thread 中运行 uvicorn，并监听
#    ``127.0.0.1`` 的临时端口。Codex binary 通过 Streamable HTTP 调用它，
#    因而不需要再启动一个持有第二份 Toolkit 的 MCP 子进程。
# 4. Toolkit 工具执行线程
#    MCP 的异步 ``call_tool`` 使用事件循环的 executor 调用同步
#    ``toolkit.execute_tool``，避免机器人动作、VLA RPC 等长操作阻塞 uvicorn
#    事件循环。Toolkit 自己再保证同一物理环境上至多有一个活动工具操作。
#
# 数据路径可概括为：
#
#   Codex model
#       -> Codex binary 选择 MCP 工具
#       -> HTTP POST 到本进程 HttpMcpServer
#       -> executor 中调用 Toolkit.execute_tool
#       -> MCP 文本/图像 content 返回 Codex
#       -> SDK stream 发出 item/completed 等事件
#       -> _Recorder 写日志、更新 Dashboard 和统计信息
#
# 因为 HTTP 服务和普通 ``solve`` worker 都是 daemon thread，进程退出时它们不会
# 阻止解释器结束。正常路径仍必须显式 ``stop``/``close``；daemon 只是异常退出的
# 最后保险，并不代表可以忽略资源清理。
#
# 二、三种互斥运行模式
# --------------------
#
# ``CodexPlanner.solve`` 通过两个可选参数选择模式：
#
# * 非交互模式：``input_queue`` 和 ``dashboard_interaction`` 都是 ``None``。
#   主线程启动同步 worker，worker 创建一个 thread 和一个 turn，持续消费事件，
#   直到模型结束、报错或外层超时。
# * 终端 steering 模式：提供 ``input_queue``。主体仍是同步 worker，但 turn 内
#   额外启动 ``codex-steer`` daemon thread。终端新消息调用 ``turn.steer`` 注入
#   当前 turn；队列中的 ``None`` 表示中断。
# * Dashboard 模式：提供 ``dashboard_interaction``。此路径使用 ``AsyncCodex``、
#   ``_CodexDashboardSession`` 和 ``DashboardPlannerControl``，可以连续创建 turn、
#   steering 当前 turn、响应 Esc 中断，并在新任务替换旧任务时安全取消工具。
#
# 终端队列和 Dashboard port 不能同时提供；两者分别属于线程阻塞模型和 asyncio
# 控制模型。若混用，消息确认、中断及 turn 完成计数将没有唯一所有者，因此入口
# 直接抛出 ``ValueError``。
#
# 三、CodexPlanner.solve 的同步路径
# --------------------------------
#
# ``solve`` 首先拼接 system prompt 与用户任务，然后创建三个 artifact 路径和一个
# ``_Recorder``。MCP 服务必须先 ready，之后 Codex 配置才能引用确定的 URL。
#
# 同步 SDK 的 ``turn.stream()`` 是阻塞迭代器，因此不能直接占用调用 ``solve`` 的
# 线程。``codex-sdk`` worker 执行 ``_run_session``，主线程只负责：
#
# * 以 ``timeout_s`` 等待 worker；
# * 超时时通过 ``_interrupt`` 先中断当前 turn，再关闭 Codex context；
# * 追加 timeout/error 记录，而不是覆盖已经产生的事件日志；
# * 中断后再给 worker 15 秒 grace period 完成流关闭和上下文清理；
# * 无论成功、错误还是超时，都在 ``finally`` 停止 MCP HTTP 服务。
#
# ``state`` 是主线程与 worker 的小型交接字典：worker 依次放入 ``codex``、
# ``thread``、``turn``、最终文本或异常；外层用它执行中断并取得结果。正常完成时
# ``Thread.join`` 也形成状态可见边界。超时且 grace period 后仍存活时，worker 是
# daemon，不会永久阻止进程退出，但此时 MCP 已关闭，残留 turn 不能再可靠调用工具。
#
# 错误优先级是：外层 timeout/worker exception 优先，其次是 Recorder 从 Codex
# ``error``、``fatal`` 或失败 turn 中解析出的错误。即便发生错误，已经生成的文本、
# token 统计和 artifact 路径仍会放入 ``PlannerResult``，便于复盘而非只返回异常。
#
# 四、_run_session：thread、turn、事件流与终端 steering
# -----------------------------------------------------
#
# ``openai_codex.Codex`` 是同步 context manager。进入 context 后：
#
# 1. ``thread_start`` 创建可承载多个 turn 的 Codex thread；
# 2. 本路径只调用一次 ``thread.turn(prompt)`` 创建初始 turn；
# 3. ``turn.stream`` 逐项产生 SDK notification；
# 4. 每个原始事件先写 ``.stream.jsonl``，再交给 Recorder 投影成人类日志；
# 5. context 退出时关闭 SDK/Codex 资源。
#
# 开启终端 steering 后，``next_user_line`` 在单独线程等待队列。普通字符串会先写
# ``[user]`` 日志，再通过 ``turn.steer`` 发送；``None`` 会调用 ``turn.interrupt``。
# ``write_lock`` 只保护两个线程共同写入 ``chunks`` 和可读日志，原始 SDK stream
# 仍只由 worker 写入，因此不需要同一把锁。
#
# turn 结束时设置 ``stop_steer``，并向 input_queue 回填 ``None``，用于唤醒可能仍
# 阻塞在队列读取上的 steering thread。该线程本身也是 daemon；这里不 join 它，
# 避免终端输入源异常时反向卡住 Codex worker。
#
# 五、Dashboard 异步路径
# ---------------------
#
# ``_solve_dashboard`` 在 ``asyncio.run`` 创建的事件循环中工作，并使用：
#
# * ``AsyncCodex``：异步启动 thread、turn 和消费 stream；
# * ``DashboardPlannerControl``：后端无关的消息队列、Esc 中断及任务替换控制器；
# * ``_CodexDashboardSession``：把 Codex 的当前 turn 暴露成 control 所需 driver；
# * ``DashboardInteractionPort``：Dashboard Session 的并发安全状态投影接口。
#
# ``emit_event`` 对每个 SDK 事件执行与同步路径相同的“原始 JSONL + Recorder + 可读
# 文本”三重投影。``emit_user`` 则记录 Dashboard 用户消息；初始 prompt 只显示占位
# 文本，避免把很长的 system prompt 重复写入界面，同时发出 ``initial_prompt`` 事件。
#
# 初始化顺序很重要：先创建 Codex thread 并成功提交 initial turn，再调用
# ``control.start`` 开放 Dashboard 输入。这样若后端连初始提交都失败，前端不会短暂
# 显示为可接受消息的活动会话。
#
# Dashboard timeout 使用 ``asyncio.wait_for``。无论主体如何退出，清理顺序都是：
# 先让 Toolkit 的活动操作到达可取消边界，再关闭 Codex session，最后停止 MCP。
# Toolkit 清理异常只在尚无更早错误时成为最终错误，避免遮蔽真正的模型/turn 故障。
#
# 六、_CodexDashboardSession 的 turn 状态机
# ----------------------------------------
#
# Session 的核心不变量是：一个 Codex thread 可以经历多个 turn，但任一时刻只有
# ``self._turn`` 指向的一个活动 turn；对应的消费任务和完成事件分别保存在
# ``_turn_task``、``_turn_done``。
#
# ``submit`` 有两种语义：
#
# * 当前 turn 存在：调用 ``turn.steer(text)``，返回 0。消息加入既有 turn，未新增
#   一个未来的 ``turn/completed``，所以 control 不增加 outstanding completion。
# * 当前无 turn：调用 ``thread.turn(text)``，创建消费 task，返回 1。control 据此
#   记录还有一个待完成的后端请求，并将 planner activity 设为 busy。
#
# 这正是 Dashboard 中“模型正在思考时补充指令”和“模型空闲后开始新一轮”的区别；
# UI 都表现为发消息，但 Codex 协议层一个是 steer，另一个是新 turn。
#
# ``interrupt`` 先请求 Codex 中断，再最多等待 15 秒看到匹配的完成边界。正常收到
# ``turn/completed`` 时，消费协程会完成状态结算，故返回 0；若等待超时，则取消本地
# consume task、主动清空 turn 引用并返回 1，让 control 修正未能自然结算的 completion
# 计数。没有活动 turn 时返回 0，保证重复 Esc 是幂等的。
#
# ``close`` 同样先尝试 interrupt，并等待最多 15 秒；随后无论消费任务成功、取消或
# 抛错，都抑制清理阶段异常并关闭 AsyncCodex。``_closing`` 防止重复 close，也阻止
# turn 完成回调在关闭过程中继续启动下一轮 Dashboard 工作。
#
# 七、_consume_turn 与 Dashboard 控制协作
# --------------------------------------
#
# 每个事件都先交给 ``emit_event``，之后才检查控制语义：
#
# * 工具 item 完成时调用 ``control.tool_completed``，让等待中的用户消息尽早 flush。
#   若当前 turn 仍活动，flush 最终走 ``steer``，不会平白新建 turn。
# * Recorder 的 ``turns`` 达到预算且还未 finish 时，请求 interrupt；``limit_reached``
#   确保同一事件流只发一次中断请求。
# * 只有 ``turn/completed`` 才是释放当前 turn、设置 done event 的可靠协议边界。
# * failed turn 保存 Recorder 中更具体的错误；有错误、finish 或预算耗尽时结束
#   interaction，否则调用 ``control.complete`` 结算当前 completion 并发送排队消息。
#
# 注意本后端的 ``turns`` 由 Recorder 在完整 ``agentMessage`` item 到达时递增，代表
# 可见的模型回答轮次，而不是 MCP 工具数，也不一定等同于 thread 中创建的 turn 数。
# 工具调用由独立的 ``tool_calls`` 计数。
#
# 如果 stream 在没有 ``turn/completed`` 的情况下结束，这是协议不完整，不按正常
# 完成处理。异常路径必须清空 turn、唤醒所有等待者、记录字符串错误并封闭 control，
# 防止 Dashboard 永久停留在 busy。
#
# 八、_Recorder：事件投影而非会话驱动器
# --------------------------------------
#
# Recorder 不创建 thread/turn，也不执行工具。它是纯事件适配层，一次 observe 同时
# 可能更新四类输出：人类可读文本、Dashboard transcript、累计 usage、finish/error。
#
# 主要 Codex notification 的处理方式：
#
# * ``thread/started``、``turn/started``：写简短系统边界；
# * ``item/completed``：按 item 类型渲染用户、回答、reasoning 或工具结果；
# * ``thread/tokenUsage/updated``：以 SDK 的累计 total 替换本地 usage，并通知面板；
# * ``turn/completed``：写状态、耗时、token 和工具调用汇总，提取 turn error；
# * 名称含 ``requestApproval``：留下审批痕迹。当前配置是 deny_all，不做交互批准；
# * ``error``、``fatal``：截断序列化后的 payload 并记录为 Recorder error。
#
# ``agentMessage`` 更新 ``final_response`` 并增加 turns；因此 ``.last`` 始终保存最后
# 一个完整 agent message，而非整个 transcript。``reasoning`` 只投影到 thinking 事件，
# 不会被误当作最终回答或消耗 turn 预算。
#
# 工具 item 包括 ``mcpToolCall``、``dynamicToolCall``、``commandExecution`` 和
# ``fileChange``。MCP 名称会去掉 ``mcp__rpent__`` 前缀，以匹配 Toolkit 的短名称。
# Dashboard 分开发送 tool_call 与 tool_result；可读日志只保存摘要，避免把图片 base64、
# 大段 stdout 或文件内容重复写入文本 artifact。原始事件仍保留在 JSONL 中。
#
# 九、finish 协议
# ---------------
#
# RPent 不是把任意自然语言“完成了”视为任务完成，而是寻找成功结束的 ``finish``
# MCP/dynamic tool item。只有同时满足以下条件才设置 ``finish_result``：
#
# * 去掉 MCP namespace 后工具名大小写无关地等于 ``finish``；
# * item 状态为空或 ``completed``；
# * item 没有 error；
# * arguments 是字典，或是可解析成字典的 JSON 字符串。
#
# 捕获结果会补入 ``{"_finish": True, ...}``。首次有效 finish 胜出，后续重复事件
# 不覆盖它；失败的 finish 调用也不能误终止任务。
#
# 十、三个输出文件
# ----------------
#
# ``_output_paths`` 生成：
#
# * 主文件（通常 ``codex_<recipe>.txt``）：面向人的 transcript，持续 flush；
# * ``<主文件>.stream.jsonl``：每行一个原始 SDK method/payload，面向故障诊断；
# * ``<主文件>.last``：最后一条完整 agent response，便于其他程序直接读取。
#
# 未显式给 output_path 时，用 ``NamedTemporaryFile(delete=False)`` 只分配唯一路径，
# 真正内容随后由 session 以写模式生成。JSONL 每条写入后都 flush，进程意外终止时
# 也尽量保留最后一个完整事件。同步路径的 timeout/error 是追加记录，Dashboard 路径
# 则在已经打开的流内写入对应事件。
#
# 十一、Codex 配置与自定义 Responses provider
# -------------------------------------------
#
# ``_build_config`` 把 MCP URL 写入 Codex config override，并复制当前环境变量。若设置
# ``CODEX_API_KEY``，只将值放入 ``RPENT_CODEX_PROVIDER_KEY`` 环境项；配置文本保存的
# 是 env key 名称，不是密钥本身。
#
# 设置 ``CODEX_BASE_URL`` 时，``_codex_mcp_config_overrides`` 注册名为
# ``rpent_proxy`` 的 provider：
#
# * 去掉 base URL 尾部斜杠；
# * 若末尾不是 ``/v1``，自动补 ``/v1``；
# * 固定 ``wire_api=responses``；
# * 通过 ``env_key=RPENT_CODEX_PROVIDER_KEY`` 让 binary 读取凭证。
#
# 因此自定义地址必须兼容 OpenAI Responses wire API，而不仅是传统 chat completions。
# 未设置 base URL 时不覆盖 Codex 默认 provider。``CODEX_BIN`` 可替换 SDK 启动的
# binary 路径，主要用于本地安装或调试。
#
# ``experimental_api=False`` 是工具兼容关键项：关闭 namespace tools、web search、
# image generation 等实验 API，让 binary 在内部把 namespace MCP 工具转换成普通
# function tools，同时保留名称到 MCP namespace 的路由映射。
#
# ``_turn_options`` 对每个 thread/turn 统一使用 deny_all、full_access、repo cwd 和
# 可选 model。full_access 是 Codex sandbox 级别，不会绕过 RPent Toolkit 自身的工具
# 注册、串行执行和取消规则。``extra_dirs`` 当前仅保存在实例中，尚未写入 Codex
# config；不能仅因构造器收到该参数就假定 binary 已获得对应目录配置。
#
# 十二、SDK/Pydantic 数据兼容辅助函数
# ------------------------------------
#
# 不同 ``openai_codex`` 版本的 notification payload 可能是普通 dict、Pydantic model、
# RootModel 或枚举。文件尾部的小函数集中吸收这些表示差异：
#
# * ``_unwrap``：若对象有 ``root``，取得 RootModel 内层值；
# * ``_get``：对 dict 使用 ``get``，对模型/对象使用 ``getattr``；
# * ``_jsonable``：优先 ``model_dump(mode="json")``，递归处理容器，并只记录 bytes
#   大小，防止二进制直接进入 JSON；
# * ``_status``：兼容枚举的 ``value`` 与普通字符串；
# * ``_message_to_json``：只保留 notification 的 method 和 JSON-safe payload；
# * ``_int_attr``：将缺失或空 token 字段安全归零。
#
# 这些函数不是业务状态机，而是 SDK 边界的防腐层。新增事件类型时应优先在 Recorder
# 中显式处理，不能让“能序列化”被误解为“已具备正确业务语义”。
#
# ``_extract_text`` 会遍历字符串/列表，并在检测到 data URI 或明显 base64 图片时返回
# ``<image omitted>``，避免 transcript 膨胀。``_summarise_item`` 只保留路径、状态、
# exit code、截断后的命令及大型字段尺寸；``_short_json`` 则限制错误 payload 长度。
#
# 十三、错误和资源清理边界
# ------------------------
#
# 清理遵循“先停止产生新工作，再释放传输层”的原则：活动 Toolkit 操作先请求取消并
# 等待安全边界，Codex turn/session 随后 interrupt/close，MCP server 最后 stop。
# 对于同步路径，MCP stop 位于 ``finally``；对于 Dashboard 路径，session 清理位于
# 内层 ``finally``，MCP stop 位于外层 ``finally``，保证任一初始化后异常都能回收。
#
# `_interrupt` 和 close 中的部分异常被刻意抑制，因为它们运行在首个错误之后；清理
# 异常不应覆盖 timeout、模型失败或工具失败的根因。与之相对，正常事件消费中的协议
# 缺失不会被忽略，而会转成 session error。
#
# 最终结果中的 error 合并顺序也遵循“最接近控制入口的根因优先”：同步路径使用外层
# error 再回退 Recorder；Dashboard 使用 timeout/主体/cleanup error，再回退 session
# error，最后回退 Recorder。无论哪个错误胜出，raw stream 都是排查底层事件顺序的
# 权威 artifact。
#
# ---------------------------------------------------------------------------
# Public backend
# ---------------------------------------------------------------------------


class CodexPlanner:
    """Planner backed by the OpenAI Codex Python SDK."""

    def __init__(
        self,
        *,
        output_dir: str,
        dashboard_events: DashboardEventSink,
        repo_root: str | Path | None = None,
        timeout_s: int = 600,
        extra_dirs: list[str] | None = None,
        output_path: str | Path | None = None,
        model: str | None = None,
    ):
        """Initialize the Codex SDK backend."""
        # TaskRun 日志和 artifact 所在目录；Codex 的命令工作目录由 repo_root 单独指定。
        self._output_dir = str(output_dir)
        self._repo_root = str(repo_root) if repo_root else str(get_repo_root())
        # 同步 worker 和 Dashboard 整体会话共用该 wall-clock 上限。
        self._timeout_s = timeout_s
        # 为 Planner 工厂接口保留的额外目录；当前 CodexConfig 尚未消费该字段。
        self._extra_dirs = extra_dirs or []
        # 显式路径用于稳定 transcript 命名；未提供时每次 solve 分配临时路径。
        self._output_path = Path(output_path) if output_path else None
        # 显式构造参数优先于 CODEX_MODEL；None 让 Codex 使用账户/配置默认模型。
        self._model = model or os.environ.get("CODEX_MODEL", None)
        # 自定义 endpoint 必须支持 Responses wire API；API key 只通过环境传给 binary。
        self._base_url = os.environ.get("CODEX_BASE_URL", None)
        self._api_key = os.environ.get("CODEX_API_KEY", None)
        self._dashboard_events = dashboard_events
        # thread_start 与每次 turn 复用同一 options，保证模型、cwd 和审批策略一致。
        self._turn_options = {
            # RPent 无交互审批通道；未预先允许的 Codex 原生敏感操作一律拒绝。
            "approval_mode": openai_codex.ApprovalMode.deny_all,
            "cwd": self._repo_root,
            "model": self._model,
            # full_access 是 Codex sandbox 级别；RPent MCP 工具仍受 Toolkit 注册表控制。
            "sandbox": openai_codex.Sandbox.full_access,
        }

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
        """Run one or more Codex SDK turns for the given prompt."""
        if input_queue is not None and dashboard_interaction is not None:
            # 终端 steering 和 Dashboard 分别拥有不同的 ACK、中断及 completion 所有权。
            raise ValueError(
                "input_queue and dashboard_interaction cannot be used together"
            )
        # Codex turn 只有单一 prompt 参数；Dashboard 展示仍保留独立 initial_user_text，
        # 避免把 system prompt 当作用户消息显示。
        prompt = f"{system_prompt}\n\n{user_message}" if system_prompt else user_message
        if dashboard_interaction is not None:
            # Dashboard 全异步路径使用 AsyncCodex，不创建下面的同步 worker thread。
            return asyncio.run(
                self._solve_dashboard(
                    prompt=prompt,
                    initial_user_text=user_message,
                    toolkit=toolkit,
                    max_turns=max_turns,
                    interaction=dashboard_interaction,
                )
            )
        # 普通与终端模式共享同步 Codex SDK worker 和三个输出 artifact。
        output_path, raw_stream_path, last_message_path = self._output_paths()
        recorder = _Recorder(
            max_turns=max_turns,
            dashboard_events=self._dashboard_events,
        )
        # 主线程与 daemon worker 的小型交接区：worker 依次发布 codex/thread/turn/text/error。
        # 正常 join 是其可见性边界；超时路径只用已有引用执行尽力中断。
        state: dict[str, Any] = {}

        # Start the in-thread MCP HTTP server so Codex can reach the
        # shared toolkit without spawning a subprocess.
        # 必须先获得 ready URL，再构造 Codex config 并启动 worker。
        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        logger.info("mcp http endpoint: %s", mcp_url)

        model_desc = self._model or "(configured default)"
        logger.info("prompt: %d chars", len(prompt))
        logger.info("output_dir: %s", self._output_dir)
        logger.info(
            "invoking Codex SDK model %s (timeout=%ds)",
            model_desc,
            self._timeout_s,
        )

        started = time.time()
        # 同步 turn.stream() 可能长期阻塞，因此放入 daemon worker；调用 solve 的主线程
        # 只负责 timeout、中断和最终 PlannerResult 组装。
        worker = threading.Thread(
            target=self._run_session,
            args=(
                prompt,
                output_path,
                raw_stream_path,
                last_message_path,
                recorder,
                state,
                mcp_url,
                input_queue,
            ),
            name="codex-sdk",
            daemon=True,
        )
        worker.start()
        error: str | None = None
        try:
            worker.join(timeout=self._timeout_s)

            if worker.is_alive():
                # 超时后先中断 turn/关闭 codex context，再追加错误；不覆盖已经落盘的流。
                error = f"Codex SDK timed out after {self._timeout_s}s"
                _interrupt(state)
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                logger.info(rendered.rstrip())
                # 给 stream/context manager 有限 grace period；worker 为 daemon，不能无限阻塞。
                worker.join(timeout=15)
            elif "error" in state:
                # worker 捕获的 SDK/文件/steering 异常经 state 回传到统一错误协议。
                exc = state["error"]
                error = f"{type(exc).__name__}: {exc}"
                rendered = f"\n[codex-planner] {error}\n"
                with open(output_path, "a") as out_f:
                    out_f.write(rendered)
                with open(raw_stream_path, "a") as raw_f:
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                logger.info(rendered.rstrip())
        finally:
            # 正常、worker 错误和 timeout 都停止 localhost MCP；否则端口/daemon 会泄漏。
            mcp_server.stop()

        elapsed = time.time() - started
        # worker 正常完成时优先使用内存 chunks；极早失败则回读已写入的部分文件。
        text = state.get("text", "") or output_path.read_text(errors="replace")
        # 控制入口捕获的 timeout/异常优先，Recorder 事件级错误作为回退。
        error = error or recorder.error

        logger.info("Codex SDK finished in %.1fs", elapsed)
        logger.info("output: %s", output_path)
        logger.info("raw stream: %s", raw_stream_path)

        return PlannerResult(
            # 仅成功 completed 的 finish 工具 item 才设置，不能用普通自然语言替代。
            finish_result=recorder.finish_result,
            # 可读 transcript 聚合成统一后端消息；完整事件仍保存在 stream JSONL。
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(elapsed, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            error=error,
        )

    # -- internal session --------------------------------------------------

    def _output_paths(self) -> tuple[Path, Path, Path]:
        if self._output_path is None:
            # NamedTemporaryFile 仅分配唯一名称，delete=False 保证 solve 后 artifact 可追踪。
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".out", prefix="codex_sdk_task_", delete=False
            ) as file_obj:
                output_path = Path(file_obj.name)
        else:
            output_path = self._output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
        return (
            # 主文件：面向人；stream：完整事件；last：最后一个完整 agentMessage。
            output_path,
            output_path.with_suffix(output_path.suffix + ".stream.jsonl"),
            output_path.with_suffix(output_path.suffix + ".last"),
        )

    def _run_session(
        self,
        prompt: str,
        output_path: Path,
        raw_stream_path: Path,
        last_message_path: Path,
        recorder: "_Recorder",
        state: dict[str, Any],
        mcp_url: str,
        input_queue: "queue.Queue[str | None] | None" = None,
    ) -> None:
        try:
            # 仅 worker 线程写 Codex stream；steering 线程只共享 chunks/out_f。
            chunks: list[str] = []
            with openai_codex.Codex(config=self._build_config(mcp_url)) as codex:
                # 尽早发布引用，外层 timeout 即使发生在 thread/turn 初始化中也能尽力 close。
                state["codex"] = codex
                thread = codex.thread_start(**self._turn_options)
                state["thread"] = thread

                with (
                    open(output_path, "w") as out_f,
                    open(raw_stream_path, "w") as raw_f,
                ):
                    # stream worker 与 codex-steer 都会写可读 transcript，必须串行化。
                    write_lock = threading.Lock()

                    turn = thread.turn(prompt, **self._turn_options)
                    state["turn"] = turn

                    stop_steer: threading.Event | None = None
                    if input_queue is not None:
                        # 终端模式才创建输入泵；非交互路径不额外启动线程。
                        stop_steer = threading.Event()

                        def _steer() -> None:
                            while True:
                                # 同步 queue 等待留在 daemon thread，不阻塞 Codex stream worker。
                                nxt = next_user_line(input_queue)
                                if stop_steer.is_set():
                                    # turn 已结束时丢弃随后被唤醒的 EOF，不重复 interrupt。
                                    return
                                if nxt is None:
                                    try:
                                        # /quit/EOF 请求当前 turn 中断；异常只属清理路径。
                                        turn.interrupt()
                                    except Exception:
                                        pass
                                    return
                                rendered = f"\n[user] {nxt}\n"
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                                try:
                                    # steer 注入当前活动 turn，不创建新的 turn/completed 边界。
                                    turn.steer(nxt)
                                except Exception as e:
                                    rendered = f"\n[codex-planner] steer failed: {e}\n"
                                    with write_lock:
                                        chunks.append(rendered)
                                        out_f.write(rendered)
                                        out_f.flush()
                                    logger.info(rendered.strip())
                                    return

                        threading.Thread(
                            target=_steer,
                            name="codex-steer",
                            daemon=True,
                        ).start()

                    try:
                        for event in turn.stream():
                            # 原始事件先落 JSONL，再经 Recorder 生成可读文本和 Dashboard 事件。
                            _write_jsonl(raw_f, _message_to_json(event))
                            if rendered := recorder.observe(event):
                                with write_lock:
                                    chunks.append(rendered)
                                    out_f.write(rendered)
                                    out_f.flush()
                                logger.info(rendered.strip())
                    finally:
                        if stop_steer is not None:
                            # stream 无论正常、异常或 interrupt 结束，都让输入线程停止。
                            stop_steer.set()
                            if input_queue is not None:
                                # 唤醒仍阻塞在 queue.get 的 next_user_line；线程是 daemon 不 join。
                                input_queue.put(None)

            # 退出 Codex context 后再发布完整文本，主线程正常 join 后读取。
            state["text"] = "".join(chunks)
            if recorder.final_response is not None:
                # `.last` 只保存最后一个完整 agentMessage，便于程序直接消费最终回答。
                last_message_path.write_text(recorder.final_response)
        except Exception as e:
            # worker 不跨线程抛异常，通过 state 交给 solve 统一记录和返回。
            state["error"] = e

    async def _solve_dashboard(
        self,
        *,
        prompt: str,
        initial_user_text: str,
        toolkit: Toolkit,
        max_turns: int,
        interaction: DashboardInteractionPort,
    ) -> PlannerResult:
        """Run a controllable sequence of turns on one Codex thread."""
        output_path, raw_stream_path, last_message_path = self._output_paths()
        recorder = _Recorder(
            max_turns=max_turns, dashboard_events=self._dashboard_events
        )
        # Dashboard 多个 turn 共用 Recorder 与 chunks，统计和 transcript 在 TaskRun 内累计。
        chunks: list[str] = []
        error: str | None = None
        started = time.time()

        # AsyncCodex binary 仍通过 localhost HTTP MCP 访问当前进程的唯一 Toolkit。
        mcp_server = HttpMcpServer(toolkit)
        mcp_url = mcp_server.start()
        try:
            with open(output_path, "w") as out_f, open(raw_stream_path, "w") as raw_f:

                def emit_event(event: Any) -> None:
                    # 与同步路径保持相同投影顺序：raw -> Recorder -> 可读 transcript。
                    _write_jsonl(raw_f, _message_to_json(event))
                    rendered = recorder.observe(event)
                    if rendered:
                        chunks.append(rendered)
                        out_f.write(rendered)
                        out_f.flush()

                def emit_user(text: str, *, initial: bool = False) -> None:
                    # 初始 SDK prompt 合并了 system 指令，日志仅写占位符；后续消息写原文。
                    display = (
                        "[initial task instructions submitted]" if initial else text
                    )
                    rendered = f"\n[user] {display}\n"
                    chunks.append(rendered)
                    out_f.write(rendered)
                    out_f.flush()
                    self._dashboard_events.emit(
                        TranscriptEvent(
                            {"type": "initial_prompt"}
                            if initial
                            else {"type": "user", "text": text}
                        )
                    )

                control = DashboardPlannerControl(
                    interaction=interaction,
                    cancel_active_and_wait=toolkit.cancel_active_and_wait,
                    emit_user=emit_user,
                    emit_initial_user=lambda: emit_user(
                        initial_user_text, initial=True
                    ),
                )
                # Session 作为 Control driver：submit 返回新增 completion 数，interrupt 返回
                # 未自然结算的 completion 数；Control 据此维护 busy/idle。
                session = _CodexDashboardSession(
                    config=self._build_config(mcp_url),
                    options=self._turn_options,
                    recorder=recorder,
                    emit_event=emit_event,
                    control=control,
                )
                try:
                    await asyncio.wait_for(session.run(prompt), timeout=self._timeout_s)
                except asyncio.TimeoutError:
                    # wait_for 取消主体后立即封存输入；session.close 在 finally 中清理 turn。
                    error = f"Codex SDK timed out after {self._timeout_s}s"
                    control.end()
                    _write_jsonl(raw_f, {"type": "timeout", "message": error})
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    control.end()
                    _write_jsonl(raw_f, {"type": "error", "message": error})
                finally:
                    try:
                        # 先停止同步环境动作，再 interrupt/close Codex，避免工具在会话后继续。
                        await control.cancel_active_toolkit()
                    except Exception as exc:
                        cleanup_error = (
                            "Codex toolkit cancellation failed: "
                            f"{type(exc).__name__}: {exc}"
                        )
                        logger.warning(cleanup_error)
                        # 主 timeout/模型错误优先，清理错误只在此前无根因时成为最终错误。
                        error = error or cleanup_error
                    await session.close()
        finally:
            # session 清理后再关闭 MCP transport，避免活动 turn 的最后调用突然断链。
            mcp_server.stop()

        if recorder.final_response is not None:
            # Dashboard 多 turn 中持续覆盖 final_response，最终只落最后一条完整回答。
            last_message_path.write_text(recorder.final_response)
        text = "".join(chunks)
        return PlannerResult(
            finish_result=recorder.finish_result,
            messages=[{"role": "codex_sdk", "content": text}],
            stats={
                "backend": "codex_sdk",
                "elapsed_s": round(time.time() - started, 1),
                "output_chars": len(text),
                "output_path": str(output_path),
                "raw_stream_path": str(raw_stream_path),
                "last_message_path": str(last_message_path),
                "last_message_chars": len(recorder.final_response or ""),
                **recorder.stats(),
            },
            # 错误优先级：外层 timeout/cleanup > session 协议/turn > Recorder 事件错误。
            error=error or session.error or recorder.error,
        )

    # -- config builder ----------------------------------------------------

    def _build_config(self, mcp_url: str) -> Any:
        # 复制当前进程环境供 Codex binary 使用，不原地污染 os.environ。
        env = {**os.environ}
        if self._api_key:
            # 配置 override 只引用环境变量名，密钥值不会写入 config 或日志。
            env[PROVIDER_ENV_KEY] = self._api_key
        kwargs: dict[str, Any] = {
            "config_overrides": tuple(
                _codex_mcp_config_overrides(
                    mcp_url=mcp_url,
                    base_url=self._base_url,
                )
            ),
            "cwd": self._repo_root,
            "env": env,
            # Disable experimental API features (namespace tools,
            # web_search, image_generation).  This forces the binary to
            # convert namespace MCP tools to function tools internally
            # while preserving its own name→namespace mapping so that
            # ``function_call`` responses can be routed through MCP.
            "experimental_api": False,
        }
        if codex_bin := os.environ.get("CODEX_BIN"):
            # 允许使用指定的本地 Codex binary，主要用于安装路径差异或调试版本。
            kwargs["codex_bin"] = codex_bin
        return openai_codex.CodexConfig(**kwargs)


class _CodexDashboardSession:
    """Own one Codex thread and its current interruptible turn."""

    def __init__(
        self,
        *,
        config: Any,
        options: dict[str, Any],
        recorder: "_Recorder",
        emit_event,
        control: DashboardPlannerControl,
    ) -> None:
        # config/options 在整个 Dashboard TaskRun 内固定；一个 Codex thread 可创建多个 turn。
        self._config = config
        self._options = options
        self._recorder = recorder
        self._emit_event = emit_event
        self._control = control
        # 以下三个引用按 AsyncCodex -> thread -> 当前 turn 的所有权层级建立。
        self._codex: Any | None = None
        self._thread: Any | None = None
        self._turn: Any | None = None
        # done 由匹配的 turn/completed 设置；interrupt/close 用它等待协议完整结束。
        self._turn_done: asyncio.Event | None = None
        self._turn_task: asyncio.Task[Any] | None = None
        self._closing = False
        self.error: str | None = None

    async def run(self, prompt: str) -> None:
        # 必须先成功创建 client/thread/initial turn，再调用 control.start 开放输入。
        self._codex = openai_codex.AsyncCodex(self._config)
        self._thread = await self._codex.thread_start(**self._options)
        await self.submit(prompt)
        await self._control.start()
        # Control 循环等待 Dashboard 版本变化，并把本 Session 当作 driver 调用。
        await self._control.run(self)

    async def submit(self, text: str) -> int:
        if self._closing or self._thread is None:
            raise RuntimeError("Codex conversation is closed")
        if self._turn is not None:
            # 活动 turn 中的 Dashboard 输入走 steer，不产生新的 turn/completed。
            await self._turn.steer(text)
            # Control 不增加 outstanding completion；当前 turn 原有 completion 仍在。
            return 0
        # idle 时创建新 turn，并由唯一消费 task 读取其全部 notification。
        self._turn = await self._thread.turn(text, **self._options)
        self._turn_done = asyncio.Event()
        self._turn_task = asyncio.create_task(
            self._consume_turn(self._turn, self._turn_done)
        )
        # 新 turn 将来需要一个 turn/completed 结算，因此返回 1。
        return 1

    async def interrupt(self) -> int:
        # 捕获当前引用，避免 await 期间 completion handler 清空 self 字段导致错配。
        turn = self._turn
        done = self._turn_done
        if turn is None or done is None:
            # 无活动 turn 时重复 Esc/任务替换是幂等 no-op。
            return 0
        await turn.interrupt()
        try:
            # 优先等待 SDK 发出可靠 turn/completed，由 consume task 正常结算状态。
            await asyncio.wait_for(done.wait(), timeout=15)
        except asyncio.TimeoutError:
            if self._turn_task is not None:
                # SDK 未给完成边界，只能取消本地消费者并主动释放 turn 槽位。
                self._turn_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self._turn_task
            self._turn = None
            self._turn_done = None
            self._turn_task = None
            # 返回 1 告诉 Control：一个 completion 未经正常 complete，需要手工扣除。
            return 1
        # ``_consume_turn`` reports the matching completed turn boundary.
        return 0

    async def close(self) -> None:
        if self._closing:
            # close 可被外层 finally 和错误路径重复调用，必须幂等。
            return
        # 先封住新 submit，再处理当前 turn。
        self._closing = True
        turn = self._turn
        done = self._turn_done
        if turn is not None and done is not None:
            with contextlib.suppress(Exception):
                await turn.interrupt()
            try:
                # 给 SDK 正常关闭 turn 的机会，保留完整事件和统计。
                await asyncio.wait_for(done.wait(), timeout=15)
            except asyncio.TimeoutError:
                if self._turn_task is not None:
                    self._turn_task.cancel()
        if self._turn_task is not None:
            # 消费 task 的清理异常不能覆盖使 close 被调用的首个根因。
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await self._turn_task
        if self._codex is not None:
            with contextlib.suppress(Exception):
                await self._codex.close()
# turn.stream()      # 持续读取该回合产生的事件
# turn.steer(text)   # 向正在执行的回合追加指导
# turn.interrupt()   # 请求中断该回合
# turn.run()         # 等待回合完成并汇总结果
    async def _consume_turn(self, turn: Any, done: asyncio.Event) -> None:
        # 防止达到 max_turns 后对流中的每个后续事件重复发送 interrupt。
        limit_reached = False
        try:
            async for event in turn.stream(): # 它会不断收到该回合的事件 item/started item/completed item/agentMessage/delta thread/tokenUsage/updated turn/completed
                # 先记录事件，让 Recorder 更新 finish/turn/tool/usage，再做控制判断。
                self._emit_event(event)
                method = str(_get(event, "method", "")) # 说明发生了什么事件；
                payload = _get(event, "payload") # 该事件的详细数据。

                if (
                    method == "item/completed"
                    and _is_tool_item(_get(payload, "item"))
                    and self._recorder.finish_result is None
                    and self._recorder.turns < self._recorder.max_turns
                ):
                    # 工具结果已完整落地，是把 pending Dashboard 消息 steer 进当前 turn 
                    # 的安全边界；finish/预算终点则禁止再 flush。
                    await self._control.tool_completed(self)

                if (
                    method != "turn/completed"
                    and not limit_reached
                    and self._recorder.finish_result is None
                    and self._recorder.turns >= self._recorder.max_turns
                ):
                    limit_reached = True
                    # 预算达到后请求 SDK 中断，但继续读取直到可靠 completed 边界。
                    await turn.interrupt()

                if method != "turn/completed":
                    continue
                status = _status(_get(payload, "turn"))  #Codex 事件 event 携带的具体数据对象。
                # 只有匹配的 turn/completed 才释放活动槽位并唤醒 interrupt/close 等待者。
                self._turn = None
                self._turn_done = None
                done.set()
                if self._closing:
                    return
                if status == "failed":
                    self.error = self._recorder.error or "Codex turn failed"
                if (
                    self.error is not None
                    or self._recorder.finish_result is not None
                    or self._recorder.turns >= self._recorder.max_turns
                ):
                    # 错误、finish 和预算耗尽都结束交互，不允许旧 TaskRun 再接收消息。
                    self._control.end()
                else:
                    # 正常完成结算一个 outstanding completion，并可能从 idle 开启下一 turn。
                    await self._control.complete(self)
                return
            # stream 静默结束但缺少 completed 会使 completion 永远悬挂，按协议错误处理。
            raise RuntimeError("Codex turn stream ended without turn/completed")
        except Exception as exc:
            # 所有异常路径都释放 turn 并唤醒等待者，避免 interrupt/close 卡满 15 秒。
            self._turn = None
            self._turn_done = None
            done.set()
            self.error = f"{type(exc).__name__}: {exc}"
            if not self._closing:
                self._control.end()


# ---------------------------------------------------------------------------
# Observation layer
# ---------------------------------------------------------------------------


@dataclass
class _Recorder:
    """Pure adapter: consume Codex SDK events, emit text + accumulate stats."""

    max_turns: int
    dashboard_events: DashboardEventSink
    # turns 只在完整 agentMessage item 到达时递增；reasoning 和工具 item 不占 turn。
    turns: int = 0
    # tool_calls 按 completed 的 Codex tool item 计数，包含 MCP、命令和文件修改。
    tool_calls: int = 0
    # tokenUsage/updated 提供累计 total，故每次整体覆盖而不是增量相加。
    usage: dict[str, int] = field(
        default_factory=lambda: {
            "total_input_tokens": 0,
            "total_cached_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_output_tokens": 0,
        }
    )
    # 始终保存最后一个完整 agentMessage，供 `.last` artifact 使用。
    final_response: str | None = None
    # 首个合法且成功 completed 的 finish 工具参数。
    finish_result: dict[str, Any] | None = None
    # error/fatal notification 或失败 turn 中提取的人类可读错误。
    error: str | None = None

    def stats(self) -> dict[str, int]:
        # 返回普通 JSON-safe 快照，不暴露内部 usage dict 引用。
        return {"turns_used": self.turns, "tool_calls": self.tool_calls, **self.usage}

    def observe(self, event: Any) -> str:
        # notification method 是稳定分派键；payload 可能是 dict、Pydantic model 或 RootModel。
        method = str(_get(event, "method", ""))
        payload = _get(event, "payload")

        if method in {"thread/started", "turn/started"}:
            # 生命周期边界只写可读日志，不影响 turn/tool 统计。
            return f"[codex-system] {method}\n"
        if method == "item/completed":
            return self._render_item(_get(payload, "item"))
        if method == "thread/tokenUsage/updated":
            # SDK 给的是 thread 累计 total；更新后同时发布 Dashboard UsageEvent。
            self._set_usage(_get(payload, "token_usage"))
            return ""
        if method == "turn/completed":
            return self._render_turn_completed(_get(payload, "turn"))
        if "requestApproval" in method:
            # 当前 deny_all 不提供批准交互，但保留事件痕迹便于诊断被拒绝操作。
            return f"[codex-approval] {method}\n"
        if method in {"error", "fatal"}:
            # 错误 payload 可能很大，只裁剪人类日志；raw JSONL 已保存完整 JSON-safe 数据。
            self.error = _short_json(_jsonable(payload), limit=500)
            return f"[codex-error] {self.error}\n"
        # 未识别事件仍由 emit_event 写入 raw stream，不在可读 transcript 猜测其语义。
        return ""

    # -- per-item handlers -------------------------------------------------

    def _render_item(self, item: Any) -> str:
        # RootModel 先解包，后续统一用 dict/object 兼容访问。
        item = _unwrap(item)
        item_type = str(_get(item, "type", ""))

        if item_type == "userMessage":
            # SDK echo 的用户消息只写可读日志；Dashboard 用户事件由 Control emit_user 发布。
            text = _extract_text(_get(item, "content"))
            return f"\n[codex][user] {text}\n" if text else ""

        if item_type in {"hookPrompt", "plan"}:
            # 内部 hook/plan 不作为模型最终回答，也不占 RPent turn。
            return ""

        if item_type == "agentMessage":
            text = str(_get(item, "text", "")).strip()
            if not text:
                return ""
            # `.last` 每次覆盖；turns 按完整可见回答计数。
            self.final_response = text
            self.turns += 1
            self.dashboard_events.emit(TranscriptEvent({"type": "text", "text": text}))
            return (
                f"\n[agent] === turn {self.turns}/{self.max_turns} ===\n"
                f"[codex] {text}\n"
            )

        if item_type == "reasoning":
            text = _extract_text(_get(item, "summary") or _get(item, "content"))
            if text:
                # reasoning 单独投影为 thinking，不覆盖 final_response 或增加 turns。
                self.dashboard_events.emit(
                    TranscriptEvent({"type": "thinking", "text": text})
                )
            return f"[codex-reasoning] {text}\n" if text else ""

        if _is_tool_item(item):
            # item/completed 才进入本分支，因此计数代表已结算工具项。
            self.tool_calls += 1
            if item_type in {"mcpToolCall", "dynamicToolCall"}:
                # 去掉 MCP namespace，使 Dashboard/finish 协议使用 Toolkit 短名。
                name = strip_mcp_prefix(str(_get(item, "tool", item_type)))
                self._maybe_capture_finish(name, item)
            elif item_type == "commandExecution":
                name = str(_get(item, "command", item_type))
            else:
                name = "fileChange"
            # 可读/Dashboard 只保存尺寸、状态、路径等摘要；raw stream 保留完整 item。
            payload = _summarise_item(item)
            data = _jsonable(item)
            args = data.get("arguments", {}) if isinstance(data, dict) else {}
            # Codex 只在 completed 时给出完整 item，因此这里连续发布 call/result 展示对。
            self.dashboard_events.emit(
                TranscriptEvent({"type": "tool_call", "tool": name, "args": args})
            )
            self.dashboard_events.emit(
                TranscriptEvent(
                    {"type": "tool_result", "tool": name, "result": payload}
                )
            )
            return f"[tool<-] {name}: {json.dumps(payload, ensure_ascii=False)}\n"

        return ""

    def _render_turn_completed(self, turn: Any) -> str:
        # status 可能是枚举或字符串；嵌套 _get 同时兼容两者。
        status = str(_get(_get(turn, "status"), "value", _get(turn, "status", "")))
        duration_ms = _get(turn, "duration_ms")
        if error := _get(turn, "error"):
            # 优先使用结构化 message，缺失时退回整个错误对象文本。
            self.error = str(_get(error, "message", str(error)))

        parts = ["[codex-result]", status]
        if duration_ms is not None:
            parts.append(f"duration={float(duration_ms) / 1000:.1f}s")
        usage_line = (
            f"\n[usage] in={self.usage['total_input_tokens']} "
            f"cached={self.usage['total_cached_input_tokens']} "
            f"out={self.usage['total_output_tokens']} "
            f"reasoning={self.usage['total_reasoning_output_tokens']} "
            f"tool_calls={self.tool_calls}"
        )
        return " ".join(p for p in parts if p) + usage_line + "\n"

    # -- helpers -----------------------------------------------------------

    def _set_usage(self, usage: Any) -> None:
        if usage is None:
            # 并非每个 SDK 版本/事件都携带 token_usage，缺失时保留上一累计快照。
            return
        # tokenUsage/updated 通常把计数放在 total 下；兼容直接传 total 对象的版本。
        total = _get(usage, "total", usage)
        self.usage = {
            "total_input_tokens": _int_attr(total, "input_tokens"),
            "total_cached_input_tokens": _int_attr(total, "cached_input_tokens"),
            "total_output_tokens": _int_attr(total, "output_tokens"),
            "total_reasoning_output_tokens": _int_attr(
                total, "reasoning_output_tokens"
            ),
        }
        # DashboardState 对 UsageEvent 整体替换，故这里发送累计值而非 delta。
        self.dashboard_events.emit(
            UsageEvent(
                inp=self.usage["total_input_tokens"],
                out=self.usage["total_output_tokens"],
                tool_calls=self.tool_calls,
            )
        )

    def _maybe_capture_finish(self, name: str, item: Any) -> None:
        if self.finish_result is not None:
            # 首个合法 finish 胜出，重复事件不能覆盖任务终态。
            return
        if name.lower() != "finish":
            return
        status = _status(item)
        if status and status != "completed":
            # running/failed/cancelled 的 finish 均不能结束 Planner。
            return
        if _get(item, "error") not in (None, ""):
            return
        data = _jsonable(item)
        args = data.get("arguments") if isinstance(data, dict) else None
        if isinstance(args, str):
            try:
                # SDK 版本可能保留 JSON 字符串；只接受最终解析为 object 的参数。
                args = json.loads(args)
            except Exception:
                args = None
        if isinstance(args, dict):
            self.finish_result = {"_finish": True, **args}


# ---------------------------------------------------------------------------
# Codex config overrides
# ---------------------------------------------------------------------------


def _codex_mcp_config_overrides(
    *,
    mcp_url: str,
    base_url: str | None,
) -> list[str]:
    config: list[tuple[str, Any]] = [
        # Codex binary 通过 Streamable HTTP 连接当前进程的 RPent MCP server。
        ("mcp_servers.rpent.url", mcp_url),
    ]
    if base_url:
        normalized = base_url.rstrip("/")
        if not normalized.endswith("/v1"):
            # Provider 配置要求 API 根路径；避免用户传 host 时遗漏 /v1。
            normalized = normalized + "/v1"
        config.extend(
            [
                ("model_provider", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.name", PROVIDER_ID),
                (f"model_providers.{PROVIDER_ID}.base_url", normalized),
                (f"model_providers.{PROVIDER_ID}.wire_api", "responses"),
                (f"model_providers.{PROVIDER_ID}.env_key", PROVIDER_ENV_KEY),
            ]
        )
    # Codex config_overrides 使用 key=<JSON literal> 文本，json.dumps 负责字符串转义。
    return [f"{key}={json.dumps(value)}" for key, value in config]


# ---------------------------------------------------------------------------
# SDK utilities
# ---------------------------------------------------------------------------


def _interrupt(state: dict[str, Any]) -> None:
    # 超时可能发生在初始化任意阶段，因此分别检查 turn 和 codex 引用并尽力清理。
    if (turn := state.get("turn")) is not None:
        try:
            # 先停止产生新事件/工具请求。
            turn.interrupt()
        except Exception:
            # 已有 timeout 是主错误，清理异常不能覆盖它。
            pass
    if (codex := state.get("codex")) is not None:
        try:
            codex.close()
        except Exception:
            pass


def _write_jsonl(file_obj, value: dict[str, Any]) -> None:
    file_obj.write(json.dumps(value, ensure_ascii=False, default=str) + "\n")
    # 长 turn 每条事件立即落盘，异常退出时保留尽可能完整的时序。
    file_obj.flush()


def _message_to_json(message: Any) -> dict[str, Any]:
    # raw artifact 保留 Codex notification 的 method/payload 边界，并清理不可序列化对象。
    return {
        "method": _get(message, "method", ""),
        "payload": _jsonable(_get(message, "payload")),
    }


def _jsonable(value: Any) -> Any:
    value = _unwrap(value)
    if hasattr(value, "model_dump"):
        # Pydantic SDK 模型优先自行导出 JSON mode，枚举/日期等由模型转换。
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, bytes):
        # 不把二进制内容写进 JSONL，只保留类型和大小防止日志爆炸。
        return {"type": "bytes", "size": len(value)}
    return value


def _unwrap(value: Any) -> Any:
    # Pydantic RootModel 把真实 payload 放在 root；普通对象原样返回。
    return getattr(value, "root", value)


def _get(value: Any, key: str, default: Any = None) -> Any:
    # SDK 边界统一兼容普通 dict 和 Pydantic/属性对象。
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def _is_tool_item(item: Any) -> bool:
    # 仅 completed item 的这四种类型进入统一工具摘要/计数分支。
    return str(_get(_unwrap(item), "type", "")) in {
        "mcpToolCall",
        "dynamicToolCall",
        "commandExecution",
        "fileChange",
    }


def _status(item: Any) -> str:
    value = _get(item, "status", "")
    # Pydantic Enum 取 value，普通字符串直接转换。
    return str(getattr(value, "value", value))


def _summarise_item(item: Any) -> dict[str, Any]:
    data = _jsonable(item)
    if not isinstance(data, dict):
        # 未知形状只能提供总字符规模，避免完整 repr 污染 transcript。
        return {"size": _payload_size(data)}

    summary: dict[str, Any] = {}
    # 小型诊断字段可直接保留；大 payload 在下方只记录尺寸。
    for key in ("path", "file_path", "filename", "status", "state", "exit_code"):
        value = data.get(key)
        if value not in (None, ""):
            summary[key] = value
    if command := (data.get("command") or data.get("cmd")):
        command_text = str(command)
        if len(command_text) > 200:
            # 命令只在可读摘要裁剪；raw JSONL 仍保存完整 item。
            command_text = command_text[:200] + f"...(+{len(command_text) - 200})"
        summary["command"] = command_text
    for key in ("content", "text", "output", "stdout", "stderr", "result"):
        if key in data and data[key] not in (None, ""):
            # 图片、stdout 和结果可能巨大，只暴露字符规模。
            summary[f"{key}_size"] = _payload_size(data[key])

    if not summary:
        # 至少保留非大型字段名，帮助识别未知 item 形状而不泄漏内容。
        summary["keys"] = sorted(
            key for key in data if key not in {"content", "text", "output"}
        )
    return summary


def _extract_text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, str):
        text = value.strip()
        if "data:image" in text or (
            "base64" in text and ("image" in text or "iVBOR" in text)
        ):
            # 用户/工具消息中可能内联图片；可读 transcript 用占位符防止体积爆炸。
            return "<image omitted>"
        return text
    if isinstance(value, list):
        # 递归拼接多段文本 block，并过滤空段。
        parts = [_extract_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    return ""


def _payload_size(value: Any) -> int:
    # 摘要使用字符串字符数，不承诺等于网络 bytes 或 token 数。
    return len(str(value or ""))


def _short_json(value: Any, *, limit: int) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    if len(text) <= limit:
        return text
    return text[:limit] + f"...(+{len(text) - limit})"


def _int_attr(value: Any, key: str) -> int:
    # token 字段缺失或 None 时归零，并规范为 JSON-safe int。
    return int(_get(value, key, 0) or 0)
