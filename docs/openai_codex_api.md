# `openai_codex/api.py` 详细中文解析

> 本文分析 RPent Conda 环境中实际安装的 `openai_codex`，不修改 `site-packages` 源码。

## 1. 分析基线

| 项目 | 值 |
|---|---|
| Python 环境 | `/home/hirobot/anaconda3/envs/rpent` |
| 源文件 | `/home/hirobot/anaconda3/envs/rpent/lib/python3.10/site-packages/openai_codex/api.py` |
| `openai-codex` 版本 | `0.147.0` |
| 源文件行数 | 811 |
| SHA-256 | `673defd0ccf1348a86c2bb589cb3a1a69cb315b0a3ecb29525c52f0515a82476` |
| RPent 集成文件 | `rpent/planner/codex.py` |
| RPent MCP 桥 | `rpent/planner/utils/http_mcp_server.py` |

本文以该安装副本为准。重新安装或升级 `openai-codex` 后，类、参数和底层行为可能变化，应重新核对版本与 SHA-256。

## 2. 阅读结论

`api.py` 是 Codex Python SDK 的**高层门面层**。它把底层 JSON-RPC 客户端包装成六个主要对象：

- `Codex`：同步 SDK 入口；
- `AsyncCodex`：异步 SDK 入口；
- `Thread` / `AsyncThread`：持久对话线程；
- `TurnHandle` / `AsyncTurnHandle`：一次可流式消费、可 steer、可 interrupt 的运行回合。

它本身不直接调用 OpenAI HTTP API，也不直接实现 MCP。真实运行链为：

```text
Python 调用方
  -> openai_codex.api 高层对象
  -> CodexClient / AsyncCodexClient
  -> 本地 codex app-server 子进程
  -> stdio 上的逐行 JSON-RPC
  -> Codex runtime 再访问模型服务和 MCP 服务
```

同步底层 `CodexClient` 会启动：

```text
<bundled-codex-binary> [--config key=value ...] app-server --listen stdio://
```

`AsyncCodexClient` 并没有另建一套原生 asyncio stdio transport，而是持有一个同步 `CodexClient`，通过 `asyncio.to_thread()` 把阻塞操作卸载到工作线程。这个事实对于理解线程、取消和关闭行为非常重要。

## 3. 模块职责与分层

### 3.1 `api.py` 负责什么

1. 暴露用户友好的同步和异步入口；
2. 将 Python 参数组装为 generated Pydantic 请求模型；
3. 将文本、图像、技能和 mention 输入规范化为 wire input；
4. 把高级 `ApprovalMode`、`Sandbox` 映射到底层协议字段；
5. 创建 thread、启动 turn；
6. 按 turn ID 消费定向通知；
7. 将完整事件流汇总为 `TurnResult`；
8. 提供登录、账户、模型列表和线程生命周期 API；
9. 通过 context manager 保证底层进程关闭。

### 3.2 `api.py` 不负责什么

- 不直接创建 `subprocess.Popen`；该逻辑在 `client.py`；
- 不直接读取 stdout；唯一 reader thread 在 `CodexClient`；
- 不直接维护响应/通知队列；该逻辑在 `_message_router.py`；
- 不直接向 OpenAI endpoint 发 HTTP 请求；由 Codex runtime 完成；
- 不直接执行 MCP 工具；Codex runtime 连接配置中的 MCP server；
- 不负责 RPent Dashboard、transcript 或 `finish` 语义；这些属于 `rpent/planner/codex.py`。

## 4. 导入与依赖边界

### 4.1 标准库

```python
import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterator
```

- `asyncio`：异步初始化锁、异步上下文及异步 API；
- `dataclass(slots=True)`：为 thread/turn handle 提供轻量数据容器，限制动态属性；
- `Iterator` / `AsyncIterator`：同步和异步通知流的类型标注。

### 4.2 审批策略

`ApprovalMode` 只有两个公开值：

| 值 | wire 行为 |
|---|---|
| `auto_review` | `askForApproval=on-request`，reviewer 为 `auto-review` |
| `deny_all` | `askForApproval=never`，不设置 reviewer |

`thread_start()` 的默认值是 `auto_review`；而 resume、fork、turn 的参数可为 `None`，表示不覆盖 thread 继承的配置。

### 4.3 输入类型

`api.py` 重导出：

- `TextInput(text)`；
- `ImageInput(url)`：数据 URL 或 URL 形式图像；
- `LocalImageInput(path)`；
- `SkillInput(name, path)`；
- `MentionInput(name, path)`；
- `Input`、`InputItem`。

字符串输入会先转成 `TextInput`。wire 形状分别为 `text`、`image`、`localImage`、`skill`、`mention`。单项最终也会被包装为列表。

### 4.4 登录模块

同步入口返回 `ChatgptLoginHandle` 或 `DeviceCodeLoginHandle`；异步入口返回对应的 async handle。登录不是简单的单次响应：底层会以 `login_id` 注册专用通知队列，等待 `account/login/completed`。

### 4.5 运行结果模块

`_run.py` 定义 `TurnResult` 和收集器。收集器从流中提取：

- `item/completed`：追加完整 thread item；
- token usage 更新：保存最新 usage；
- `turn/completed`：取得最终 turn 状态。

若 turn 为 `failed`，收集器直接抛 `RuntimeError`；否则最终响应优先取最后一个 `phase=final_answer` 的 agent message，找不到时再退回 phase 未知的最后消息。

### 4.6 沙箱映射

公开 `Sandbox` 值为：

- `read_only`；
- `workspace_write`；
- `full_access`。

thread 生命周期参数使用 mode 枚举；turn override 使用结构化 policy。`full_access` 在 wire 层映射为 danger-full-access，使用时必须明确其安全影响。

### 4.7 generated 类型

`generated.v2_all` 中的类是 app-server 协议的 Pydantic 模型，如 `ThreadStartParams`、`TurnStartParams`、`TurnCompletedNotification`。`api.py` 负责高层参数到这些 wire 类型的装配，但并不定义这些协议结构。

## 5. `CodexConfig`：运行时启动配置

`CodexConfig` 实际定义于 `client.py`，但由 `api.py` 导入，因此通常从 SDK 高层入口使用。字段如下：

| 字段 | 默认值 | 作用 |
|---|---|---|
| `codex_bin` | `None` | 指定 Codex executable；否则使用 SDK 固定依赖提供的 bundled binary |
| `launch_args_override` | `None` | 完全替换默认启动参数，主要用于测试或特殊启动方式 |
| `config_overrides` | `()` | 每项转换为一组 `--config <key=value>` 参数 |
| `cwd` | `None` | `Popen` 启动目录 |
| `env` | `None` | 覆盖继承环境中的键，不是清空整个环境 |
| `client_name` | `codex_python_sdk` | initialize 中的客户端标识 |
| `client_title` | `Codex Python SDK` | initialize 中的显示名 |
| `client_version` | SDK 版本 | initialize 中的版本 |
| `experimental_api` | `True` | initialize capability `experimentalApi` |

默认 binary 解析规则：显式 `codex_bin` 优先，并检查文件存在；否则从 `codex_cli_bin.bundled_codex_path()` 获取固定 runtime。bundled runtime 的目录还会被放到子进程 `PATH` 前部。

### 5.1 配置覆盖不是普通 Python dict

`config_overrides` 是 `tuple[str, ...]`，每个字符串应是 Codex CLI 能解析的 `key=<literal>`。RPent 使用 `json.dumps()` 构造值，例如：

```python
(
    'mcp_servers.rpent.url="http://127.0.0.1:54321/mcp/"',
    'model_provider="rpent_custom"',
)
```

它们最终变成：

```text
--config mcp_servers.rpent.url="..." --config model_provider="..."
```

### 5.2 `config` 参数与 `CodexConfig` 不同

`Codex.thread_start(config=...)` 中的 `config` 是 `ThreadStartParams` 的 JSON object 字段，属于某个 thread 的运行配置；`CodexConfig` 则控制 Python SDK 如何启动整个本地 app-server。二者不要混淆。

## 6. 同步入口 `Codex`

### 6.1 构造函数

逻辑顺序为：

```python
self._client = CodexClient(config=config)
try:
    self._client.start()
    self._init = validate_initialize_metadata(self._client.initialize())
except Exception:
    self._client.close()
    raise
```

关键点：

1. **同步构造立即启动**：`Codex(...)` 不是惰性对象；构造时就启动 runtime 并 initialize；
2. **异常回滚**：start 或 initialize 失败都会 close；
3. **元数据校验**：initialize 响应由 `validate_initialize_metadata()` 验证；
4. **网络请求不等于构造**：构造主要启动本地 runtime 和建立 JSON-RPC 会话，模型请求通常在 turn 开始后发生。

### 6.2 上下文管理

```python
with Codex(config) as codex:
    ...
```

`__enter__()` 只返回 `self`，因为构造阶段已启动；`__exit__()` 调用 `close()`。底层 close 是幂等的：进程引用为空时立即返回。

推荐始终使用 `with`，否则异常路径容易遗留 app-server 子进程和 reader/stderr daemon thread。

### 6.3 `metadata`

返回 initialize 时保存的 `InitializeResponse`，可包含：

- `serverInfo.name/version`；
- `userAgent`；
- `platformFamily`；
- `platformOs`。

同步对象构造成功后一定已初始化，因此该 property 不需要额外状态检查。

### 6.4 关闭流程

底层同步 close：

1. 将 `_proc` 设为 `None`，阻止重复关闭；
2. 关闭 stdin；
3. `terminate()` 并最多等待 2 秒；
4. 失败则 `kill()`；
5. reader 和 stderr thread 分别最多 join 0.5 秒。

这是“尽快回收本地 runtime”的清理，不等价于等待活动 turn 自然完成。希望保留完整终态时，应先 interrupt 并继续消费到 `turn/completed`。

## 7. 账户与登录 API

### 7.1 API key 登录

`login_api_key(api_key)` 发送：

```text
account/login/start
  type = apiKey
  apiKey = <传入值>
```

该方法直接接收密钥。业务代码应避免打印、序列化或写入 transcript。

### 7.2 ChatGPT 浏览器登录

`login_chatgpt()` 启动浏览器式登录并返回 live handle。调用方通过 handle 等待完成、查看状态或取消；不是调用此方法后立即保证登录成功。

### 7.3 Device Code 登录

`login_chatgpt_device_code()` 用于设备码流程，同样返回 live handle。

### 7.4 查询和登出

- `account(refresh_token=False)` 调用 `account/read`；
- `refresh_token=True` 是否触发刷新由 runtime 协议处理；
- `logout()` 调用 `account/logout` 清除当前会话。

## 8. Thread 生命周期 API

Codex 中的 `Thread` 是可跨多个 turn 复用的对话容器。`Thread` Python 对象只保存底层 client 引用和 thread ID，真实历史与状态由 runtime 管理。

### 8.1 `thread_start()`

主要参数：

| 参数 | 含义 |
|---|---|
| `approval_mode` | thread 默认审批策略，默认 `auto_review` |
| `base_instructions` | 基础指令 |
| `developer_instructions` | 开发者指令 |
| `config` | thread 级 JSON 配置 |
| `cwd` | thread 工作目录 |
| `ephemeral` | 是否为临时 thread |
| `model` / `model_provider` | 模型与 provider |
| `personality` | personality override |
| `sandbox` | thread 沙箱模式 |
| `service_name` / `service_tier` | 服务路由/层级 |
| `session_start_source` / `thread_source` | 来源元数据 |

流程为：高级审批和沙箱值先映射到 wire 字段，构造 `ThreadStartParams`，调用 `thread/start`，最后返回 `Thread(client, started.thread.id)`。

### 8.2 `thread_list()`

支持归档过滤、cursor/limit 分页、cwd、provider、搜索词、section、排序键/方向、来源类型和 state DB 限制，直接返回 `ThreadListResponse`，而不是 `Thread` 列表的简化包装。

### 8.3 `thread_resume()`

按 ID 恢复既有 thread。可覆盖审批、指令、cwd、模型、personality、沙箱和 service tier。参数为 `None` 时不主动覆盖已有设置。

### 8.4 `thread_fork()`

从既有 thread 创建新 thread。它返回一个新 ID，可与原 thread 分叉发展。支持 `ephemeral` 和 `thread_source`，但当前签名没有 `personality` 参数。

### 8.5 archive / unarchive

- `thread_archive(id)` 返回 `ThreadArchiveResponse`；
- `thread_unarchive(id)` 返回可继续使用的 `Thread`。

### 8.6 `models()`

调用 `model/list`。`include_hidden=False` 默认排除隐藏模型。返回值是 runtime 报告的模型列表，因此比在 Python 代码中硬编码模型名更可靠。

## 9. `Thread`：同步对话句柄

`Thread` 是 `@dataclass(slots=True)`：

```python
@dataclass(slots=True)
class Thread:
    _client: CodexClient
    id: str
```

它不拥有 client 生命周期；关闭 `Codex` 后，已有 `Thread` 也无法继续发请求。

### 9.1 `turn()`：启动但不自动收集

`turn(input, ...)` 执行：

1. `_normalize_run_input()`：字符串变成 `TextInput`；
2. `_to_wire_input()`：所有输入变成 wire object 列表；
3. 映射可选 approval override；
4. 映射可选 sandbox policy；
5. 构造 `TurnStartParams`；
6. 调用底层 `turn/start`；
7. 返回 `TurnHandle(client, thread_id, turn_id)`。

可覆盖参数包括 cwd、reasoning effort、model、JSON output schema、personality、sandbox、service tier 和 reasoning summary。

底层 `CodexClient.turn_start()` 对同一个 thread ID 使用 thread-start lock，避免并发启动 turn 或 goal 操作发生竞态。请求成功后，它会尽早注册 turn notification queue。

### 9.2 `run()`：启动并完整收集

`Thread.run()` 是便利方法：

```text
turn() -> turn.stream() -> _collect_turn_result() -> stream.close()
```

它适合只关心最终结果的调用方；若需要实时显示事件、执行自定义工具观察、steer 或 interrupt，应使用 `turn()`。

`finally: stream.close()` 很重要：即使结果收集器抛错，也会执行 generator 的 `finally`，注销 turn queue。

### 9.3 `read()`

读取 thread 当前状态。`include_turns=True` 时要求 runtime 一并返回 turn 历史，数据量可能明显增加。

### 9.4 `set_name()` 和 `compact()`

- `set_name(name)` 调用 `thread/name/set`；
- `compact()` 调用 `thread/compact/start`，返回“压缩已开始”的响应，而非保证压缩已经完成。

## 10. `TurnHandle`：控制活动回合

结构为：

```python
@dataclass(slots=True)
class TurnHandle:
    _client: CodexClient
    thread_id: str
    id: str
```

### 10.1 `steer()`

`steer(input)` 将新增输入送入**当前活动 turn**：

```text
turn/steer
  threadId
  expectedTurnId
  input
```

`expectedTurnId` 是并发保护。若该 turn 已结束或 thread 的活动 turn 已变化，runtime 可拒绝请求，防止消息错误注入另一个回合。

Steer 不创建新的 `TurnHandle`，通常也不会增加一个新的 `turn/completed` 结算边界。

### 10.2 `interrupt()`

发送 `turn/interrupt` 请求。它表示“请求中断”，返回 `TurnInterruptResponse`，但调用返回不等价于事件流已经结束。可靠做法是继续消费，直到匹配的 `turn/completed`。

### 10.3 `stream()`

同步 generator 的核心逻辑：

```python
register_turn_notifications(turn_id)
try:
    while True:
        event = next_turn_notification(turn_id)
        yield event
        if event 是匹配 turn_id 的 turn/completed:
            break
finally:
    unregister_turn_notifications(turn_id)
```

三个关键点：

1. **按 turn 定向**：不会把其他 turn 的通知混进当前流；
2. **完成条件严格**：method、payload 类型和 payload 内 turn ID 都要匹配；
3. **必须关闭/消费完**：提前停止迭代时应显式 `stream.close()`，否则 queue 注册可能延迟到 generator 被回收时才释放。

### 10.4 `run()`

对于已经启动的 handle，`turn_handle.run()` 只负责消费和汇总，不再启动第二个 turn。

## 11. Notification 与消息路由

`Notification` 由：

```python
Notification(method: str, payload: NotificationPayload)
```

组成。已知 method 会被解析成 generated Pydantic payload；未知 method 或 payload 校验失败时降级为 `UnknownNotification(params=raw_dict)`，提高前后版本兼容性。

### 11.1 为什么不能多个线程直接读 stdout

stdio 是一个有序 JSON-RPC 消息流，同时包含：

- Python 发出 request 后的 response；
- runtime 主动发出的 notification；
- runtime 发给 Python 的 approval request。

如果每个 turn 或 request 自己读 stdout，消息会被错误消费者抢走。因此 `CodexClient` 只启动一个 reader thread，按消息形状分类：

```text
有 method 且有 id     -> runtime 请求 Python，调用 approval handler 后写 response
有 method 且无 id     -> notification，解析并路由
没有 method           -> Python request 的 response，按 id 唤醒 waiter
```

### 11.2 `MessageRouter` 的队列

路由器维护：

- 每个 request ID 的一次性 response queue；
- 每个 login ID 的 notification queue；
- 每个 turn ID 的 notification queue；
- thread-scoped goal operation route；
- 无法归属到 turn/login 的 global notification queue。

### 11.3 早到事件缓冲

turn 可能在 `turn/start` response 返回前立刻发送 item 通知。路由器会将尚未注册 queue 的 turn 通知放入 pending deque；注册时回放。

但对早到的 `turn/completed`，当前路由逻辑不会无限保留一个已结束 turn 的 pending 队列。底层 `turn_start()` 因此在获得响应后立即注册，尽量缩小竞态窗口。

### 11.4 传输失败唤醒

reader loop 异常时调用 `fail_all(exc)`，把同一个异常放入所有 response、login、turn、goal 和 global queue，避免等待线程永久阻塞。

## 12. `TurnResult` 的汇总规则

字段如下：

| 字段 | 含义 |
|---|---|
| `id` | turn ID |
| `status` | completed/failed/interrupted 等协议状态 |
| `error` | runtime turn error |
| `started_at` / `completed_at` | runtime 时间戳 |
| `duration_ms` | duration |
| `final_response` | 由完整 agent message item 推导出的最终文本 |
| `items` | 所有匹配 turn 的 completed item |
| `usage` | 事件流中最后一次 token usage 快照 |

注意：收集器不靠 delta 拼最终答案，而是读取 `item/completed` 中的完整 agent message。这样可以避免重复 delta 或丢片导致最终文本错误。

若事件流结束但未见匹配的 `turn/completed`，抛出：

```text
RuntimeError: turn completed event not received
```

若 completed turn 的状态是 failed，则也抛 `RuntimeError`，而不是返回带 failed 状态的普通结果。

## 13. 异步入口 `AsyncCodex`

### 13.1 与同步入口最重要的差异

同步 `Codex` 在 `__init__` 中立即启动；异步构造函数不能 `await`，所以 `AsyncCodex` 使用惰性初始化：

```python
self._client = AsyncCodexClient(config=config)
self._init = None
self._initialized = False
self._init_lock = asyncio.Lock()
```

初始化发生在：

- `async with` 的 `__aenter__()`；或
- 第一次 awaited API 调用中的 `_ensure_initialized()`。

### 13.2 双重检查锁

`_ensure_initialized()` 在锁外和锁内各检查一次 `_initialized`：

1. 已初始化时快速返回；
2. 多个协程首次并发调用时只有一个进入初始化；
3. 后续等待者拿到锁后再次检查，避免重复 start/initialize。

失败时会 close client、清空 metadata 和状态，再抛原异常。因此失败后理论上允许后续调用重新尝试初始化。

### 13.3 `metadata` 的前置条件

该 property 不能 await。如果尚未初始化，会抛 `RuntimeError` 并提示使用：

```python
async with AsyncCodex() as codex:
    print(codex.metadata)
```

### 13.4 异步 close 与重用

`close()` await 底层关闭，然后清空初始化状态。此后再次调用 awaited API 会重新启动一个 runtime。虽然实现允许这样做，更清晰的资源模型仍是一个 context 对应一个使用周期。

### 13.5 异步 transport 的真实实现

`AsyncCodexClient` 文档明确写为同步客户端的 async wrapper：

```python
return await asyncio.to_thread(fn, *args, **kwargs)
```

因此：

- app-server 仍由同步 `CodexClient` 拥有；
- stdout reader 仍是同步 reader thread；
- 阻塞的 queue wait 被放到 asyncio 默认线程池；
- 取消等待中的 coroutine 不一定能强行停止底层线程中已经运行的同步函数；
- 清理应通过 `interrupt()`、流关闭和 `AsyncCodex.close()` 协作完成。

## 14. `AsyncThread` 与 `AsyncTurnHandle`

### 14.1 `AsyncThread`

它保存 `_codex: AsyncCodex` 而不是直接保存 async client。这使每个方法都能先调用 `_ensure_initialized()`，并在 runtime 被 close 后按需重新初始化。

方法与同步版一一对应：

- `await run(...)`；
- `await turn(...)`；
- `await read(...)`；
- `await set_name(...)`；
- `await compact()`。

### 14.2 异步流

```python
async for event in turn.stream():
    ...
```

内部每次 `next_turn_notification()` 都通过线程卸载等待同步 queue。`finally` 中同步注销 route，因为它只是一个受锁保护的内存操作，不需要 await。

若手工保存 async generator 并提前退出，应调用：

```python
stream = turn.stream()
try:
    async for event in stream:
        ...
finally:
    await stream.aclose()
```

### 14.3 异步 `run()`

`AsyncTurnHandle.run()` 使用 `_collect_async_turn_result()`，并在 finally 中 `await stream.aclose()`。`AsyncThread.run()` 先启动 turn，再使用同样的收集模式。

## 15. 同步与异步 API 对照

| 同步 | 异步 | 说明 |
|---|---|---|
| `Codex` | `AsyncCodex` | SDK 根入口 |
| `with Codex()` | `async with AsyncCodex()` | 推荐资源范围 |
| 构造时初始化 | context entry/首次 await 初始化 | 生命周期差异 |
| `Thread` | `AsyncThread` | 对话 thread |
| `TurnHandle` | `AsyncTurnHandle` | 活动 turn |
| `for event in turn.stream()` | `async for event in turn.stream()` | 通知流 |
| `turn.steer()` | `await turn.steer()` | 当前 turn 注入 |
| `turn.interrupt()` | `await turn.interrupt()` | 请求中断 |
| `stream.close()` | `await stream.aclose()` | 提前结束消费时清理 |
| `codex.close()` | `await codex.close()` | runtime 回收 |

功能语义基本镜像，但异步版的内部仍使用同步 transport + worker threads，不能把它理解为完全无阻塞线程的纯 asyncio 实现。

## 16. 参数继承与 override 规则

### 16.1 Thread 创建时的默认值

`thread_start()` 中：

- `approval_mode` 有明确默认 `auto_review`；
- 其他多数参数为 `None`，交给 runtime 配置决定；
- sandbox `None` 不主动覆盖。

### 16.2 Resume/Fork/Turn 的可选覆盖

这些方法的 approval mode 为 `None` 时，映射函数返回 `(None, None)`，Pydantic `exclude_none` 后不发送字段。因此不是显式关闭审批，而是继承已有配置。

### 16.3 Pydantic wire 序列化

底层 `_params_dict()` 对 generated models 调用：

```python
model_dump(by_alias=True, exclude_none=True, mode="json")
```

这意味着：

- wire 使用 camelCase alias；
- `None` 字段被省略；
- Enum 等转换为 JSON 兼容值；
- 非 dict 结果会报 `TypeError`。

## 17. 错误模型

底层错误层次包括：

- `CodexError`：SDK 基类；
- `TransportClosedError`：stdio 已关闭或进程退出；
- `JsonRpcError` / `CodexRpcError`；
- `ParseError`：`-32700`；
- `InvalidRequestError`：`-32600`；
- `MethodNotFoundError`：`-32601`；
- `InvalidParamsError`：`-32602`；
- `InternalRpcError`：`-32603`；
- `ServerBusyError`；
- `RetryLimitExceededError`。

### 17.1 transport 错误中的 stderr tail

若 stdout EOF，错误会附带最多约 2000 字符的 stderr tail。底层持续用 daemon thread 排空 stderr，并在最多 400 行的 deque 中保存尾部，既防止 pipe 填满导致子进程死锁，也保留排障信息。

### 17.2 未知通知不是立即错误

未知 method 或 payload 校验失败会降级为 `UnknownNotification`。这是一种协议前向兼容策略：调用方仍可查看 raw params，流也不会因新增事件类型直接崩溃。

### 17.3 默认 approval handler

若没有提供 handler，底层对命令执行和文件修改 approval request 返回 `accept`。但高层 `ApprovalMode.deny_all` 会在 runtime 配置层尽量避免产生这些升级请求。安全策略应在 thread/turn 参数中明确设置，不能只依赖默认 handler。

## 18. 资源所有权与正确清理

所有权链为：

```text
Codex / AsyncCodex
  owns CodexClient / AsyncCodexClient
    owns Codex app-server subprocess
    owns stdout reader thread
    owns stderr drain thread
    owns MessageRouter

Thread / AsyncThread
  borrows root client

TurnHandle / AsyncTurnHandle
  borrows root client
  owns one logical turn notification registration while stream is active
```

推荐清理顺序：

1. 停止提交新输入；
2. 若 turn 活动，调用 `interrupt()`；
3. 继续消费或有限等待 `turn/completed`；
4. 提前退出流时 close/aclose generator；
5. 最后 close 根 `Codex`；
6. 若有外部 MCP server，再关闭 MCP server。

直接 close 根 client 会终止 transport，所有等待者会收到 transport 异常，可能丢失 turn 最后的 item、usage 和 completed 通知。

## 19. 同步使用示例

### 19.1 只取最终结果

```python
import openai_codex

config = openai_codex.CodexConfig(cwd="/path/to/project")
with openai_codex.Codex(config=config) as codex:
    thread = codex.thread_start(
        model="gpt-5.5",
        approval_mode=openai_codex.ApprovalMode.deny_all,
        sandbox=openai_codex.Sandbox.workspace_write,
    )
    result = thread.run("分析当前项目结构")
    print(result.status)
    print(result.final_response)
```

在当前 ChatGPT 登录方式下，RPent 已验证使用 `gpt-5.5`；不要假定账户支持其他模型，应优先用 `codex.models()` 核对 runtime 可见列表。

### 19.2 流式观察并中断

```python
with openai_codex.Codex() as codex:
    thread = codex.thread_start()
    turn = thread.turn("执行一个较长任务")
    stream = turn.stream()
    try:
        for event in stream:
            print(event.method)
            if should_stop(event):
                turn.interrupt()
                # 不立刻 break：继续读取 turn/completed。
    finally:
        stream.close()
```

### 19.3 活动 turn 中 steer

```python
turn = thread.turn("先检查项目")
turn.steer("额外关注依赖冲突")
for event in turn.stream():
    handle(event)
```

实际程序通常在另一个线程读取用户输入，再调用 `steer()`；同时只能有一个消费者读取该 turn stream。

## 20. 异步使用示例

```python
import openai_codex

async def main() -> None:
    config = openai_codex.CodexConfig(cwd="/path/to/project")
    async with openai_codex.AsyncCodex(config=config) as codex:
        print(codex.metadata)
        thread = await codex.thread_start(
            model="gpt-5.5",
            approval_mode=openai_codex.ApprovalMode.deny_all,
        )
        turn = await thread.turn("分析项目")
        async for event in turn.stream():
            print(event.method)
```

需要超时时，应把“取消 Python await”和“中断 runtime turn”区分开：

```python
turn = await thread.turn("长任务")
try:
    async with asyncio.timeout(300):
        async for event in turn.stream():
            handle(event)
except TimeoutError:
    await turn.interrupt()
```

生产代码还应在 interrupt 后有限等待完成边界，并最终关闭 async generator 和 `AsyncCodex`。

## 21. RPent `CodexPlanner` 如何使用该 API

RPent 同时使用同步和异步两套高层 API。

### 21.1 公共 turn options

`CodexPlanner` 构造：

```python
self._turn_options = {
    "approval_mode": openai_codex.ApprovalMode.deny_all,
    "cwd": self._repo_root,
    "model": self._model,
    "sandbox": openai_codex.Sandbox.full_access,
}
```

这些 options 同时传给 `thread_start()` 和每次 `thread.turn()`，保证模型、cwd、审批和 sandbox 一致。`full_access` 是 Codex runtime 的沙箱级别；模型能调用哪些 RPent 机器人技能仍由 Toolkit 注册表和 MCP server 决定。

### 21.2 普通/终端路径：同步 API 放入 worker thread

RPent 的同步路径：

```text
主线程
  -> 启动 HttpMcpServer
  -> 创建 daemon worker
  -> join(timeout)

codex-sdk worker
  -> with Codex(config)
  -> thread_start()
  -> thread.turn()
  -> for event in turn.stream()
```

因为同步 `turn.stream()` 会阻塞等待通知，RPent 不让它占住调用 `solve()` 的主线程。主线程负责 wall-clock timeout；超时时先 `turn.interrupt()`，再 `codex.close()`，并给 worker 15 秒 grace period。

终端 steering 另有一个 `codex-steer` daemon thread：

- 用户普通输入调用 `turn.steer(text)`；
- `/quit` 或 EOF 调用 `turn.interrupt()`；
- 事件流仍只由 `codex-sdk` worker 消费，避免多消费者竞争。

### 21.3 Dashboard 路径：异步 API

Dashboard 使用：

```text
AsyncCodex
  -> 一个 AsyncThread
  -> 任意时刻至多一个 AsyncTurnHandle
  -> 一个 asyncio task 独占消费该 turn.stream()
```

当当前 turn 活动时，新 Dashboard 消息走 `steer()`，返回新增 completion 数 0；idle 时创建新 turn，返回新增 completion 数 1。中断后优先等待 `turn/completed`，15 秒内没有可靠边界才取消本地 consumer task 并手工修正 completion 计数。

这说明 `turn.interrupt()` 的正确语义确实是“请求中断”，而非“同步完成清理”。

### 21.4 Recorder 与 SDK 的边界

`api.py` 只提供原始 typed notifications。RPent `_Recorder` 才负责：

- 把事件写入 `.stream.jsonl`；
- 生成人类可读 transcript；
- 将事件投影给 Dashboard；
- 统计 turn、tool、token；
- 识别 RPent `finish` 工具；
- 保存最后一个完整 agent message。

因此 SDK `TurnResult.final_response` 与 RPent `finish_result` 是不同概念。模型自然语言回答不等价于调用 RPent `finish` 工具。

## 22. RPent HTTP MCP 工具链

`api.py` 不实现 MCP，但 RPent 通过 `CodexConfig.config_overrides` 把本地 MCP endpoint 注入 Codex runtime：

```text
mcp_servers.rpent.url="http://127.0.0.1:<port>/mcp/"
```

完整链路：

```text
模型产生工具调用
  -> Codex runtime
  -> localhost Streamable HTTP MCP
  -> uvicorn 独立线程/事件循环
  -> MCP call_tool handler
  -> asyncio executor
  -> Toolkit.execute_tool(name, arguments)
  -> ToolResult 文本/图像 block
  -> MCP CallToolResult
  -> Codex runtime
  -> turn notification stream
```

### 22.1 为什么使用 HTTP MCP

Codex binary 是独立子进程，而 RPent 的 Toolkit、环境连接、VLA 和 SAM3 client 位于当前 Python 进程。Streamable HTTP MCP 让 binary 调用**同一份 Toolkit 实例**，无需创建第二个机器人环境或再启动一个 Toolkit 子进程。

### 22.2 Ready 检查

`HttpMcpServer.start()` 不是只检查 TCP 端口，而是发送真实 MCP `initialize` JSON-RPC。只有 ASGI lifespan、session manager 和 MCP 分派完整可用后才把 URL 返回给 Codex。

### 22.3 同步 Toolkit 不阻塞 uvicorn loop

MCP handler 用 `run_in_executor()` 执行 `Toolkit.execute_tool()`。机器人 RPC 或模型推理可能长时间阻塞，但不会堵塞 MCP server 的 asyncio event loop。物理环境操作的进一步串行化由 Toolkit 自身负责。

### 22.4 MCP 名称

模型侧名称通常形如：

```text
mcp__rpent__<tool>
```

Toolkit 内部使用短名。桥接层接受前缀全名和短名，并在调用 Toolkit 前移除 namespace。

### 22.5 自定义模型 endpoint

设置 `CODEX_BASE_URL` 时，RPent 注入自定义 provider：

- 自动补 `/v1`；
- `wire_api` 固定为 `responses`；
- API key 从环境变量名读取；
- key 本身不写入 `config_overrides`。

因此自定义 endpoint 必须兼容 Responses wire API。

## 23. 常见误解

### 23.1 `Codex()` 是否立即调用 GPT API

它立即启动本地 runtime 并 initialize，但真正促使模型处理任务的是 `thread.turn()` / `thread.run()` 对应的 turn start。网络访问由 runtime 完成，不是 `api.py` 直接用 HTTP client 发起。

### 23.2 `Thread` 是否等于 Python 线程

不是。这里的 Thread 是 Codex 对话线程；底层同时确实存在 stdout reader thread、stderr drain thread 和 async offload worker thread，但它们是不同概念。

### 23.3 `turn.run()` 与 `thread.run()` 是否相同

- `thread.run(input)`：先创建 turn，再收集；
- `turn_handle.run()`：turn 已经创建，只收集。

### 23.4 `steer()` 是否开启新轮次

不是。它向预期的活动 turn 注入输入。若希望得到新的独立 completion 边界，应等当前 turn 结束后调用 `thread.turn()`。

### 23.5 `interrupt()` 返回是否代表 turn 已结束

不是。仍应等待匹配的 `turn/completed`，或在超时后取消本地消费者并关闭根 client。

### 23.6 `AsyncCodex` 是否完全不使用线程

不是。当前 0.147.0 的 async client 通过 `asyncio.to_thread()` 包装同步 client；同步 client 自身还有 stdout/stderr 线程。

### 23.7 `close()` 后 handle 是否还能用

`Thread` 和 `TurnHandle` 借用根 client。根 client 关闭后，它们继续调用通常会得到 `TransportClosedError`。异步根对象可能在下次 awaited API 时重新初始化，但旧活动 turn 不应被视为可恢复。

## 24. 排障指南

### 24.1 binary 找不到

典型原因：

- SDK 安装缺少固定的 `codex_cli_bin` runtime 包；
- 显式 `CodexConfig.codex_bin` 路径错误；
- 指向的文件不存在。

优先检查当前 Conda 环境中的 package，而不是系统 Python。

### 24.2 initialize 失败

检查：

1. runtime stderr tail；
2. `launch_args_override` 是否破坏了 app-server/stdio 参数；
3. cwd 是否存在；
4. env 是否覆盖了必要变量；
5. SDK 和 bundled runtime 版本是否匹配。

### 24.3 stream 永久等待

检查：

- turn 是否真实启动成功；
- 是否有别的代码错误地消费同一个 stream；
- runtime 是否已经退出；
- 是否提前注销 turn route；
- 中断后是否仍在等待一个从未到达的 completed 边界。

RPent 为 interrupt/close 等待设置 15 秒上限，超时后取消本地 consumer。

### 24.4 turn failed

高层 `run()` 会把 failed turn 转成 `RuntimeError`。若需要完整错误上下文，使用 `turn()` + `stream()`，先持久化原始 notifications，再根据 `turn/completed` payload 判断。

### 24.5 模型 400 或不可用

这通常是账户、登录方式或模型权限问题，不是 Python checkpoint 问题。使用 `codex.models()` 查看 runtime 当前可见模型；当前 RPent 环境的 ChatGPT 登录方式已验证 `gpt-5.5` 可用。

### 24.6 MCP 工具不可见

依次检查：

1. `HttpMcpServer.start()` 的 initialize probe 是否成功；
2. `mcp_servers.rpent.url` override 是否正确；
3. URL 是否带正确 `/mcp/` 路径；
4. `Toolkit.get_tools_spec()` 是否列出工具；
5. Codex binary 是否能访问 loopback；
6. `experimental_api=False` 是否与当前 runtime 的 MCP function-tool 路由需求一致。

### 24.7 工具执行失败但 stream 未异常

RPent 将 `ToolResult.result["error"]` 转成 MCP `CallToolResult(isError=True)`，模型可能收到错误文本并继续恢复，因此 Python stream 不一定抛异常。应同时查看 MCP/Toolkit 日志和 item completed payload。

## 25. API 符号索引

### 25.1 `Codex`

- `metadata`
- `close()`
- `login_api_key()`
- `login_chatgpt()`
- `login_chatgpt_device_code()`
- `account()`
- `logout()`
- `thread_start()`
- `thread_list()`
- `thread_resume()`
- `thread_fork()`
- `thread_archive()`
- `thread_unarchive()`
- `models()`

### 25.2 `AsyncCodex`

与 `Codex` 镜像，除 `metadata` property 外，业务操作均为 async；另有内部 `_ensure_initialized()`。

### 25.3 `Thread` / `AsyncThread`

- `run()`
- `turn()`
- `read()`
- `set_name()`
- `compact()`

### 25.4 `TurnHandle` / `AsyncTurnHandle`

- `steer()`
- `interrupt()`
- `stream()`
- `run()`

## 26. 总结

`openai_codex/api.py` 的核心价值不是实现模型协议，而是建立清晰的对象生命周期：

```text
Codex runtime 生命周期
  -> Thread 对话生命周期
    -> Turn 执行生命周期
      -> Notification 流
        -> TurnResult 汇总
```

理解该文件时最需要记住六点：

1. `Codex` 同步构造立即启动并 initialize，`AsyncCodex` 惰性初始化；
2. 真正 transport 是本地 `codex app-server` 子进程上的 stdio JSON-RPC；
3. 所有 stdout 消息由唯一 reader thread 读取，再按 request/login/turn 分流；
4. `steer` 注入当前 turn，`interrupt` 只请求中断，可靠终点是 `turn/completed`；
5. async API 当前通过线程卸载包装同步 client，不是独立的原生 async transport；
6. RPent 在其上叠加 HTTP MCP、Toolkit、Dashboard 控制、Recorder 与 `finish` 约定，这些不属于 `api.py` 本身。
