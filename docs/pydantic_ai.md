# Pydantic AI 详细介绍与 RPent 集成解析

> 本文以 RPent 的 `rpent` Conda 环境中实际安装的版本为准，而不是其他历史版本的 API。
>
> - `pydantic-ai-slim`: **2.32.0**
> - `pydantic`: **2.13.4**
> - `pydantic-core`: **2.46.4**
> - Python: **3.10**
> - RPent 依赖声明：`pydantic-ai-slim[anthropic,openai]>=2.1`

## 目录

1. [Pydantic AI 是什么](#1-pydantic-ai-是什么)
2. [安装包与模块组成](#2-安装包与模块组成)
3. [总体架构](#3-总体架构)
4. [Agent：应用层入口](#4-agent应用层入口)
5. [Model 与 Provider](#5-model-与-provider)
6. [模型 ID 与 infer_model](#6-模型-id-与-infer_model)
7. [消息模型](#7-消息模型)
8. [工具系统](#8-工具系统)
9. [依赖注入与 RunContext](#9-依赖注入与-runcontext)
10. [结构化输出](#10-结构化输出)
11. [同步、异步、流式与图执行](#11-同步异步流式与图执行)
12. [多模态输入与 ToolReturn](#12-多模态输入与-toolreturn)
13. [Usage、预算和请求限制](#13-usage预算和请求限制)
14. [Capabilities](#14-capabilities)
15. [历史消息和长对话](#15-历史消息和长对话)
16. [错误、重试、超时与取消](#16-错误重试超时与取消)
17. [测试与禁止真实请求](#17-测试与禁止真实请求)
18. [RPent 中的完整集成链路](#18-rpent-中的完整集成链路)
19. [RPent 三种运行模式](#19-rpent-三种运行模式)
20. [RPent 工具和图像桥接](#20-rpent-工具和图像桥接)
21. [常见误区](#21-常见误区)
22. [排障指南](#22-排障指南)
23. [最小示例](#23-最小示例)
24. [源码阅读路线](#24-源码阅读路线)
25. [总结](#25-总结)
## 1. Pydantic AI 是什么

Pydantic AI 是一个用 Python 构建 LLM Agent 的框架。它的核心目标是把不同模型供应商的接口差异隐藏在统一抽象后面，同时使用 Pydantic 的类型验证和 JSON Schema 能力描述输入、工具参数和结构化输出。

它解决的主要问题包括：

- 用统一的 `Agent` API 调用 OpenAI、Anthropic、Google、Bedrock 等不同模型；
- 将普通 Python 函数或现有 JSON Schema 注册为模型工具；
- 自动验证工具参数和最终结构化输出；
- 维护模型消息、工具调用和工具结果之间的协议关系；
- 支持同步、异步、流式事件和底层执行图；
- 统计 token、请求数和缓存使用量，并设置运行预算；
- 通过 `RunContext` 向工具和动态指令注入应用依赖；
- 在统一消息模型中表达文本、思考、图片、文件、工具调用和工具返回；
- 通过 capabilities 在不修改核心 Agent 的情况下加入 thinking、历史处理等能力。

Pydantic AI 不是模型本身，也不会在本地生成 GPT/Claude 的回答。它是位于应用代码和模型供应商 SDK/API 之间的编排层。

```text
应用业务代码
    ↓
Pydantic AI Agent
    ↓
Model 抽象
    ↓
Provider 与供应商 SDK/HTTP client
    ↓
OpenAI / Anthropic / Google / 兼容 API
```

在 RPent 中，它承担的是“高层模型推理循环”，而 LIBERO 环境、Pi0.5、SAM3 和机器人技能仍属于 RPent 自己的运行时。

## 2. 安装包与模块组成

### 2.1 `pydantic-ai` 和 `pydantic-ai-slim`

Pydantic AI 提供完整元包和精简发行版。RPent 安装的是：

```toml
pydantic-ai-slim[anthropic,openai]>=2.1
```

`slim` 版本允许项目只安装所需 provider 的可选依赖，减少不使用供应商 SDK 带来的安装体积和依赖冲突。本环境因此不存在名为 `pydantic-ai` 的 distribution metadata，但 Python 模块仍然是：

```python
import pydantic_ai
```

当前实际 distribution 是：

```text
pydantic-ai-slim 2.32.0
```

### 2.2 重要模块

| 模块 | 主要职责 |
|---|---|
| `pydantic_ai` | 导出 `Agent`、`Tool`、`RunContext`、`BinaryContent` 等常用类型 |
| `pydantic_ai.agent` | Agent 配置和运行入口 |
| `pydantic_ai.models` | `Model` 抽象、模型推断、流式响应和请求参数 |
| `pydantic_ai.providers` | Provider、认证、base URL 和 SDK client 生命周期 |
| `pydantic_ai.messages` | 请求、响应、part、流事件和工具事件 |
| `pydantic_ai.tools` | 工具定义、参数 schema、验证和执行包装 |
| `pydantic_ai.output` | 结构化输出定义和模式 |
| `pydantic_ai.usage` | token、请求、费用和工具调用限制 |
| `pydantic_ai.capabilities` | Thinking、ProcessHistory 等可组合能力 |
| `pydantic_ai.exceptions` | HTTP、usage、验证和用户配置错误 |

## 3. 总体架构

理解 Pydantic AI 时，应区分以下对象：

```text
Agent
 ├── Model
 │    ├── Provider
 │    │    └── SDK / HTTP client / credentials / base_url
 │    └── ModelProfile
 ├── Instructions / System Prompt
 ├── Tools / Toolsets
 ├── Output type
 ├── Model settings
 ├── Capabilities
 └── Run configuration
```

### 3.1 Agent

Agent 是应用层编排入口，负责：

- 接收用户输入；
- 组装指令、历史、工具和输出 schema；
- 调用 Model；
- 识别模型返回的工具调用；
- 执行工具并把结果返回模型；
- 重复模型—工具循环，直到产生最终输出或达到限制。

### 3.2 Model

Model 是供应商无关的模型适配器。具体实现例如：

- `OpenAIResponsesModel`
- `OpenAIChatModel`
- `AnthropicModel`
- `GoogleModel`
- `BedrockConverseModel`

Model 负责把 Pydantic AI 的统一消息和工具定义转换成某家 API 的请求格式，并将供应商响应转换回统一 `ModelResponse` 或流式事件。

### 3.3 Provider

Provider 负责连接供应商：

- API key 和认证；
- base URL；
- SDK client；
- HTTP client 生命周期；
- provider 级模型 profile；
- 供应商特有配置。

因此 Model 和 Provider 不是同一个概念：Model 负责“协议适配”，Provider 负责“连接谁、用什么客户端和凭证”。

### 3.4 Run

一次 `run()` 或 `iter()` 表示一次 Agent 执行。一个 run 内可能包含多次模型请求和多次工具调用：

```text
用户输入
  → 模型请求 1
  → 工具调用 A
  → 工具结果 A
  → 模型请求 2
  → 工具调用 B
  → 工具结果 B
  → 模型请求 3
  → 最终输出
```

所以“run”“模型 request”“模型回答 turn”和“tool call”不能当成同一个计数。
## 4. Agent：应用层入口

当前版本的 `Agent` 构造器包含以下关键参数：

```python
Agent(
    model=None,
    *,
    output_type=str,
    instructions=None,
    system_prompt=(),
    deps_type=object,
    model_settings=None,
    retries=None,
    tools=(),
    toolsets=None,
    end_strategy="graceful",
    tool_timeout=None,
    max_concurrency=None,
    capabilities=None,
)
```

### 4.1 `model`

可以传入：

- 已构造的 `Model` 实例；
- `provider:model` 字符串；
- 已知模型名称类型；
- `None`，稍后在 `run()` 时提供。

RPent 不让 Agent 自己解析字符串，而是先调用 `infer_model()`，再把 Model 实例传入 Agent。这允许 RPent 在 Provider 创建阶段注入自定义 `base_url`。

### 4.2 `instructions` 与 `system_prompt`

两者都能影响模型行为，但用途不同：

- `instructions` 是当前推荐的 Agent 指令机制，可支持静态字符串、动态函数和依赖上下文；
- `system_prompt` 是系统提示兼容入口，可包含一个或多个字符串。

RPent 将环境生成的 `system_prompt` 传给 Agent 的 `instructions`：

```python
Agent(
    model,
    instructions=system_prompt or None,
)
```

### 4.3 `output_type`

`output_type` 声明最终输出的 Python 类型。默认是 `str`，也可以是 Pydantic model、dataclass、TypedDict、联合类型或其他受支持 schema。

### 4.4 `tools` 与 `toolsets`

- `tools` 适合直接注册函数或 `Tool` 对象；
- `toolsets` 适合动态、分组或外部来源的工具集合；
- MCP、动态 capability 等场景通常更接近 toolset 的概念。

RPent 使用 `tools`，因为它已经拥有自己的 Toolkit 注册表，并将每个 schema 转成独立 `Tool`。

### 4.5 `retries`

用于输出验证或工具参数验证失败后的模型重试。它不是底层网络连接重试的同义词。网络重试通常由 Provider SDK、HTTP transport 或应用层策略控制。

### 4.6 `end_strategy`

决定模型在同一响应里既返回最终结果又包含其他工具调用时如何结束。`graceful` 通常允许框架妥善处理当前响应，而不是粗暴丢弃剩余协议内容。

## 5. Model 与 Provider

### 5.1 Model 的统一接口

`Model` 基类最重要的方法是：

```python
async def request(messages, model_settings, model_request_parameters)

@asynccontextmanager
async def request_stream(
    messages,
    model_settings,
    model_request_parameters,
    run_context=None,
)
```

具体 Model 子类完成供应商格式映射。可选能力还包括：

- `count_tokens()`：请求前 token 估算；
- `compact_messages()`：供应商原生上下文压缩；
- `cancel_suspended_response()`：取消后台响应；
- `continuation_delay()`：暂停响应继续执行前的轮询间隔；
- `prepare_request()`：合并设置和解析工具/输出模式；
- `prepare_messages()`：规范化跨供应商历史消息。

### 5.2 ModelProfile

`ModelProfile` 描述模型能力，例如：

- 是否支持 function tools；
- 是否支持图片输出；
- 是否支持 JSON Schema 结构化输出；
- 是否支持 thinking；
- 支持哪些原生工具；
- 使用什么 JSON Schema transformer；
- 默认结构化输出模式；
- 是否支持行内 system prompt；
- 工具延迟加载和动态增加方式。

Profile 的合并优先级大致是：

```text
DEFAULT_PROFILE
  < Provider 对该模型的默认 profile
  < 用户传入的 profile override
```

最终还会与具体 Model 类真正实现的能力取交集，避免 profile 声称支持但适配器不会渲染。

### 5.3 ModelSettings

`ModelSettings` 是一次请求的参数集合，常见字段包括：

- `max_tokens`
- temperature
- top-p
- timeout
- provider-specific settings
- thinking 配置

模型构造时的默认 settings 会与每次 run/request 的 settings 合并，后者覆盖前者。

RPent 为普通模型设置：

```python
ModelSettings(max_tokens=max_tokens)
```

对于 Anthropic 则使用 `AnthropicModelSettings` 额外启用指令、工具定义和消息缓存。

### 5.4 Provider factory

RPent 的 Provider 创建逻辑如下：

```python
def _provider_factory(provider_name: str):
    if not base_url:
        return infer_provider(provider_name)

    provider_cls = infer_provider_class(provider_name)
    params = inspect.signature(provider_cls.__init__).parameters
    kwargs = {}
    if "base_url" in params:
        kwargs["base_url"] = base_url
    return provider_cls(**kwargs)
```

这段代码的意义是：

1. 没有自定义地址时，按官方规则创建 Provider；
2. 有自定义地址时，找到对应 Provider 类；
3. 只有构造器支持 `base_url` 才注入；
4. API key 仍由 Provider 按自己的环境变量规则读取。

它允许 `openai-chat:` 等模型连接兼容服务，但兼容服务必须实现相应 wire protocol。
## 6. 模型 ID 与 `infer_model`

### 6.1 模型 ID 格式

当前版本使用明确的：

```text
provider:model_name
```

例如：

```text
openai:gpt-5.5
openai-chat:某个 Chat Completions 兼容模型
anthropic:claude-opus-4-8
```

`parse_model_id()` 只在第一个冒号处分割：

```python
provider_name, model_name = parse_model_id("openai:gpt-5.5")
# ("openai", "gpt-5.5")
```

无前缀时返回：

```python
(None, original_string)
```

除特殊测试模型外，`infer_model()` 要求明确 provider，不再仅根据模型名字猜供应商。

### 6.2 `infer_model()` 做什么

```python
api_model = infer_model(
    model,
    provider_factory=_provider_factory,
)
```

执行步骤是：

1. 如果参数已经是 `Model` 实例，原样返回；
2. 字符串 `test` 创建 `TestModel`；
3. 解析 provider 和 model name；
4. 创建 Provider；
5. 根据 provider 选择具体 Model 子类；
6. 返回 Model 对象。

它**不会立即发起模型请求**。真正的网络访问发生在 Agent 执行图推进到模型请求节点，进而调用：

```python
model.request(...)
# 或
model.request_stream(...)
```

### 6.3 `openai:` 与 `openai-chat:`

在当前版本中：

- `openai:`、`openai-responses:` 等路由到 `OpenAIResponsesModel`；
- `openai-chat:` 及一组 Chat-compatible provider 路由到 `OpenAIChatModel`。

因此二者不是名字别名，而是不同的线协议选择。兼容端点只支持 Chat Completions 时，应使用 Chat 路径；要求 Responses API 的功能则必须由服务端真正实现 Responses wire API。

### 6.4 已知模型名称不是账号权限

`known_model_names()` 返回当前库知道如何命名或描述的模型 ID。当前环境中该列表有数百项，但这不表示当前 API key、ChatGPT 登录方式或兼容服务有权使用全部模型。

模型是否可调用最终取决于：

- 供应商账户权限；
- API key；
- endpoint 支持；
- 模型是否实际部署；
- 请求所用协议；
- 供应商区域和组织配置。

## 7. 消息模型

Pydantic AI 不使用单一的 `{"role": ..., "content": ...}` 字典贯穿内部，而是使用类型化消息和 part。

### 7.1 顶层消息

常见顶层类型：

- `ModelRequest`：发往模型的请求侧消息；
- `ModelResponse`：模型返回的响应侧消息。

### 7.2 请求侧 part

常见请求 part：

- `SystemPromptPart`
- `UserPromptPart`
- `ToolReturnPart`
- `RetryPromptPart`
- `InstructionPart`
- `ToolAvailabilityDeltaPart`

### 7.3 响应侧 part

常见响应 part：

- `TextPart`
- `ThinkingPart`
- `ToolCallPart`
- `FilePart`
- `CompactionPart`

### 7.4 为什么要类型化

类型化消息可以：

- 保证工具调用 ID 与工具结果正确配对；
- 区分最终文本和 reasoning；
- 表达二进制图片、文件及供应商元数据；
- 在不同 Provider 之间转换历史；
- 对流式 delta 进行增量聚合；
- 保留 usage、finish reason、response ID 和状态。

### 7.5 历史必须保持工具协议完整

典型工具历史应是：

```text
ModelResponse(ToolCallPart)
ModelRequest(ToolReturnPart)
```

如果只保存工具调用却缺少结果，许多 Provider 会返回 400。RPent Dashboard 在中断时专门修复历史前沿：能取得对应 request 时追加 request；旧 SDK 无法取得时移除最后一条孤立工具调用响应。

## 8. 工具系统

### 8.1 从 Python 函数生成工具

最简单的工具是普通函数：

```python
from pydantic_ai import Agent, RunContext

agent = Agent("test")

@agent.tool_plain
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b
```

框架可以从类型注解和 docstring 生成参数 JSON Schema。

### 8.2 带 `RunContext` 的工具

```python
@agent.tool
def lookup(ctx: RunContext[MyDeps], key: str) -> str:
    return ctx.deps.store[key]
```

带上下文的工具可访问依赖、usage、消息、当前工具名和调用 ID 等运行信息。

### 8.3 从已有 JSON Schema 创建工具

如果应用本身已经维护 schema，可使用：

```python
Tool.from_schema(
    function=handler,
    name="move_to",
    description="Move end effector to a world XYZ target.",
    json_schema={
        "type": "object",
        "properties": {
            "xyz": {
                "type": "array",
                "items": {"type": "number"},
            }
        },
        "required": ["xyz"],
    },
    takes_ctx=False,
)
```

RPent 使用的正是该路径，因为 `Toolkit.get_tools_spec()` 已经提供标准 JSON Schema。

### 8.4 工具返回值

普通工具可返回字符串、字典或可序列化对象。需要同时返回文本和多模态内容时，可使用：

```python
ToolReturn(
    return_value={"status": "ok"},
    content=[BinaryContent(data=image_bytes, media_type="image/png")],
)
```

其中：

- `return_value` 是工具的逻辑结果；
- `content` 是提供给模型的附加用户内容；
- `metadata` 可保存应用元数据；
- `tools` 可用于动态工具相关场景。

### 8.5 工具错误与重试

工具参数由 schema/Pydantic 验证。应用还可以将业务失败作为普通结构化结果返回，让模型观察错误并修正下一次调用。RPent 的 Toolkit 采用这种模式：未知工具、参数错误、取消和环境异常大多转成 `ToolResult`，而不是直接终止 Agent。
## 9. 依赖注入与 `RunContext`

Pydantic AI 的依赖注入不是全局服务容器，而是每次 run 显式传入 `deps`，工具或动态指令通过 `RunContext` 读取。

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class AppDeps:
    database: object
    tenant_id: str

agent = Agent(
    "test",
    deps_type=AppDeps,
)

@agent.tool
async def get_record(ctx: RunContext[AppDeps], record_id: str) -> dict:
    return await ctx.deps.database.get(ctx.deps.tenant_id, record_id)
```

运行时：

```python
result = await agent.run(
    "Find record A-100",
    deps=AppDeps(database=db, tenant_id="tenant-example"),
)
```

`RunContext` 当前还可携带：

- 当前 Model；
- `RunUsage` 和 `UsageLimits`；
- prompt 和消息历史；
- retry 计数；
- 当前工具名和 tool call ID；
- run ID 与 conversation ID；
- model settings；
- pending messages；
- capability 状态。

RPent 没有把 `LiberoToolkit` 放入 `deps`，而是通过闭包把 Toolkit 捕获到每个 Tool handler 中。这是合理的另一种设计：

```python
def _make_tool_function(toolkit, name):
    def _call(**kwargs):
        return toolkit.execute_tool(name, kwargs)
    return _call
```

## 10. 结构化输出

结构化输出是 Pydantic AI 相比直接调用模型 SDK 的核心优势之一。

### 10.1 Pydantic model 输出

```python
from pydantic import BaseModel
from pydantic_ai import Agent

class Classification(BaseModel):
    label: str
    confidence: float

agent = Agent(
    "test",
    output_type=Classification,
)
```

框架负责：

1. 为 `Classification` 生成 JSON Schema；
2. 根据 ModelProfile 选择输出方式；
3. 验证模型结果；
4. 失败时按重试设置要求模型修正；
5. 返回类型化对象。

### 10.2 输出模式

当前模型层支持的主要模式包括：

- `tool`：通过特殊 output tool 返回对象；
- `native`：使用供应商原生 JSON Schema/response format；
- `prompted`：把 schema 说明写入 prompt；
- `text`：普通文本；
- `auto`：根据 profile 选择默认结构化模式。

不是所有模型都支持原生 JSON Schema。若强制使用不支持的模式，框架会抛出配置错误，而不是静默假装成功。

### 10.3 RPent 为什么没有使用结构化最终输出

RPent 的 API Planner 将 Agent 默认输出保留为字符串，并使用 Toolkit 的 `finish` 工具建立跨后端统一协议：

```json
{
  "status": "success",
  "summary": "Task completed"
}
```

Observer 识别工具名后生成：

```python
{"_finish": True, "status": "success", "summary": "Task completed"}
```

这样 Pydantic AI、Claude SDK 和 Codex SDK 三个 Planner 都能返回一致的 `PlannerResult.finish_result`。

## 11. 同步、异步、流式与图执行

### 11.1 `run_sync()`

同步入口，适合没有运行中 event loop 的脚本：

```python
result = agent.run_sync("Hello")
print(result.output)
```

在已有 asyncio event loop 中不应调用会自行运行事件循环的同步封装。

### 11.2 `run()`

异步完成式运行：

```python
result = await agent.run("Hello")
```

它执行完整模型—工具循环后返回 `AgentRunResult`。

### 11.3 `run_stream()`

面向最终响应的流式消费，适合实时显示文本。需要注意，流式最终输出和底层所有 Agent 事件不是同一抽象层。

### 11.4 `iter()`

`iter()` 暴露底层 Agent 执行图：

```python
async with agent.iter("Do the task") as run:
    async for node in run:
        ...
```

它允许应用观察或手动推进节点，适合：

- 自定义工具事件日志；
- 在工具边界接受外部指令；
- 实现 Dashboard 中断；
- 保存历史 checkpoint；
- 在特定节点插入控制逻辑。

RPent 使用 `iter()`，而不是简单 `run()`，因为它必须实时记录工具调用、控制机器人动作、接收用户 steering，并把 usage/transcript 推送到 Dashboard。

### 11.5 Node 和流事件

RPent 重点处理：

- CallToolsNode：模型响应包含工具调用；
- EndNode：本次 run 正常结束；
- `FunctionToolCallEvent`：一个工具调用开始；
- `FunctionToolResultEvent`：工具调用完成并形成结果。

执行方式：

```python
if Agent.is_call_tools_node(node):
    observe(node.model_response)

    async with node.stream(run.ctx) as stream:
        async for event in stream:
            observe_tool_event(event)
```

这使工具调用和结果能够边执行边投影到日志与 Dashboard。
## 12. 多模态输入与 `ToolReturn`

### 12.1 `BinaryContent`

当前版本通过 `BinaryContent` 表达内存中的二进制输入：

```python
BinaryContent(
    data=image_bytes,
    media_type="image/png",
)
```

它可以用于图像，也可表达受支持的音频或文档媒体类型。是否真正可发送取决于目标 Model/Profile 和 Provider。

### 12.2 URL 与上传文件

消息层还支持文件 URL、上传文件和视频 URL 等类型。库在需要下载 URL 时包含 SSRF 防护：

- 只允许 HTTP/HTTPS；
- 默认阻止私有和内部地址；
- 阻止云元数据 endpoint；
- 防止 DNS rebinding；
- 限制响应体大小。

不要因为框架提供 URL 类型，就假定任意内网 URL 都能被安全读取。

### 12.3 RPent 的图像路径

RPent 的工具结果内部采用文本/图片 content blocks。API Planner 转换为：

```text
Toolkit ToolResult
  → text block
  → base64 image block
  → decode 为 bytes
  → BinaryContent
  → ToolReturn
  → Pydantic AI 消息历史
```

此外，RPent 注册 `read_image` 工具，让模型按 artifact 名和 step 读取已经由 `EnvState` 登记的图像。它不是任意文件读取器：

- artifact 必须属于该 step；
- 文件必须存在；
- 扩展名必须是 PNG/JPEG；
- 路径由 EnvState 管理；
- `--no-images` 模式只返回文字说明，不读取图片 bytes。

### 12.4 Provider 不支持视觉时

如果兼容 endpoint 返回 4xx 且错误中明确出现 image，RPent 会提示使用：

```text
--no-images
```

此时模型仍可使用：

- `view_env_state` 的文本状态；
- `back_project` 的世界坐标；
- 数值工具结果；
- artifact 存在性信息。

## 13. Usage、预算和请求限制

### 13.1 `RunUsage`

`RunUsage` 用于累计一次或多次 run 的消耗。常用字段包括：

- `input_tokens`
- `output_tokens`
- `cache_read_tokens`
- `cache_write_tokens`
- `requests`
- provider-specific details

是否能获得某项统计取决于 Provider 响应。

### 13.2 `UsageLimits`

当前版本支持：

```python
UsageLimits(
    cost_limit=None,
    request_limit=50,
    tool_calls_limit=None,
    input_tokens_limit=None,
    output_tokens_limit=None,
    total_tokens_limit=None,
    per_request_input_tokens_limit=None,
    count_tokens_before_request=False,
)
```

限制的意义不同：

- `request_limit`：模型请求次数；
- `tool_calls_limit`：工具调用次数；
- token limits：输入、输出或总 token；
- `cost_limit`：费用上限；
- `count_tokens_before_request`：请求前预估，需要 Model 支持 token counting。

超过限制通常抛 `UsageLimitExceeded`。

### 13.3 RPent 的预算

RPent 使用：

```python
UsageLimits(request_limit=max_turns + 1)
```

同时自己维护：

- `observer.turns`
- `observer.tool_calls`
- `usage.requests`

它们不是同一个量：

| 指标 | 含义 |
|---|---|
| `turns` | RPent 观察到的带工具调用模型响应数 |
| `tool_calls` | 实际 FunctionToolCallEvent 数量 |
| `usage.requests` | Provider 模型请求次数 |

一个模型响应可以包含多个工具调用，最终无工具文本也可能产生一次 request，却不增加 RPent 的工具 turn。

## 14. Capabilities

Capabilities 是可组合的 Agent 行为扩展。它们可以参与模型选择、请求包装、事件流、工具可见性和历史处理。

### 14.1 `Thinking`

RPent 配置：

```python
Thinking(effort="high")
```

这表示请求较高 reasoning effort。它不保证每家 Provider 都支持，也不保证一定产生公开的 `ThinkingPart`。实际行为由 ModelProfile 和供应商协议决定。

### 14.2 `ProcessHistory`

RPent 配置：

```python
ProcessHistory(processor=_prune_history_images)
```

该 processor 在历史重发前运行。RPent 用它限制图片历史：

- 总预算为 4 MiB decoded bytes；
- 从最新向最旧保留；
- 至少保留最近 2 张；
- 被裁剪图片替换成文字占位符；
- 使用 `dataclasses.replace()`，不原地修改源消息。

它只处理即将发送给模型的历史副本，不删除磁盘上的 EnvState artifact，也不修改 RPent transcript。

### 14.3 延迟 capability 与动态工具

当前模型层支持工具延迟加载和中途增加。工具可能处于：

- `visible`
- `deferred`
- `withheld`
- `via_history`

具体表示取决于 Model 是否支持 schema deferral、tool search 或动态 addition。没有原生机制时，框架可能通过系统公告或合成工具搜索交换来保留语义。

RPent 当前主要使用 `Thinking` 和 `ProcessHistory`，并未直接依赖复杂的 capability 动态加载流程。
## 15. 历史消息和长对话

### 15.1 `message_history`

新 run 可以接收上一 run 的完整历史：

```python
result = await agent.run(
    "Continue",
    message_history=previous_messages,
)
```

在 `iter()` 路径中，可以通过：

```python
history = run.all_messages()
```

获得包含用户输入、模型回答、工具调用和工具结果的统一消息历史。

### 15.2 历史和 transcript 不是一回事

RPent 同时维护两种记录：

1. Pydantic AI `ModelMessage` 历史：用于下一次模型请求，保留协议类型和 BinaryContent；
2. RPent JSON-safe transcript：用于日志、Dashboard 和最终产物，不重复保存图片 bytes。

只保存 transcript 字典通常不能无损恢复 Pydantic AI 的工具协议历史。

### 15.3 Compaction

部分 Provider 支持原生消息压缩。模型层使用 `CompactionPart` 表达压缩边界，并根据适配器声明决定：

- 是否要求加密压缩内容；
- 边界前历史是否可裁剪；
- standing system prompt 是否已由压缩结果保留；
- 是否需要重新插入指令。

该机制比简单删除前 N 条消息更复杂，因为工具调用/结果、系统提示和 Provider 缓存前缀必须保持有效。

### 15.4 Prompt cache

Prompt cache 和消息压缩不同：

- cache 旨在降低重复前缀的费用或延迟；
- compaction 旨在缩短上下文；
- history image pruning 旨在限制多模态请求体。

RPent 对 Anthropic 启用：

```python
anthropic_cache_instructions=True
anthropic_cache_tool_definitions=True
anthropic_cache_messages=True
```

其他 Provider 使用其默认策略。

## 16. 错误、重试、超时与取消

### 16.1 常见异常类型

- `UserError`：模型名、能力或配置不合法；
- `ModelHTTPError`：Provider 返回 HTTP 错误；
- `UsageLimitExceeded`：达到 request/token/cost/tool 限制；
- 参数或输出验证错误：工具输入或结构化输出不符合 schema；
- `asyncio.TimeoutError`：应用层 `wait_for` 超时；
- `CancelledError`：异步 run 或 stream 被取消。

### 16.2 重试层次

应区分：

1. 结构化输出/工具参数验证重试；
2. Provider SDK 对临时网络错误的重试；
3. 应用级任务重试；
4. 机器人工具自己的恢复策略。

把所有失败都配置成无限重试会导致重复费用或重复执行有副作用的工具。

### 16.3 超时

Pydantic AI 的 model settings 或 Provider 可具有请求超时，RPent 还在 Planner 外层增加整体超时：

```python
await asyncio.wait_for(session.run(...), timeout=planner_timeout)
```

整体超时取消的是 asyncio 任务，但同步环境工具可能仍在 worker thread 中运行，所以 RPent 还调用：

```python
toolkit.cancel_active_and_wait()
```

让机器人动作在项目定义的安全边界停止。

### 16.4 流式取消

`StreamedResponse.cancel()` 会先标记取消，再要求具体适配器关闭 stream。取消导致的预期 transport error 会被识别，最终状态可标为 `interrupted`；提前停止迭代但未正常耗尽的结果应是 `incomplete`，不能误标为 `complete`。

### 16.5 远端请求不一定真的停止

本地取消无法保证远端 Provider 尚未完成计算。是否支持服务端 job cancellation 取决于 Model 实现。涉及有副作用的工具时，应用必须使用自己的幂等、串行化和安全取消机制。

RPent 的 Toolkit 保证同一环境只有一个 active operation，并提供 `cancel_active_and_wait()`。

## 17. 测试与禁止真实请求

Pydantic AI 提供测试模型，例如 `TestModel` 和 `FunctionModel`。它们适合：

- 验证工具 schema；
- 测试 Agent 控制流；
- 模拟模型响应；
- 避免真实 API 费用。

模型模块还提供全局请求开关：

```python
from pydantic_ai.models import override_allow_model_requests

with override_allow_model_requests(False):
    ...
```

真实付费/远程 Model 应在发送请求前检查该开关。测试模型和纯本地计算不受影响。

注意：这是进程级全局变量，不是并发任务隔离的 context variable；并发测试应避免互相覆盖。
## 18. RPent 中的完整集成链路

RPent 的 API Planner 调用链如下：

```text
CLI / Dashboard 参数
  │
  ├─ planner_type = "api"
  ├─ model = "provider:model"
  ├─ base_url（可选）
  ├─ max_tokens
  └─ planner_timeout_s
  ↓
rpent.planner.base.build_planner()
  ↓
_provider_factory(provider_name)
  ↓
pydantic_ai.models.infer_model()
  ↓
具体 Provider + Model 实例
  ↓
ApiAgentLoop(model=...)
  ↓
ApiAgentLoop.solve(..., toolkit=...)
  ↓
_build_agent()
  ├─ Model
  ├─ instructions
  ├─ Toolkit → Pydantic AI Tools
  ├─ ModelSettings
  ├─ Thinking
  └─ ProcessHistory
  ↓
Agent.iter()
  ↓
模型请求节点
  ↓
CallToolsNode
  ↓
FunctionToolCallEvent
  ↓
Toolkit.execute_tool()
  ↓
ToolResult → ToolReturn / BinaryContent
  ↓
FunctionToolResultEvent
  ↓
下一次模型请求
  ↓
finish / EndNode / error / budget
  ↓
PlannerResult
```

### 18.1 Planner 工厂

`rpent/planner/base.py` 要求 API 后端显式提供模型 ID：

```text
anthropic:claude-opus-4-8
openai:gpt-5.5
openai-chat:glm-5.2
```

这避免 Pydantic AI 无法判断 Provider 和 wire protocol。

### 18.2 `infer_model()` 阶段

工厂创建 Model 对象，但不发网络请求：

```python
api_model = infer_model(
    model,
    provider_factory=_provider_factory,
)
```

### 18.3 Agent 构造阶段

`ApiAgentLoop._build_agent()` 创建：

```python
Agent(
    self._model,
    instructions=system_prompt or None,
    tools=_build_tools(toolkit, no_images=self._no_images),
    model_settings=_build_model_settings(...),
    capabilities=[
        Thinking(effort="high"),
        ProcessHistory(processor=_prune_history_images),
    ],
)
```

### 18.4 真正请求阶段

真正访问 Provider 是在：

```python
async with agent.iter(...) as run:
    async for node in run:
        ...
```

推进图节点时发生，而不是 `infer_model()` 或 `Agent(...)` 构造时发生。

### 18.5 输出统一

Pydantic AI 的消息、usage 和工具事件最终转换为 RPent 的：

```python
PlannerResult(
    finish_result=...,
    messages=...,
    stats=...,
    error=...,
)
```

因此上层 CLI 不需要理解 Pydantic AI 内部类型。

## 19. RPent 三种运行模式

### 19.1 非交互模式

- 单个 Agent run；
- 没有终端队列；
- 外层 `asyncio.wait_for`；
- EndNode、finish、预算或错误后返回。

### 19.2 终端交互模式

- 使用线程安全 `queue.Queue`；
- 当前 run 中的输入通过 `run.enqueue(..., priority="asap")` 加入下一请求边界；
- run 结束后用 `asyncio.to_thread(queue.get)` 等待下一条消息；
- 下一消息作为新 seed，并复用 `run.all_messages()`；
- 等待用户输入的时间不计入普通 Planner 总超时。

### 19.3 Dashboard 模式

- 每条浏览器消息对应独立 Agent run；
- `_ApiDashboardSession` 维护完整历史 checkpoint；
- 单个 asyncio task 串行处理 pending prompts；
- 消息真正开始时才 deferred ACK；
- 工具完成后才允许切换到新 prompt；
- Esc 会取消 active run 和尚未执行的 pending prompt；
- 同一个 `RunUsage` 跨独立 run 累积。

这种设计避免用户消息在机器人动作中间粗暴切断工具 call/result 协议。
## 20. RPent 工具和图像桥接

### 20.1 Toolkit schema 转 Tool

RPent Toolkit 已经维护：

```python
{
    "name": "move_to",
    "description": "...",
    "input_schema": {...},
}
```

API Planner 逐项转换：

```python
Tool.from_schema(
    function=_make_tool_function(toolkit, name),
    name=name,
    description=spec.get("description", ""),
    json_schema=spec.get("input_schema") or {
        "type": "object",
        "properties": {},
    },
    takes_ctx=False,
)
```

这让同一 Toolkit 可以同时服务：

- Pydantic AI 进程内工具；
- Codex HTTP MCP；
- 其他 Planner 后端。

### 20.2 调用转换

```text
模型 JSON arguments
  ↓
Pydantic AI Tool callable
  ↓
toolkit.execute_tool(name, kwargs)
  ↓
RPent ToolResult
  ↓
content_blocks
  ├─ text
  └─ image/base64
  ↓
API Planner 转换
  ├─ str
  └─ BinaryContent
  ↓
ToolReturn
```

### 20.3 Toolkit 的额外保护

Pydantic AI 负责模型工具协议，但 RPent Toolkit 负责环境安全：

- 同一物理环境串行执行工具；
- 未知工具返回结构化错误；
- 参数 `TypeError` 转成可观察结果；
- 环境异常保存 traceback；
- 非只读工具执行后捕获新状态；
- Dashboard 取消通过 active operation event 协调；
- 工具结果可带相机图片。

### 20.4 `finish` 不是框架内建

Pydantic AI 不知道 RPent 任务何时“物理完成”。RPent 使用普通工具：

```text
finish(status, summary)
```

Observer 在 `FunctionToolCallEvent` 中识别它，设置 `_finish` sentinel。工具结果仍正常进入历史，避免破坏调用协议。

## 21. 常见误区

### 误区 1：`infer_model()` 会立即调用 GPT

不会。它只构造 Provider/Model。实际请求发生在 Agent run 推进时。

### 误区 2：Model 就是 Provider

不是。Model 负责协议和能力适配；Provider 负责连接、凭证、base URL 和 client。

### 误区 3：`openai:` 和 `openai-chat:` 完全等价

不是。前者在当前版本走 Responses 模型，后者走 Chat 模型。兼容端点需要匹配对应协议。

### 误区 4：已知模型列表表示账号都可调用

不是。库知道名称不等于账户有权限，也不等于 endpoint 部署了该模型。

### 误区 5：一次 Agent run 只发一次 HTTP 请求

不一定。每次工具结果返回后通常还需要下一次模型请求。

### 误区 6：turn、request 和 tool call 数相等

不相等。一轮模型响应可以调用多个工具；最终文本也可能产生请求但没有工具调用。

### 误区 7：取消 asyncio task 就一定停止机器人动作

不一定。同步工具可能在线程或远端服务中运行，需要应用自己的安全取消协议。

### 误区 8：把图片从 transcript 删除就不会再发给模型

不一定。模型使用的是 Pydantic AI message history，不是 RPent JSON transcript。RPent 使用 `ProcessHistory` 专门裁剪历史图片。

### 误区 9：结构化输出只是 `json.loads`

不是。它包括 schema 生成、供应商输出模式选择、验证、重试和类型化结果。

### 误区 10：Tool 参数验证可以代替业务安全检查

不能。JSON Schema 只能验证形状和基本约束；机械臂工作空间、最大位移、幂等性和取消边界仍需 Toolkit/环境实现。

## 22. 排障指南

### 22.1 `Unknown model`

检查：

- 是否使用 `provider:model`；
- provider 名是否正确；
- 当前版本是否安装对应 optional dependency；
- 是否把 Responses 模型误写成 Chat provider，反之亦然。

### 22.2 401/403

通常是：

- API key 缺失或错误；
- 组织/项目无权限；
- endpoint 与 key 不匹配；
- 自定义 Provider 没按预期读取环境变量。

### 22.3 404 或 model not found

可能是：

- 模型名错误；
- endpoint 没有该部署；
- Azure/网关使用部署名而不是公开模型名；
- 账号区域不支持；
- 库已知名称与服务端实际可用模型不同。

### 22.4 400：图片不支持

使用文本模型却发送了 `BinaryContent`。RPent 可重试：

```text
--no-images
```

### 22.5 400：工具调用缺少结果

多发生于取消或手工拼接历史。检查最后一个 `ToolCallPart` 是否有匹配的 `ToolReturnPart` 和 tool call ID。

### 22.6 工具 schema 被 Provider 拒绝

不同供应商支持的 JSON Schema 子集不同。Pydantic AI 会通过 ModelProfile 的 transformer 进行适配，但复杂 schema 仍可能需要简化：

- 避免不受支持的组合关键字；
- 检查 strict 模式；
- 确保 required/properties 一致；
- 检查兼容 endpoint 是否真的实现 tools。

### 22.7 UsageLimitExceeded

检查 `UsageLimits`：

- request limit 是否过小；
- 工具循环是否没有业务终止条件；
- 模型是否重复修正失败工具参数；
- 是否把每个用户消息都开成新 run，却错误复用独立预算。

### 22.8 对话越来越慢或请求越来越大

检查：

- 是否持续重发所有图片；
- tool result 是否包含大段 base64；
- 是否启用 prompt cache；
- 是否需要历史 processor 或 compaction；
- transcript 与模型历史是否重复嵌套。

### 22.9 Dashboard 中断后下一次请求 400

检查历史前沿是否存在孤立 `ToolCallPart`。RPent `_ApiDashboardSession._run_agent()` 的 finally 专门处理该问题。
## 23. 最小示例

以下示例用于说明 API 形状。使用真实 Provider 时，需要安装对应 optional dependency 并配置凭证。

### 23.1 最简单文本 Agent

```python
from pydantic_ai import Agent

agent = Agent("test")
result = agent.run_sync("用一句话介绍机器人操作规划")
print(result.output)
```

`test` 使用测试模型，不产生真实 API 费用。

### 23.2 普通工具

```python
from pydantic_ai import Agent

agent = Agent("test")

@agent.tool_plain
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

result = agent.run_sync("计算 7 乘以 8")
print(result.output)
```

### 23.3 依赖注入

```python
from dataclasses import dataclass
from pydantic_ai import Agent, RunContext

@dataclass
class Deps:
    prefix: str

agent = Agent("test", deps_type=Deps)

@agent.tool
def format_name(ctx: RunContext[Deps], name: str) -> str:
    return f"{ctx.deps.prefix}{name}"

result = agent.run_sync(
    "格式化名称 robot",
    deps=Deps(prefix="demo/"),
)
print(result.output)
```

### 23.4 Pydantic 结构化输出

```python
from pydantic import BaseModel, Field
from pydantic_ai import Agent

class Pose(BaseModel):
    x: float
    y: float
    z: float
    confidence: float = Field(ge=0, le=1)

agent = Agent("test", output_type=Pose)
result = agent.run_sync("返回一个示例三维位姿")
print(result.output)
```

### 23.5 从 JSON Schema 创建工具

```python
from pydantic_ai import Agent, Tool


def move_to(**kwargs):
    xyz = kwargs["xyz"]
    return {"target": xyz, "accepted": True}

move_tool = Tool.from_schema(
    function=move_to,
    name="move_to",
    description="Move to world XYZ.",
    json_schema={
        "type": "object",
        "properties": {
            "xyz": {
                "type": "array",
                "items": {"type": "number"},
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["xyz"],
    },
    takes_ctx=False,
)

agent = Agent("test", tools=[move_tool])
```

### 23.6 图片工具返回

```python
from pydantic_ai import BinaryContent, ToolReturn


def image_result(image_bytes: bytes) -> ToolReturn:
    return ToolReturn(
        return_value={"status": "ok"},
        content=[
            BinaryContent(
                data=image_bytes,
                media_type="image/png",
            )
        ],
    )
```

### 23.7 异步图迭代

```python
async with agent.iter("执行任务") as run:
    async for node in run:
        if Agent.is_call_tools_node(node):
            print("模型请求调用工具")
        elif Agent.is_end_node(node):
            print("run 结束")
```

### 23.8 Usage 限制

```python
from pydantic_ai.usage import UsageLimits

result = await agent.run(
    "执行任务",
    usage_limits=UsageLimits(
        request_limit=10,
        total_tokens_limit=20_000,
        tool_calls_limit=20,
    ),
)
```

## 24. 源码阅读路线

### 24.1 先看 RPent 集成

1. `rpent/planner/base.py`
   - `build_planner()`
   - `_provider_factory()`
   - `infer_model()` 调用点
2. `rpent/planner/api_loop.py`
   - `ApiAgentLoop.solve()`
   - `_build_agent()`
   - `agent.iter()`
   - `_ApiRunObserver`
   - `_ApiDashboardSession`
   - Toolkit/图片转换

### 24.2 再看 Pydantic AI 核心

当前 Conda 环境路径：

```text
/home/hirobot/anaconda3/envs/rpent/lib/python3.10/site-packages/pydantic_ai/
```

建议顺序：

1. `models/__init__.py`
   - `ModelRequestParameters`
   - `Model`
   - `StreamedResponse`
   - `infer_model()`
   - tool resolution
2. `providers/__init__.py`
   - Provider 抽象和 `infer_provider()`
3. Agent 实现
   - 构造参数
   - `run()` / `iter()`
4. `messages.py`
   - request/response parts
   - tool events
5. `tools.py`
   - `Tool`
   - schema 和执行包装
6. `usage.py`
   - `RunUsage`
   - `UsageLimits`
7. `capabilities/`
   - `Thinking`
   - `ProcessHistory`

### 24.3 调试时的观察顺序

出现模型或工具问题时，按以下层次定位：

```text
模型 ID 是否正确
  ↓
Provider 是否正确构造、凭证/base_url 是否正确
  ↓
Model 路由是 Responses 还是 Chat
  ↓
Agent instructions/tools/settings 是否正确
  ↓
模型响应是否产生 ToolCallPart
  ↓
FunctionToolCallEvent 参数是否正确
  ↓
Toolkit.execute_tool 是否成功
  ↓
FunctionToolResultEvent 是否完整
  ↓
历史是否保持 call/result 配对
  ↓
usage / timeout / finish 是否终止循环
```

## 25. 总结

Pydantic AI 的价值不只是“调用一个 LLM API”，而是为 Agent 应用提供统一、类型安全且可观测的执行层：

- `Agent` 负责编排模型和工具循环；
- `Model` 统一供应商协议；
- `Provider` 管理连接、认证和 client；
- Pydantic/JSON Schema 验证工具参数和结构化输出；
- 类型化消息保证文本、thinking、工具和多模态历史可转换；
- `iter()` 暴露底层执行图，支持复杂交互控制；
- `RunUsage` 与 `UsageLimits` 管理成本和请求预算；
- capabilities 提供 thinking、历史处理和动态能力扩展。

在 RPent 中，Pydantic AI 被定位为 Provider 无关的高层推理后端：它负责模型请求和工具协议，RPent Toolkit 负责机器人技能、环境状态、安全取消和 artifact，`ApiAgentLoop` 则把两者连接起来，并统一输出为 `PlannerResult`。
