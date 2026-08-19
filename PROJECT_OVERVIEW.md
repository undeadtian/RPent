# RPent 项目架构、流程与功能总结

## 一句话概括

**RPent 是一个面向具身智能和机器人任务的 Agent 编排框架。** 它让大语言模型负责高层规划，通过工具调用组合 **Pi0.5 VLA 策略、解析式机器人动作、SAM3 视觉分割和 RGB-D 空间定位**，形成“观察 → 推理 → 执行 → 再观察”的闭环。

它不是单独训练 VLA 的框架，更接近机器人领域的 **Agent Runtime / Orchestrator**。

## 整体架构

```text
CLI / 交互终端 / Web Dashboard
              │
              ▼
      EnvSpec + PromptBundle
      环境发现、参数和提示词
              │
              ▼
            Planner
   API / Claude Code / Codex
              │
        读取工具定义并决策
              ▼
            Toolkit
   工具注册、调用分发、状态采集
              │
       ┌──────┼───────────┐
       ▼      ▼           ▼
   Pi0.5 VLA  解析式动作   SAM3 / RGB-D
   抓取/接触  移动/旋转    分割/空间定位
       │      │           │
       └──────┼───────────┘
              ▼
        LIBERO 环境服务
              │
              ▼
 EnvState：图像、深度、坐标、执行结果
              │
              └────反馈给 Planner，进入下一轮
```

框架主要分为以下几层：

| 层次 | 目录/文件 | 职责 |
|---|---|---|
| 入口与调度 | `rpent/cli/main.py` | 参数解析、运行时初始化、启动 Planner、清理资源、保存结果 |
| 环境插件 | `rpent/envs/`、`robots/libero/__init__.py` | 动态发现环境，通过 `EnvSpec` 注入参数、Prompt 和初始化逻辑 |
| 规划器 | `rpent/planner/` | 封装 API、Claude Code、Codex 三种 LLM Agent 后端 |
| 工具系统 | `rpent/tools/toolkit.py` | 注册工具、生成工具 Schema、执行调用、捕获执行后状态 |
| 机器人能力 | `robots/libero/tools.py` | Pi0.5、位置控制、姿态控制、夹爪、分割、反投影等能力 |
| 服务与 RPC | `robots/libero/*_server.py`、`rpent/utils/` | 环境、VLA、SAM3 服务以及 HTTP/socket 通信 |
| 状态与产物 | `EnvState` 相关代码 | 保存每一步状态、RGB-D、世界坐标、视频和 recipe |
| Dashboard | `rpent/dashboard/` | FastAPI + SSE 的运行监控和交互界面 |

## 主要运行流程

### 1. 解析命令并加载环境

程序入口是：

```text
rpent -> rpent.cli.main:main
```

`main()` 先读取 `--env`，然后动态导入：

```text
robots.<env_name>
```

再通过对应环境的 `get_env_spec()` 得到：

- 环境专属参数；
- System/User Prompt；
- 配置解析函数；
- 运行时初始化函数；
- Dashboard 扩展配置。

当前仓库真正实现的环境只有 `robots/libero`。

### 2. 构建任务配置与 Planner

LIBERO 配置会根据：

```text
suite + task + seed
```

生成任务标识、输出目录和 Prompt 变量，例如：

```text
object_swap_t2_s0
```

然后通过 `build_planner()` 创建以下一种规划器：

- `api`：通过 Pydantic AI 接入 Anthropic、OpenAI 或 OpenAI-compatible API；
- `claude_code`：使用 Claude Agent SDK；
- `codex`：使用 OpenAI Codex SDK。

三种后端都遵循统一的 `Planner.solve()` 接口，输入包括：

- System Prompt；
- 用户任务；
- Toolkit；
- 最大轮数；
- 可选交互输入。

### 3. 启动或连接运行服务

普通本地模式下，Agent 主进程会额外启动三个服务：

1. `env_server`：运行 LIBERO/RLinf 仿真环境；
2. `vla_server`：加载 Pi0.5 并预测低层动作序列；
3. `sam3_server`：提供文本或点提示的目标分割。

加上主进程，本地完整运行通常包含四个进程。

三个服务也可以通过参数连接到已有远端服务：

```text
--env-endpoint
--vla-endpoint
--sam3-endpoint
```

系统支持 HTTP 和 socket，因此可以把 GPU 模型与仿真环境拆到不同机器部署。Dashboard 模式会复用 VLA/SAM3 服务，每个任务单独创建环境服务。

### 4. 初始化 Toolkit 和环境状态

`LiberoToolkit` 组合通用工具和 `LiberoPrimitives`，随后：

- 重置 LIBERO 环境；
- 获取初始机器人状态；
- 保存第一帧 RGB-D 数据；
- 创建 `EnvState`；
- 开始录制 episode 视频。

Toolkit 是 Planner 与机器人系统之间的统一边界：Planner 只看到工具名称、描述和 JSON Schema，不需要了解 RPC、模型加载或仿真细节。

### 5. LLM 进入工具调用闭环

Planner 每一轮大致执行：

```text
读取当前观察
→ 判断下一步
→ 选择工具和参数
→ Toolkit 执行
→ 环境或模型服务处理
→ 自动记录新状态
→ 图像和执行结果返回 Planner
→ 再次判断
```

工具能力主要分成三类。

#### VLA primitive

- `pi0_pick`：让 Pi0.5 按自然语言执行抓取，循环预测动作直到检测到下降、闭爪和抬升；
- `pi0_doubled`：适合旋钮、推、拨动等接触式操作，以 LIBERO 官方终止条件判断任务成功。

调用路径为：

```text
Planner
→ Toolkit
→ LiberoPrimitives
→ VLAClient
→ vla_server / Pi0.5
→ 动作序列
→ env.chunk_step()
```

#### 解析式 primitive

包括：

- `move_to`：移动末端执行器到世界坐标；
- `move_pose`：同时调整位置、俯仰角和偏航角；
- `rotate_wrist`：旋转手腕；
- `rotate_pitch`：调整夹爪俯仰；
- `release`：释放物体；
- `set_gripper`：控制夹爪开合。

这些动作绕过 VLA，直接生成 LIBERO OSC 控制器所需的 7 维动作。

#### 感知与定位工具

包括：

- 查看当前或历史环境状态；
- 读取主相机和腕部相机图像；
- 使用 SAM3 按文本或像素点分割物体；
- 将图像像素结合深度与相机外参反投影为世界坐标；
- 保存分割 Mask、Overlay 和目标坐标。

因此 Agent 可以完成类似流程：

```text
分割杯子
→ 获得杯子世界坐标
→ 移动到杯子上方
→ 调用 Pi0.5 抓取
→ 查看腕部相机确认
→ 移动并释放
```

### 6. 自动状态反馈

每次执行非只读工具后，`Toolkit.execute_tool()` 会自动捕获：

- 机器人末端位置与姿态；
- 夹爪状态；
- 任务是否 terminated/truncated；
- 主相机和腕部相机 RGB；
- 深度图；
- 每个像素对应的世界坐标；
- 执行命令、参数、耗时和结果。

这些内容被写入 `EnvState`，同时以文本和 Base64 图像返回给多模态 Planner。

这是项目最核心的闭环机制：**模型不是一次性生成完整动作，而是每执行一个 primitive 就重新观察和修正。**

### 7. 任务结束和结果持久化

任务通常由模型调用 `finish(status, summary)` 结束，也可能因为以下条件中止：

- 达到 `max_turns`；
- Planner 超时；
- 环境达到最大步数；
- 环境 terminated/truncated；
- 用户在交互模式中断。

结束阶段会输出：

- 每轮环境状态与图像；
- `episode.mp4`；
- Planner 对话 transcript；
- Agent 和服务日志；
- 成功 primitive 序列生成的 `recipe_*.jsonl`；
- Claude/Codex 的原始输出记录。

Recipe 可以作为后续任务参考或复用，但目前不是一个自动执行训练的学习系统。

## 主要功能

1. **可替换的高层规划器**  
   支持 API、Claude Code、Codex，并统一到 `Planner` 协议。

2. **VLA 与解析式控制混合编排**  
   模糊视觉操作交给 Pi0.5，精确位姿移动交给传统控制 primitive。

3. **多模态闭环反馈**  
   每个动作后都重新采集图像、深度、机器人状态和执行结果。

4. **视觉目标定位**  
   SAM3 分割结合 RGB-D 反投影，可把自然语言目标转换成世界坐标。

5. **LIBERO Benchmark 支持**  
   支持 standard、pro、plus 变体，以及 spatial、object、goal、10、90、swap/task 等 suite 配置。

6. **本地与分布式部署**  
   环境、VLA、SAM3 均可本地启动，也可独立部署后通过 HTTP/socket 连接。

7. **CLI、交互模式和 Dashboard**  
   支持普通批处理、终端实时干预以及 Web Dashboard 监控。

8. **完整运行记录**  
   保存状态、图像、深度、世界坐标、视频、日志、对话和可复用 recipe。

## 技术栈

主要技术包括：

- Python 3.10–3.12；
- Pydantic / Pydantic AI；
- Claude Agent SDK；
- OpenAI Codex SDK；
- FastAPI、Uvicorn、httpx；
- RLinf / LIBERO；
- OpenPI Pi0.5；
- SAM3；
- NumPy、SciPy、imageio；
- HTTP 与自定义 socket RPC。

## 当前实现边界

需要区分 README 中的愿景和仓库当前代码：

- 当前真正可运行的机器人环境只有 **LIBERO**；
- 文档中出现的 RoboCasa、Franka、SO-101、RLDX-1、DreamZero 尚未在当前源码中实现；
- 项目所说的“递归、自进化”，目前主要表现为：
  - 多轮观察—执行—反馈；
  - 文件式记忆；
  - 历史状态与 recipe 复用；
- 目前没有自动训练、自动更新模型、自动反思或自动发现新技能的完整流水线；
- Pi0.5、LIBERO-PRO、RLinf 和 SAM3 依赖外部包、模型 checkpoint、GPU/CUDA 及额外资产；
- Claude Code/Codex 能访问环境 memory 目录，API Planner 的长期记忆接入目前并不完全一致。

## 推荐阅读顺序

如果要继续深入源码，建议依次查看：

1. `rpent/cli/main.py`：完整主流程；
2. `robots/libero/__init__.py`：LIBERO 配置与三个服务初始化；
3. `rpent/planner/base.py`：Planner 抽象和后端选择；
4. `rpent/tools/toolkit.py`：工具分发与状态捕获；
5. `robots/libero/toolkit.py`：LIBERO 工具注册；
6. `robots/libero/tools.py`：VLA、控制、分割和状态保存的具体实现。

## 总结

RPent 的核心价值可以概括为：

> **把 LLM 的任务规划能力、VLA 的视觉动作能力、传统机器人控制和视觉空间感知，封装成一个可观测、可组合、可远程部署的具身 Agent 执行闭环。**
