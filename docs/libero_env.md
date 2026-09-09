# RLinf `LiberoEnv` 详细解析

本文详细解释 RPent 当前 `rpent` Conda 环境中安装的 RLinf LIBERO 环境实现：

```text
/home/hirobot/anaconda3/envs/rpent/lib/python3.10/site-packages/
  rlinf/envs/libero/libero_env.py
```

分析基于该文件当前实际内容，而不是仅依据上游接口或另一个 checkout。目标文件信息：

```text
行数：809
SHA-256：ecc3343cdd6336704af3596028939147cc1269fe38a9c55ae5e3b5b2efd7d8d2
```

> `/home/hirobot/project/RLinf/rlinf/envs/libero/libero_env.py` 与安装副本的 SHA-256 不同，说明两者不是完全相同的源码版本。本文以用户指定的 Conda 环境安装副本为准。

## 1. `LiberoEnv` 的定位

`LiberoEnv` 是 RLinf 对 LIBERO/robosuite 环境的批量封装，继承自 `gym.Env`：

```python
class LiberoEnv(gym.Env):
    ...
```

它不是单个 MuJoCo 环境本身，而是一个向量环境协调层。它负责：

1. 根据配置选择 Standard、LIBERO-PRO 或 LIBERO-PLUS；
2. 将 benchmark task 和 trial 映射为全局 reset state ID；
3. 为每个环境创建独立的 `OffScreenRenderEnv` worker；
4. 支持运行中切换 task/BDDL 文件；
5. 将原始 observation 转成 VLA 使用的图片、机器人状态和任务文本；
6. 重新计算奖励、成功指标、termination 与 truncation；
7. 支持单步、动作块、部分 reset 和 auto-reset；
8. 暴露相机渲染及标定接口。
## 2. 总体架构

运行时存在三层进程/对象边界：

```text
RPent Planner / Toolkit 进程
        │
        │ HTTP 或 socket RPC
        ▼
RPent env_server 进程
  LiberoEnvFacade
        │
        ▼
RLinf LiberoEnv
        │
        │ multiprocessing spawn + Pipe
        ▼
ReconfigureSubprocEnvWorker（每个环境一个）
        │
        ▼
LIBERO OffScreenRenderEnv
        │
        ▼
robosuite + MuJoCo + EGL
```

本文重点是中间的 `RLinf LiberoEnv`。RPent 当前把它构造成：

```python
LiberoEnv(
    cfg=cfg,
    num_envs=1,
    seed_offset=0,
    total_num_processes=1,
    worker_info=None,
)
```

因此 RPent 实际只使用一个环境，但 `LiberoEnv` 本身仍保留完整的向量环境设计，所有主要返回值仍带 `num_envs` 维。RPent 的 `LiberoEnvFacade` 负责在 RPC 边界增加或移除这个维度。

## 3. 模块导入与 LIBERO 变体路由

模块导入时立即执行：

```python
libero_type = get_libero_type()
```

`get_libero_type()` 读取：

```bash
LIBERO_TYPE=standard   # 或 pro / plus
```

### 3.1 Standard

标准模式直接使用：

```python
from libero.libero.benchmark import Benchmark
```

worker 中使用：

```python
from libero.libero.envs import OffScreenRenderEnv
```

### 3.2 LIBERO-PRO / LIBERO-PLUS

PRO/PLUS 包使用不同顶层模块名。源码会导入实际包，再注册以下别名：

```text
sys.modules["libero"]
sys.modules["libero.libero"]
sys.modules["libero.libero.benchmark"]
sys.modules["libero.libero.envs"]
```

这样仍以 `libero.*` 为硬编码导入路径的依赖可以使用 PRO/PLUS 实现。

worker 进程由 `spawn` 创建，会重新启动 Python 解释器，因此父进程的 `sys.modules` 修改不能简单视为共享内存。`get_env_fns()` 生成的 worker factory 会再次设置 `LIBERO_TYPE`、执行模块路由，并设置：

```text
LIBERO_ASSET_ROOT
LIBERO_BDDL_PATH
LIBERO_INIT_STATES_PATH
```

这些变量分别指向所选变体包的资产、BDDL 和初始状态目录。

### 3.3 为什么移除包含 `opt/libero` 的 `sys.path`

PRO/PLUS 分支会过滤：

```python
sys.path[:] = [p for p in sys.path if "opt/libero" not in p]
```

目的是避免旧的标准 LIBERO 安装路径优先于当前变体包，从而发生“环境变量要求 PRO，但 Python 实际导入 Standard”的模块污染。

## 4. 构造函数状态

构造函数签名：

```python
def __init__(self, cfg, num_envs, seed_offset,
             total_num_processes, worker_info):
```

### 4.1 随机数

```python
self.seed = self.cfg.seed + seed_offset
self._generator = np.random.default_rng(seed=self.seed)
self._generator_ordered = np.random.default_rng(seed=0)
```

- `_generator`：用于普通随机 reset state 或训练期扰动 BDDL 选择；
- `_generator_ordered`：固定 seed 0，用于非评估模式下构造可重复的 ordered ID 序列。

worker 环境随后还会执行 `env.seed(seed)`。因此源码同时维护“协调层状态选择随机数”和“底层模拟环境随机数”。

### 4.2 分组

```python
self.group_size = cfg.group_size
self.num_group = self.num_envs // self.group_size
```

reset state 先按 group 选择，再通过：

```python
reset_state_ids.repeat(self.group_size)
```

复制给组内环境。这样一组环境可以共享 task/trial，但使用其他差异化条件进行并行采样。RPent 设置 `num_envs=1, group_size=1`，所以只有一个 group。

### 4.3 主要成员

| 成员 | 含义 |
|---|---|
| `task_suite` | benchmark 实例 |
| `trial_id_bins` | 每个 task 的初始状态数量 |
| `cumsum_trial_id_bins` | trial 数量累积边界 |
| `reset_state_ids` | 当前每个环境对应的全局 reset ID |
| `task_ids` | 每个环境的 task ID |
| `trial_ids` | 每个环境在 task 内部的 trial ID |
| `env` | `ReconfigureSubprocEnv` 向量 worker 管理器 |
| `current_raw_obs` | 最近一次底层原始 observation 列表 |
| `_elapsed_steps` | 每个环境当前 episode 的步数 |
| `success_once` | 当前 episode 是否曾经成功 |
| `returns` | 成功前累计的自定义 reward |
| `prev_step_reward` | 相对奖励模式的上一时刻 reward |
| `task_descriptions` | 每个环境的自然语言任务描述 |

### 4.4 初始化顺序

```text
读取配置与随机种子
  -> 创建 benchmark
  -> 统计每个 task 的 trial 数量
  -> 构造全部 ordered reset ID
  -> 选择当前 reset ID
  -> 映射出 task_id / trial_id
  -> 创建 worker 环境
  -> 初始化奖励和 episode 指标
```

worker 创建前必须先得到 task/trial，因为每个 `OffScreenRenderEnv` 的 BDDL 文件取决于当前 task。

## 5. Worker 多进程模型

`_init_env()` 执行：

```python
env_fns = self.get_env_fns()
self.env = ReconfigureSubprocEnv(env_fns)
```

`ReconfigureSubprocEnvWorker` 明确使用：

```python
ctx = multiprocessing.get_context("spawn")
```

每个 worker 拥有：

- 独立 Python 进程；
- 一对 `multiprocessing.Pipe`；
- 一个通过 cloudpickle 传入的环境 factory；
- 一个真正的 `OffScreenRenderEnv`。

worker 是 daemon process。主进程通过 Pipe 发送命令，支持：

```text
step, reset, close, render, render_camera, seed,
getattr, setattr, check_success, set_init_state,
reconfigure, env_call, get_camera_meta 等
```

### 5.1 为什么 factory 使用默认参数捕获

`get_env_fns()` 在循环中定义：

```python
def env_fn(param=env_fn_param, _type_val=current_type_val):
    ...
```

这里使用默认参数捕获当前循环值，避免 Python 闭包晚绑定导致所有 factory 最终都引用最后一个 `env_fn_param`。

### 5.2 `param.pop("seed")`

worker factory 从参数字典取出 seed：

```python
seed = param.pop("seed")
env = WorkerEnv(**param)
env.seed(seed)
```

`seed` 不传给 `OffScreenRenderEnv.__init__`，而是在构造后调用环境 seed 接口。

### 5.3 重配置

worker 收到 `reconfigure` 时：

```text
关闭旧 OffScreenRenderEnv
  -> 从新参数取 seed
  -> 构造新 OffScreenRenderEnv
  -> 设置 seed
```

因此 task/BDDL 切换不是简单修改一个字段，而是销毁并重建底层 robosuite 环境。

## 6. BDDL 文件选择

`get_env_fn_params()` 为每个目标环境生成：

```python
{
    **base_env_args,
    "bddl_file_name": final_path,
    "seed": self.seed,
}
```

`base_env_args` 来自：

```python
OmegaConf.to_container(self.cfg.init_params, resolve=True)
```

### 6.1 基础路径

首先读取 benchmark task：

```python
task = self.task_suite.get_task(self.task_ids[env_id])
folder_name = task.problem_folder
file_name = task.bddl_file
```

标准路径为：

```text
<bddl_root>/<problem_folder>/<bddl_file>
```

### 6.2 扰动后缀

后缀按以下优先级读取：

```text
LIBERO_SUFFIX
  > LIBERO_PERTURBATION
  > cfg.perturbation_suffix
```

#### PRO

允许的扰动目录后缀：

```text
_lan, _object, _swap, _task
```

`all` 表示搜索全部四类。候选目录必须同时包含 suite keyword 并以所选扰动后缀结尾；文件名需要包含原始 task 核心名称。

#### PLUS

`all` 模式先从文件名中去除 `_view`、`_initstate`、`_noise`、`_sample`、`_light` 等扰动 marker，再跨相关 suite 目录搜索包含基础 task 名的 BDDL 文件。

### 6.3 评估与训练的候选选择

评估模式：

```python
all_candidates[(self.seed + idx_offset) % len(all_candidates)]
```

选择是确定性的，并利用 `idx_offset` 让同一次批量重配置中的不同环境分散到不同候选。

非评估模式：

```python
self._generator.choice(all_candidates)
```

使用环境随机生成器采样。

如果没有搜索到扰动候选，`final_path` 保持原始 BDDL 路径，不会因空候选直接失败。

`task_descriptions` 始终使用 benchmark task 的 `task.language`。切换到扰动 BDDL 不会自动改写自然语言描述。
## 7. Reset State ID 系统

这是文件中最重要也最容易混淆的部分。

### 7.1 全局 ID 与 task 内 trial ID

假设 suite 中：

```text
task 0 有 3 个 trial
task 1 有 2 个 trial
task 2 有 4 个 trial
```

则全局 reset state ID 空间为：

```text
全局 ID 0,1,2     -> task 0, trial 0,1,2
全局 ID 3,4       -> task 1, trial 0,1
全局 ID 5,6,7,8   -> task 2, trial 0,1,2,3
```

代码通过：

```python
self.trial_id_bins = [3, 2, 4]
self.cumsum_trial_id_bins = [3, 5, 9]
```

建立映射边界。

### 7.2 `task_id_filter`

如果配置提供 `task_id_filter`：

1. 每个值必须是整数；
2. 必须位于 `[0, num_tasks-1]`；
3. 去重并排序；
4. 展开成这些 task 对应的全部全局 reset ID。

之后随机与 ordered 选择都只在 `_valid_reset_state_ids` 中进行。

### 7.3 随机选择

`_get_random_reset_state_ids()` 优先级：

```text
specific_reset_id
  > task_id_filter 展开的合法 ID
  > suite 全部 ID
```

如果设置了 `specific_reset_id`，所有 group 都得到同一个 ID。

### 7.4 评估交错顺序

评估模式使用：

```text
(task0, trial0), (task1, trial0), ..., (taskN, trial0),
(task0, trial1), (task1, trial1), ...
```

而不是先遍历完 task 0 的全部 trial。这种 interleaved 顺序使多个并行进程更早覆盖不同任务，避免前几个进程都集中在同一 task。

### 7.5 跨进程分配

`get_reset_state_ids_all()` 会：

1. 获取过滤后、评估交错或完整 ID 数组；
2. 非评估模式用固定 RNG 打乱；
3. 如果 ID 数量少于 `total_num_processes`，重复平铺；
4. 截断为进程数的整数倍；
5. reshape 为 `[total_num_processes, ids_per_process]`。

`seed_offset` 随后作为第一维索引，决定当前进程消费哪一行 ID。

### 7.6 Ordered 游标

`_get_ordered_reset_state_ids()` 通过 `start_idx` 顺序取 ID。若剩余数量不足：

```text
重新构造 reset_state_ids_all
  -> start_idx 归零
  -> 从新序列开头继续
```

若设置 `specific_reset_id`，则不使用游标，直接返回重复的固定 ID。

### 7.7 Group 展开

`update_reset_state_ids()` 先为 `num_group` 选择 ID，再：

```python
self.reset_state_ids = reset_state_ids.repeat(self.group_size)
```

因此连续的 `group_size` 个环境共享同一个 task/trial。

## 8. 观测处理

底层 `OffScreenRenderEnv` 返回 robosuite 原始 observation 字典。`LiberoEnv` 将其转换为 VLA 输入。

### 8.1 主相机

```python
img = obs["agentview_image"]
img = img[::-1, ::-1]
```

主图像在两个空间轴上翻转，相当于旋转 180°。源码明确说明这是为了匹配训练预处理。

### 8.2 腕部相机

```python
img = obs["robot0_eye_in_hand_image"]
img = img[::-1, ::-1]
```

腕部图像执行相同的 180° 旋转。辅助函数虽然带 `resize_size` 参数，但当前实现没有实际 resize。

### 8.3 机器人状态

状态由三部分拼接：

```python
state = np.concatenate([
    obs["robot0_eef_pos"],
    quat2axisangle(obs["robot0_eef_quat"]),
    obs["robot0_gripper_qpos"],
])
```

通常可理解为：

```text
末端位置 3 维
+ 四元数转换后的 axis-angle 3 维
+ 夹爪关节位置
```

实际总维数取决于 `robot0_gripper_qpos` 的形状。

`quat2axisangle()` 假定四元数排列为 `(x,y,z,w)`，将 `w` 裁剪到 `[-1,1]`，接近零旋转时直接返回零向量，避免除以接近零的分母。

### 8.4 批量包装

`_wrap_obs()` 对每个环境执行提取，再通过：

```python
list_of_dict_to_dict_of_list(...)
to_tensor(...)
```

转成批量 tensor，并对图片执行 `clone()` 后 `torch.stack()`。

输出协议：

```python
{
    "main_images": Tensor[num_envs, H, W, C],
    "wrist_images": Tensor[num_envs, H, W, C],
    "states": Tensor[num_envs, state_dim],
    "task_descriptions": list[str],
}
```

源码没有在这里调整通道维顺序；具体 dtype/device 由 `to_tensor()` 决定。

### 8.5 `current_raw_obs`

`current_raw_obs` 保存每个 worker 最近的原始 observation：

- 构造后为 `None`；
- reset 时首次初始化为长度 `num_envs` 的列表；
- 部分 reset 只替换目标索引；
- step 时整体替换为最新 worker 返回值。

RPent 的 `env.raw_obs()` 使用这个字段获得深度、相机原始键等没有进入 VLA 包装观测的数据。

## 9. Reset 完整流程

方法签名：

```python
def reset(self, env_idx=None, reset_state_ids=None):
```

支持全部环境 reset，也支持只 reset 指定索引。

### 9.1 环境索引

`env_idx=None` 时：

```python
env_idx = np.arange(self.num_envs)
```

### 9.2 首次 reset

首次 reset 时 `self.is_start=True`：

```python
reset_state_ids = (
    self.reset_state_ids if self.use_fixed_reset_state_ids else None
)
self._is_start = False
```

如果没有固定 ID，后续进入随机选择。

### 9.3 普通 reset state 选择

若调用方没有显式提供 ID：

```python
reset_state_ids = self._get_random_reset_state_ids(len(env_idx))
```

注意：只有首次 reset 的 fixed ID 路径使用预先计算的 `self.reset_state_ids`。后续手工 reset 若不传 ID，会重新随机选择，除非 auto-reset 路径显式传入固定或 ordered ID。

### 9.4 `_reconfigure()`

该方法先把新全局 ID 转成 task/trial，然后逐环境判断 task 是否改变。

重建底层环境的条件：

```python
if task_changed or not cfg.is_eval:
    reconfig_env_idx.append(env_id)
```

因此：

- 评估模式下，同一个 task 只换 trial 时不重建环境；
- 非评估模式下，即使 task 不变也重建，以允许重新采样扰动 BDDL。

如果需要重建：

```text
生成新 env_fn 参数
  -> worker 关闭旧环境
  -> 构造新 OffScreenRenderEnv
  -> 设置 seed
```

随后对目标 worker 执行：

```python
self.env.seed(self.seed * len(env_idx))
self.env.reset(id=env_idx)
```

对 Standard/PRO，还会从 benchmark 取 init state 并调用 `set_init_state()`；PLUS 分支跳过这一步。

### 9.5 15 个零动作 warm-up

环境配置完成后固定执行 15 步：

```python
zero_actions = np.zeros((len(env_idx), 7))
if cfg.reset_gripper_open:
    zero_actions[:, -1] = -1
```

这会让机器人和仿真在 reset 后稳定。最后一次 warm-up 返回的 `raw_obs` 成为 reset observation。

需要注意：这 15 步直接调用底层 `self.env.step()`，不会经过 `LiberoEnv.step()`，因此：

- 不增加 `_elapsed_steps`；
- 不累计 `returns`；
- 不执行 `_calc_step_reward()`；
- 中间 warm-up observation 被覆盖，只保留最后一次；
- warm-up 的 termination 和 info 被读取但没有用于结束 episode。

### 9.6 reset 返回

```text
更新 current_raw_obs
  -> _wrap_obs(全部环境当前观测)
  -> 重置目标环境 metrics
  -> 返回 (obs, {})
```

即使只 reset 部分环境，返回的包装 observation 仍覆盖全部 `num_envs`，未 reset 位置沿用 `current_raw_obs` 中的旧值。

## 10. 单步 `step()`

签名：

```python
def step(self, actions=None, auto_reset=True):
```

### 10.1 动作转换

如果动作是 torch Tensor：

```python
actions = actions.detach().cpu().numpy()
```

NumPy 动作则原样交给 worker。预期形状为：

```text
[num_envs, action_dim]
```

RPent Facade 会把单环境 `[action_dim]` 扩成 `[1,action_dim]`。

### 10.2 步数和底层推进

```python
self._elapsed_steps += 1
raw_obs, _reward, terminations, info_lists = self.env.step(actions)
```

计数在调用 worker 前增加。如果 worker step 抛出异常，计数已增加，但该异常通常会导致 RPent 当前 TaskRun 失败并重建环境。

底层 `_reward` 被命名为下划线变量，后续没有使用。RLinf 按自己的规则重新计算 `step_reward`。

### 10.3 Truncation

```python
truncations = self.elapsed_steps >= cfg.max_episode_steps
```

因此达到上限的那一步就返回 truncation。

### 10.4 自定义奖励

```python
step_penalty = -1 if use_step_penalty else 0
termination_bonus = reward_coef * terminations
reward = step_penalty + termination_bonus
```

默认 RPent 配置为：

```text
use_step_penalty=False
reward_coef=1.0
```

所以 reward 通常等于 termination 成功信号。

若 `use_rel_reward=True`：

```python
reward_diff = reward - prev_step_reward
prev_step_reward = reward
```

返回奖励变化量而不是绝对值。成功从 0 变 1 时得到正奖励；后续从 1 回到 0 可能产生负差值，具体取决于 termination 信号语义。

### 10.5 指标

`_record_metrics()`：

1. 只在尚未成功的环境中累计 return；
2. 首次 termination 时记录成功步数；
3. `success_once` 对 termination 做粘滞 OR；
4. 向 `infos["episode"]` 写入：
   - `success_once`
   - `return`
   - `episode_len`
   - `reward = return / max(成功时步数或当前步数, 1)`

这里 `infos["episode"]["reward"]` 是 episode 级平均指标，不等于当前 step 返回的 reward。

### 10.6 `ignore_terminations`

如果启用：

```python
infos["episode"]["success_at_end"] = terminations
terminations[:] = False
```

成功信息被保存到 info，但对外 termination 清零。truncation 不受此开关影响。

### 10.7 Auto-reset

```python
dones = terminations | truncations
_auto_reset = auto_reset and self.auto_reset
```

只有方法参数和配置都为真才自动 reset。RPent 配置 `cfg.auto_reset=False`，所以不会在单步后自动切换 episode。

### 10.8 返回值

```python
(
    wrapped_obs,
    to_tensor(step_reward),
    to_tensor(terminations),
    to_tensor(truncations),
    infos,
)
```

所有主要数组都保留 `num_envs` 首维。
## 11. 动作块 `chunk_step()`

输入形状：

```text
[num_envs, chunk_size, action_dim]
```

实现不是底层一次性批处理，而是 Python 循环逐步执行：

```python
for i in range(chunk_size):
    actions = chunk_actions[:, i]
    ... = self.step(actions, auto_reset=False)
```

### 11.1 为什么禁止块内 auto-reset

每个内部 step 都传 `auto_reset=False`，确保动作块中途成功或超时后不会立刻切换到新 episode，再把剩余动作执行到新场景。

但是当前实现仍会继续执行 chunk 中剩余动作，因为循环没有在 termination/truncation 后 break。也就是说：

- 原始 termination/truncation 会逐步记录；
- 动作块不会提前截断；
- 如果底层环境允许，done 后的剩余 action 仍会被执行；
- RPent 外层 Facade 只在整个 `chunk_step()` 返回后才把 episode 标记为 done。

这是阅读和调用该接口时必须理解的行为。

### 11.2 堆叠结果

循环结束后：

```text
chunk_rewards          [num_envs, chunk_size]
raw_chunk_terminations [num_envs, chunk_size]
raw_chunk_truncations  [num_envs, chunk_size]
obs_list               长度 chunk_size，每项都是批量 observation
infos_list             长度 chunk_size
```

### 11.3 块级 done

```python
past_terminations = raw_chunk_terminations.any(dim=1)
past_truncations = raw_chunk_truncations.any(dim=1)
past_dones = past_terminations | past_truncations
```

如果配置允许 auto-reset，则在整个块执行完之后，只对 done 环境调用一次 `_handle_auto_reset()`，并替换 `obs_list[-1]` 和 `infos_list[-1]`。

### 11.4 结束信号压缩

若 `self.auto_reset` 或 `self.ignore_terminations` 为真，返回信号被改写为：

```text
前 chunk_size-1 个位置全部 False
最后一个位置写入该块是否曾 termination/truncation
```

否则保留每一步原始信号。

RPent 当前配置：

```text
auto_reset=False
ignore_terminations=False
```

因此 RPent 会收到每一步的原始 termination/truncation 数组。

### 11.5 RPent 如何使用动作块

VLA 返回单环境动作：

```text
[chunk_size, 7]
```

RPent `LiberoEnvFacade` 增加环境维：

```text
[1, chunk_size, 7]
```

`LiberoEnv.chunk_step()` 返回批量结果后，Facade 再移除首维。非录制模式通常只把最后 observation 返回给 primitive；录制模式请求全部帧，以生成连续动作视频。

## 12. Auto-reset 细节

`_handle_auto_reset(dones, final_obs, infos)` 会先深复制结束时 observation/info：

```python
final_obs = copy.deepcopy(_final_obs)
final_info = copy.deepcopy(infos)
```

然后找出 done 环境：

```python
env_idx = np.arange(num_envs)[dones]
```

### 12.1 新 ID 选择

- 评估模式：从 ordered 序列取下一组 ID，并更新目标位置；
- 非评估 + fixed reset：重新调用 `update_reset_state_ids()`；
- 其他情况：reset 内部随机选择。

### 12.2 返回 observation 的含义

auto-reset 后返回的是**新 episode 的 reset observation**，而不是结束 episode 的最后 observation。结束 observation 被放入 info：

```python
infos["final_observation"] = final_obs
infos["final_info"] = final_info
infos["_final_info"] = dones
infos["_final_observation"] = dones
infos["_elapsed_steps"] = dones
```

这遵循常见 Gymnasium 向量环境约定：主 observation 已属于下一 episode，`final_observation` 保存上一 episode 结束状态。

## 13. 相机接口

### 13.1 `render_camera()`

`LiberoEnv` 直接使用 worker 0：

```python
self.env.workers[0].render_camera(...)
```

worker 逐层解包 wrapper，找到 robosuite 对象及 `sim`，然后执行：

```python
sim.render(
    width=width,
    height=height,
    camera_name=camera_name,
    depth=depth,
)
```

这是一条同步 Pipe RPC。它不调用 `LiberoEnv.step()`，不会增加 `_elapsed_steps` 或修改奖励指标。

### 13.2 `get_camera_meta()`

同样只查询 worker 0，返回：

```python
{
    "camera_name": str,
    "height": int,
    "width": int,
    "intrinsic_K": 3x3 list,
    "extrinsic_cam2world": 4x4 list,
    "depth_near": float,
    "depth_far": float,
}
```

内参使用目标 height/width 计算。外参是 camera-to-world 4x4 变换。near/far 由 MuJoCo 模型中的相对裁剪面乘以 `sim.model.stat.extent` 得到。

源码注释认为 `agentview` 是固定世界相机，因此其外参在 episode 内固定；腕部相机随机器人末端运动，调用方不能跨 step 复用其外参。

### 13.3 仅 worker 0 的限制

这些接口没有接受 env index，始终查询第一个 worker。对 RPent 的 `num_envs=1` 没有歧义；若其他调用方使用多环境并行，它们不能通过这两个方法查询 worker 1..N 的相机。

## 14. 指标与奖励字段区分

容易混淆的字段如下：

| 字段 | 含义 |
|---|---|
| worker `_reward` | 底层 LIBERO/robosuite reward，本类当前丢弃 |
| `step_reward` | `_calc_step_reward()` 重新计算的当前奖励 |
| `returns` | 首次成功前累计的 step reward |
| `infos["episode"]["reward"]` | return 除以 episode 长度的指标 |
| `terminations` | 成功/底层终止信号，可能被 ignore_terminations 清零 |
| `truncations` | `_elapsed_steps >= max_episode_steps` |
| `success_once` | 当前 episode 是否曾出现 termination |

`fail_once` 在构造和 reset 中初始化，但当前文件没有在 `_record_metrics()` 中更新或输出，属于预留/未使用状态。

`info_logging_keys` 当前返回空列表，表示本类没有额外声明统一日志键；episode 指标仍直接写入 `infos`。

## 15. RPent 集成

RPent 不直接在 Planner 进程导入这个类。集成链路为：

```text
robots/libero/env_server.py
  -> build_env_cfg()
  -> make_env()
  -> LiberoEnv(num_envs=1)
  -> LiberoEnvFacade
  -> HTTP/socket RPC
  -> LiberoEnvClient
  -> LiberoPrimitives
```

### 15.1 RPent 配置

RPent 当前设置：

```text
auto_reset=False
ignore_terminations=False
use_rel_reward=False
use_step_penalty=False
reward_coef=1.0
reset_gripper_open=True
is_eval=True
group_size=1
use_fixed_reset_state_ids=True
use_ordered_reset_state_ids=True
camera size=256x256
camera_depths=True
```

`make_env()` 在创建 `LiberoEnv` 前把 `(task_id, seed)` 映射成 suite 全局 `specific_reset_id`，所以 RPent 的首次 reset 使用确定的 task/trial。

### 15.2 RPC 形状转换

```text
LiberoEnv 输入 action        [1, 7]
RPC 客户端输入 action        [7]

LiberoEnv 输入 action chunk  [1, T, 7]
RPC 客户端输入 action chunk  [T, 7]

LiberoEnv observation         [1, ...]
RPC 客户端 observation        [...]
```

RPC 服务端还会递归执行：

```python
tensor.detach().cpu().numpy()
```

因此网络客户端收到 NumPy/Python 对象，不直接持有 RLinf torch Tensor。

### 15.3 原始观测与世界坐标

VLA 使用包装后的：

```text
main_images, wrist_images, states, task_descriptions
```

RPent artifact 逻辑还通过 `current_raw_obs` 获取原始 RGB/depth，并结合 `get_camera_meta()` 反投影为逐像素世界坐标图。SAM3 mask 再与同分辨率 world map 对齐，计算目标 `world_xyz`。

### 15.4 不使用 auto-reset 的原因

机器人任务执行需要明确知道 episode 已成功或超时，不能在同一个 Planner TaskRun 中悄悄切换到新场景。因此 RPent 关闭 auto-reset，并在客户端和 Facade 两侧粘滞记录结束状态；结束后继续 step 会被拒绝。

## 16. 关键边界和潜在风险

### 16.1 安装副本与 checkout 不一致

当前 Conda 安装文件与 `/home/hirobot/project/RLinf` checkout 不同。修改 checkout 不一定影响运行中的 `rpent` 环境，反之修改 site-packages 也不会自动同步回仓库。调试时应首先确认 Python 实际导入路径：

```bash
conda activate rpent
python -c "import inspect, rlinf.envs.libero.libero_env as m; print(inspect.getfile(m))"
```

### 16.2 模块路由依赖导入时环境变量

`libero_type` 在模块导入时计算。进程已经导入模块后再修改 `LIBERO_TYPE`，不会自动重新执行顶层 Benchmark 路由。worker factory 会重新设置变量和模块别名，但父进程初始化仍要求在首次 import 前正确配置环境。

### 16.3 `spawn` 要求依赖可重新导入

worker 不是 fork 出来的父内存快照。环境 factory 及相关参数必须可 cloudpickle，worker 启动时模块导入必须在全新解释器中成功，资产路径和环境变量也必须可继承。

### 16.4 `chunk_step()` 不提前停止

即使动作块中间已成功或截断，剩余动作仍继续执行。调用方应使用合理 chunk 大小，并理解最终状态可能位于首次 success 之后若干动作。

### 16.5 Reset warm-up 忽略中间 done

15 个零动作直接调用底层 vector env，未检查中间 termination。它们用于场景稳定，不计入 episode 指标；若定制任务可能在 warm-up 中意外成功，需要单独审查这一行为。

### 16.6 `step()` 先增加 elapsed step

worker 调用前 `_elapsed_steps += 1`。若 worker 抛错后继续复用同一对象，步数会多计；RPent 当前通常在此类故障后销毁整个 env_server，因此不会尝试在不确定状态上继续。

### 16.7 相机 API 固定 worker 0

多环境调用者只能通过当前便捷方法访问 worker 0 相机。RPent 单环境模式不受影响。

### 16.8 底层 reward 被丢弃

如果自定义 LIBERO 任务依赖非二值 dense reward，本类不会直接透传它。必须确认 `_calc_step_reward()` 的 termination bonus/step penalty 逻辑符合训练或评估目标。

## 17. 调试建议

### 17.1 确认实际导入文件

```bash
conda activate rpent
python -c "import inspect; from rlinf.envs.libero.libero_env import LiberoEnv; print(inspect.getfile(LiberoEnv))"
```

### 17.2 确认变体

```bash
python -c "from rlinf.envs.libero.utils import get_libero_type; print(get_libero_type())"
```

预期是 `standard`、`pro` 或 `plus`。

### 17.3 Reset 失败

依次检查：

1. `LIBERO_TYPE` 与安装包是否匹配；
2. BDDL、assets 和 init state 路径；
3. task ID 是否在 benchmark 范围内；
4. `specific_reset_id` 是否能映射到合法 task/trial；
5. worker spawn 日志中的 import 或 EGL 错误；
6. `LIBERO_ROBOT_BASE` 是否指定了底层支持的机器人。

### 17.4 图像方向错误

包装观测中的主图和腕图已旋转 180°。原始 `current_raw_obs`、直接 `render_camera()` 和包装图像的方向语义并不完全相同。生成深度/world map 时必须对 RGB 与 depth 应用一致变换。

### 17.5 Episode 提前结束

分别检查：

- `terminations`：通常对应任务成功；
- `truncations`：由 max episode steps 产生；
- `ignore_terminations`：是否把成功信号清零；
- `chunk_step()`：是否在块中间已经出现 done；
- Facade/Client 粘滞标志：是否已经拒绝后续推进。

## 18. 核心流程图

```text
构造 LiberoEnv
  │
  ├─ 路由 Standard / PRO / PLUS
  ├─ 加载 benchmark
  ├─ 统计 task -> trial bins
  ├─ 选择全局 reset state IDs
  ├─ 映射 task IDs / trial IDs
  └─ spawn OffScreenRenderEnv workers

reset(env_idx)
  │
  ├─ 选择 reset IDs
  ├─ task 变化时重建 worker 环境
  ├─ worker reset + set_init_state
  ├─ 15 次零动作 warm-up
  ├─ 更新 current_raw_obs
  ├─ 包装图片/状态/任务文本
  └─ 清零目标 episode metrics

step(actions)
  │
  ├─ elapsed_steps += 1
  ├─ worker.step
  ├─ 包装最新 observation
  ├─ 根据 termination 计算自定义 reward
  ├─ 更新 success/return/episode_len
  ├─ 计算 truncation
  ├─ 可选 auto-reset
  └─ 返回五元组

chunk_step([N,T,A])
  │
  ├─ 循环 T 次 step(auto_reset=False)
  ├─ 堆叠 reward/termination/truncation
  ├─ 汇总块内 done
  ├─ 块结束后可选 auto-reset
  └─ 返回 obs_list + [N,T] 信号
```

## 19. 总结

`LiberoEnv` 是连接 benchmark 定义、多进程 robosuite 环境和 VLA 批量观测协议的核心协调层。理解它时应抓住五个关键点：

1. **它是向量环境，不是真正 MuJoCo worker 本身**；
2. **task/trial 通过 suite 全局 reset state ID 统一管理**；
3. **真实环境位于 spawn 子进程，task 切换可能重建 worker**；
4. **返回给策略的图像和状态经过专门包装，底层 reward 被重新计算**；
5. **chunk 会完整执行，不会在中间 done 时提前停止**。

RPent 在它之外增加了单环境 RPC 适配、Tensor 到 NumPy 转换、metadata 防错连及 episode 结束保护，使 Planner 可以在不直接加载 MuJoCo 的进程中安全使用该环境。