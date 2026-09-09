"""在独立进程中托管单环境 LIBERO，并通过 RPC 暴露环境操作。

本模块负责把 LIBERO/RLinf 的 batch-first 环境接口适配成客户端使用的单环境协议：
入站动作补上大小为 1 的环境维，出站观测、奖励和结束信号移除该维，并在 RPC wire
边界前把 torch tensor 递归转换为 CPU numpy 对象。facade 还提供原始观测、相机
渲染/标定、任务文本与缓存图像等 LIBERO 专用查询。

导入顺序是启动正确性的组成部分。必须先设置 MuJoCo 的 EGL 环境变量，再导入任何
可能传递触发 MuJoCo 的 LIBERO/robosuite 模块；带 ``--cuda-device`` 启动时，还要
先完成 EGL 设备映射与 torch 当前设备设置，最后才在 ``make_env`` 内延迟导入
``LiberoEnv``。因此不要把底部的运行时导入提升到模块顶部。
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import OmegaConf

from rpent.utils.config import (
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

# MuJoCo 首次导入时会读取后端选择；setdefault 既为本服务选择无头 EGL，又尊重
# 启动者显式提供的值。此块必须位于任何可能触发 MuJoCo 的 LIBERO/robosuite
# 运行时导入之前。断言把错误导入顺序变成即时、明确的启动失败，避免后续创建环境
# 时才出现难以定位的 OpenGL 上下文问题。
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
assert "mujoco" not in sys.modules, \
    "mujoco must not be imported before MUJOCO_GL/PYOPENGL_PLATFORM are set"

logger = get_logger("env_server")

# 让外部 RLinf checkout 可被导入：优先使用显式配置，否则回退到与 RPent 并列的
# ``rlinf`` 目录。ROBOT_PLATFORM 同样只补默认值，不覆盖调用者已有配置。
RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# 这里只为静态类型检查器提供名字，不会在正常运行时导入 torch 或 LiberoEnv。
# 真正的导入留在函数调用处：main() 可先根据 --cuda-device 配置 EGL/CUDA，随后
# make_env() 才加载会传递导入 torch、robosuite 和 MuJoCo 的 LiberoEnv。
if TYPE_CHECKING:
    import torch  # noqa: F401  (referenced at runtime in _to_numpy_tree)
    from rlinf.envs.libero.libero_env import LiberoEnv


# =============================================================================
# env_server 中文导读：进程边界、数据形状与生命周期
# =============================================================================
#
# 一、为什么环境必须放在独立进程
# ----------------
# LIBERO 环境会加载 robosuite、MuJoCo、OpenGL/EGL、torch 以及 RLinf 环境封装。这些
# 组件初始化重、持有原生资源，而且 MuJoCo 渲染后端和 CUDA/EGL 设备必须在首次导入
# 前确定。把它们放入 env_server 子进程可以：
#
# - 让 Planner 主进程不直接持有 MuJoCo / EGL 上下文；
# - 让每个 TaskRun 拥有独立环境，任务结束时可连同所有原生资源一起回收；
# - 通过 `--parent-watch` 在父进程 stdin pipe 断开后自动关闭，避免孤儿模拟器；
# - 使用同一业务协议连接本地 daemon 或用户提供的外部 endpoint。
#
# 本文件的顶层导入顺序因此属于功能逻辑，而不只是代码风格：必须先设置 MUJOCO_GL
# 和 PYOPENGL_PLATFORM，再让任何传递依赖导入 mujoco。指定 `--cuda-device` 时，
# main() 还必须先完成 EGL 映射和 torch 当前设备选择，最后才调用 make_env() 延迟
# 导入 LiberoEnv。把这些 import 移到文件顶部可能使渲染后端永久选错。
#
# 二、从调用方到模拟器的完整链路
# ----------------
#
#   LiberoEnvClient
#       │ `env.reset` / `env.step` / `env.chunk_step` / 查询 RPC
#       ▼
#   HttpRpcClient 或 SocketRpcClient
#       │ method + args + kwargs
#       ▼
#   RpcFacade.serve()
#       │ 内建 healthz/shutdown；互斥锁串行化业务 dispatch
#       ▼
#   LiberoEnvFacade._dispatch()
#       │ 仅允许 `env.*` 命名空间
#       ▼
#   本类同名方法
#       │ 单环境形状 <-> RLinf batch-first 形状
#       ▼
#   LiberoEnv(num_envs=1) -> robosuite / MuJoCo
#
# HTTP 与 socket 只影响 wire 编码，不改变以上业务语义。HTTP 将 NumPy ndarray 编码为
# `__ndarray__ + dtype + shape + base64(raw bytes)` 的 JSON 标记对象；这里的 base64
# 内容是数组原始内存，不是 PNG/JPEG。socket 使用长度前缀 pickle，可直接携带 ndarray，
# 但只适用于可信本地进程。两者都使用 `ok/result` 或 `ok/error/traceback` 响应 envelope。
# 服务端异常由传输 handler 捕获并带 traceback 返回，客户端统一转换成 RpcError。
#
# 三、单环境与 batch-first 形状适配
# ----------------
# 调用方只看到普通单环境接口，而 RLinf LiberoEnv 即使 num_envs=1 仍保留环境 batch 维：
#
#   客户端 action              [action_dim]
#       -> _expand_action       [1, action_dim]
#
#   客户端 action chunk        [chunk_size, action_dim]
#       -> _expand_chunk        [1, chunk_size, action_dim]
#
#   底层 observation value     [1, ...]
#       -> _strip/_strip_obs    [...]
#
#   底层 chunk term/trunc       [1, chunk_size]
#       -> _strip               [chunk_size]
#
# `_to_numpy_tree()` 在 RPC 边界前递归执行 tensor.detach().cpu().numpy()。因此 GPU
# tensor、autograd 历史和设备所有权都不会越过进程边界；客户端也无需安装或导入
# torch。它只负责“类型可传输化”，环境 batch 维由 `_strip*` 单独处理，二者不要混淆。
#
# 四、episode 状态机与双层保护
# ----------------
# Facade 的 `_done` 是粘滞 episode 标志：
#
#   reset 成功 -> _done=False
#       │
#       ├─ step/chunk 无结束信号 -> 仍为 False
#       └─ 任一 terminated/truncated 为真 -> _done=True
#                                                 │
#                          后续 step/chunk 在调用底层前被断言拒绝
#
# chunk 的信号可能是数组，`_record_done()` 使用 `np.asarray(...).any()`，所以块中任意
# 时刻结束都会封闭 episode。只有 reset 成功后才清零。客户端还维护自己的 terminated /
# truncated 粘滞标志，通常能在网络请求前更早拦截；服务端保护则覆盖其他客户端、状态
# 不同步和直接 RPC 调用。render/raw_obs/camera metadata 等只读查询不受 `_done` 限制，
# 因为它们不推进模拟器。
#
# 五、reset id 与可复现实验
# ----------------
# CLI 的 task 是 suite 内任务编号，RLinf 的 `specific_reset_id` 却位于整个 suite 的连续
# 初始状态空间。make_env() 先累计前序任务状态数，再以 `seed % trials` 选择当前任务
# 的局部 trial，从而把 `(suite, task, seed)` 稳定映射到一个全局 reset id。配置同时
# 关闭 auto_reset 并启用 fixed/ordered reset，确保 RPC reset 不会悄悄切换初始场景。
#
# 六、服务身份与请求串行化
# ----------------
# facade metadata 保存 suite、task、seed 和 max_episode_steps。客户端构造时先调用
# `env.get_env_meta` 严格比较，确认端口没有连到残留的旧任务，然后才 reset；这是防止
# “RPC 可用但实验语义错误”的关键检查。
#
# HTTP/socket server 可以并发接收连接，但 RpcFacade 用一把业务锁串行执行 `_dispatch`。
# 这保证 reset、step、chunk_step 和相机查询不会同时触碰非线程安全的 MuJoCo 环境。
# healthz 无需业务锁；shutdown 会等待正在执行的业务调用离开临界区后再关闭服务。
# 本类没有单独 close RPC：正常 shutdown、父进程死亡或 ProcessDaemon.stop() 会结束
# 整个进程，由操作系统和底层析构路径统一回收 MuJoCo/EGL/CUDA 资源。


# 七、配置字段逐项说明
# ----------------
# `build_env_cfg()` 生成的 OmegaConf 会直接交给 RLinf LiberoEnv。字段按职责可分为：
#
# 【环境身份与规模】
#
# - env_type="libero"：让 RLinf 选择 LIBERO 环境实现；
# - task_suite_name：benchmark suite，例如 libero_spatial；
# - group_size=1：每组只有一个环境，与本服务固定 num_envs=1 对齐；
# - seed：环境随机种子；具体初始状态还由 specific_reset_id 固定。
#
# 【episode 与结束条件】
#
# - auto_reset=False：底层发出结束信号后不自动切换 episode，必须显式 RPC reset；
# - ignore_terminations=False：保留任务成功等 termination 信号；
# - max_steps_per_rollout_epoch：RLinf rollout 层的最大步数；
# - max_episode_steps：环境层的 episode 最大步数；
# - init_params.horizon：传给 robosuite/LIBERO 的 horizon；
# - is_eval=True：按评估而非训练方式构造环境。
#
# 三个最大步数字段使用同一个 CLI 值，避免某一层提前截断而另一层仍认为 episode
# 可继续。服务端仍同时检查底层返回的 terminated 和 truncated；达到时间上限通常会
# 通过 truncated 体现，任务成功等语义通常由 terminated 表达，具体值由底层环境决定。
#
# 【奖励】
#
# - use_rel_reward=False：不切换到相对奖励模式；
# - use_step_penalty=False：不额外施加逐步惩罚；
# - reward_coef=1.0：不缩放底层奖励。
#
# 本 Facade 不解释、重算或累计奖励，只移除 num_envs=1 的环境维后原样返回客户端。
#
# 【reset 与机器人】
#
# - reset_gripper_open=True：reset 后使用张开的夹爪初态；
# - use_fixed_reset_state_ids=True：使用指定 reset state id；
# - use_ordered_reset_state_ids=True：按确定顺序读取指定状态；
# - specific_reset_id：`make_env()` 从 task/seed 计算出的 suite 全局状态编号；
# - init_params.robots：仅当 LIBERO_ROBOT_BASE 环境变量存在时写入，值包装为单元素
#   列表；否则整个键都不生成，让底层采用默认机器人，而不是传入 None。
#
# 【相机与视频】
#
# - camera_heights/camera_widths=256：底层常规 observation 相机分辨率；
# - camera_depths=True：常规观测同时生成深度；
# - video_cfg.save_video=True：允许底层保存动作/episode 视频；
# - video_cfg.info_on_video=True：允许把信息叠加到视频；
# - video_base_dir=/tmp/primitive_videos：底层临时视频目录，不等同于 RPent TaskRun
#   最终 artifact 目录，上层工具可再搬运或登记产物。
#
# `render_camera(height, width)` 可请求不同于常规 256x256 观测的即时分辨率；因此相机
# metadata 也接受 height/width，调用方应使用与目标图像完全一致的尺寸查询标定数据。
#
# 八、Facade 公开 RPC 方法表
# ----------------
# RpcFacade 自带 `healthz` 和 `shutdown`；LiberoEnvFacade 只负责 `env.*` 业务方法：
#
#   RPC                         输入                              返回
#   env.get_env_meta            无                                dict
#   env.reset                   无                                (obs, info)
#   env.step                    action[action_dim]                五元组
#   env.chunk_step              actions[T, action_dim],           五元组；obs 或 list[obs]
#                              return_all_frames: bool
#   env.raw_obs                 无                                当前 raw obs dict
#   env.render_camera           camera_name, H, W, depth          ndarray/底层图像结构
#   env.get_camera_meta         camera_name, H, W                 dict | None
#   env.get_task_language       无                                str | None
#   env.cached_image            无                                ndarray | None
#
# `env.reset`、`env.step` 和 `env.chunk_step` 会改变环境状态；其余 `env.*` 方法均为
# 查询。render_camera 虽然可能触发昂贵的 EGL 渲染，但不增加 episode step。raw_obs
# 读取 LiberoEnv 保存的 current_raw_obs；cached_image 读取 `_cached_full_image` 私有
# 缓存，两者都不保证在首次 reset 之前已有有效值，不过客户端构造时会自动 reset。
#
# `_dispatch()` 根据前缀截出属性名并调用 Facade 自身，而不是把任意方法直接转发给
# `self._env`。例如 `env.step` 命中本类的 step 适配层，客户端不能借 RPC 任意访问
# LiberoEnv 内部属性。非 `env.*` 名称由本类拒绝；healthz/shutdown 已在 RpcFacade
# 外层提前截获，不会进入这里。
#
# 九、reset / step / chunk_step 精确数据流程
# ----------------
# 【reset】
#
#   self._env.reset()
#       -> obs_batch, info
#       -> _to_numpy_tree(obs_batch)       # tensor/GPU -> NumPy/CPU
#       -> _strip_obs(...)                 # 每个观测值 [1,...] -> [...]
#       -> _done=False                     # 仅在前述操作成功后清零
#       -> 返回 obs, numpy_info
#
# 如果底层 reset 或观测转换抛错，`_done=False` 这一赋值不会执行。异常经统一 RPC error
# envelope 返回，客户端的 reset 也只在 RPC 成功后清除本地结束标志，两端保持一致。
#
# 【单步 step】
#
#   检查 _done=False
#       -> np.asarray(action)[None]         # [A] -> [1,A]
#       -> self._env.step(...)
#       -> obs 转 CPU NumPy并逐键去环境维
#       -> reward/term/trunc 转 NumPy并取索引 0
#       -> _record_done(term, trunc)
#       -> 返回 obs, reward, term, trunc, info
#
# info 只递归转换，不调用 `_strip`，因为它的嵌套结构由 RLinf 定义，不应假设顶层一定
# 是环境 batch。reward、term、trunc 明确遵循 batch-first 契约，所以安全取唯一索引。
#
# 【动作块 chunk_step】
#
#   检查 _done=False
#       -> actions[T,A][None]               # [T,A] -> [1,T,A]
#       -> self._env.chunk_step(...)
#       -> 对 obs_list 中每一时刻的观测逐键去环境维
#       -> reward/term/trunc 去掉首个环境维
#       -> term/trunc 中任一元素为真即置 _done=True
#       -> return_all_frames ? obs_list : obs_list[-1]
#
# Facade 假设底层 chunk_step 至少返回一帧，因为 `return_all_frames=False` 会访问
# `obs_list[-1]`。动作块长度与动作维度的合法性由底层 LiberoEnv 校验；本适配层只增加
# 环境维，不静默 reshape、裁剪或填充错误动作。
#
# 十、转换函数的边界与注意事项
# ----------------
# `_to_numpy_tree()` 递归处理 dict/list/tuple，并保持这些容器的类型；非 torch tensor
# 对象原样返回。它不是通用 JSON 编码器：真正的 ndarray JSON/base64 编码发生在
# HttpRpcServer；socket 则由 pickle 处理。分层顺序为：
#
#   GPU tensor -> CPU ndarray -> HTTP JSON 标记 / socket pickle -> 客户端 ndarray
#
# tensor 的 `.detach()` 表明该 RPC 是推理/环境边界，不传播梯度；`.cpu()` 可能引发
# GPU 同步和数据复制，因此大图像调用的耗时包含这部分成本。NumPy 输入不会额外复制，
# 除非后续 HTTP 解码端为可写数组建立副本。
#
# `_strip(v)` 只做 `v[0]`，不会判断形状，也不会递归。该简单约定依赖两个构造不变量：
# LiberoEnv 使用 num_envs=1，且被 strip 的值确实含环境维。若底层 RLinf 接口改变返回
# 形状，应在这里显式适配，而不是让错误索引产生表面合法但语义错误的数据。
#
# 十一、CLI 参数与进程生命周期
# ----------------
#
# - `--transport`：http 或 socket，默认 http；
# - `--host` / `--port`：监听地址；port=0 让操作系统选择端口；
# - `--suite` / `--task` / `--seed`：共同决定任务和 reset 状态；
# - `--max-episode-steps`：同步写入三层 horizon 配置，并进入 metadata；
# - `--cuda-device`：物理 CUDA ordinal，用于 EGL 映射和 torch 当前设备；
# - `--parent-watch`：监听 stdin EOF；父 ProcessDaemon 退出时触发服务 shutdown。
#
# 普通 RPent 运行不会依赖 port=0 后再解析日志，而是在父进程先选一个本地端口并显式
# 传入。服务构造顺序是：解析参数 -> 可选清除 CUDA_VISIBLE_DEVICES -> 配置 EGL ->
# 导入/设置 torch -> 延迟构造 LiberoEnv -> 构造 metadata Facade -> 开始 serve。
# `serve()` 只有在环境构造成功后才绑定业务服务，因此 healthz ready 同时意味着环境
# 对象已成功创建，但不代表某次 reset/step 永远不会因场景数据或运行时资源失败。
#
# 十二、错误传播和可恢复边界
# ----------------
# Facade 方法不吞掉环境异常。`_dispatch()` 记录 warning 后重新抛出，传输层返回错误
# 文本和服务端 traceback；LiberoEnvClient 最终收到 RpcError。典型错误包括：
#
# - 环境资产、benchmark 或 reset state 加载失败；
# - EGL/CUDA 设备初始化或渲染失败；
# - 动作 shape/数值不被底层接受；
# - episode 结束后再次 step/chunk；
# - 未知 RPC 名称或参数不匹配；
# - 返回对象无法转换或序列化。
#
# RPC 超时与明确的远端错误不同：超时时客户端无法仅凭响应确认远端方法是否已经执行。
# 对会推进环境的 step/chunk 不应盲目重试；RPent 的安全恢复方式通常是终止该 TaskRun
# 拥有的 env_server，并在新进程中重新构造环境。查询类 RPC 可由上层按需求重试。
#
# `assert not self._done` 是运行期协议保护，但 Python `-O` 会移除 assert；当前 RPent
# 服务按普通解释模式启动。客户端还有一层粘滞结束检查，底层 LiberoEnv 也可能拒绝
# 非法推进。这里的双层检查用于尽早暴露调用错误，不替代上层对终止状态的正常处理。


# ---------------------------------------------------------------------------
# 环境配置与构造
# ---------------------------------------------------------------------------


def build_env_cfg(
    *,
    task_suite_name: str = "libero_spatial",
    specific_reset_id: int = 0,
    seed: int = 0,
    max_episode_steps: int = 10000,
) -> Any:
    """构造单环境、固定 reset 状态的 LIBERO/RLinf 配置。

    Args:
        task_suite_name: LIBERO benchmark suite 名称。
        specific_reset_id: suite 级 reset 状态编号，而不是单个 task 内的局部编号。
            ``make_env`` 负责把 task/seed 映射到这个全局编号。
        seed: 传给环境配置的随机种子。
        max_episode_steps: rollout 与底层环境共同使用的最大 episode 步数。

    配置关闭自动 reset，并同时启用 fixed 与 ordered reset state id，使服务进程每次
    ``reset`` 都围绕调用方选择的初始化状态工作。``group_size=1`` 与后续
    ``num_envs=1`` 相配合，是 facade 可以安全增删单个 batch 维的前提。
    """
    cfg = OmegaConf.create(
        {
            "env_type": "libero",
            "task_suite_name": task_suite_name,
            "auto_reset": False,
            "ignore_terminations": False,
            "max_steps_per_rollout_epoch": max_episode_steps,
            "max_episode_steps": max_episode_steps,
            "use_rel_reward": False,
            "use_step_penalty": False,
            "reward_coef": 1.0,
            "reset_gripper_open": True,
            "is_eval": True,
            "seed": seed,
            "group_size": 1,
            # 固定且有序地选择 suite 级 reset id，避免每次 RPC reset 漂移到另一个
            # 初始状态；specific_reset_id 的计算见 make_env。
            "use_fixed_reset_state_ids": True,
            "use_ordered_reset_state_ids": True,
            "specific_reset_id": specific_reset_id,
            "video_cfg": {
                "save_video": True,
                "info_on_video": True,
                "video_base_dir": "/tmp/primitive_videos",
            },
            "init_params": {
                "camera_heights": 256,
                "camera_widths": 256,
                # 同时渲染深度，调用方才能结合相机标定把像素反投影到世界坐标。
                "camera_depths": True,
                "horizon": max_episode_steps,
                # 若进程环境指定机器人底座，则以列表形式传给 LIBERO；未指定时
                # 完全省略该键，沿用底层默认机器人配置。
                **({"robots": [os.environ["LIBERO_ROBOT_BASE"]]}
                   if os.environ.get("LIBERO_ROBOT_BASE") else {}),
            },
        }
    )
    return cfg


def make_env(task_id: int, seed: int, suite_name: str = "libero_spatial",
             max_episode_steps: int = 10000) -> LiberoEnv:
    """创建固定到指定 task 与 seed 初始化状态的单环境 ``LiberoEnv``。

    ``task_id`` 是 suite 内任务编号，而 RLinf 配置使用跨任务连续编号的 reset id。
    本函数先累加此前所有任务的初始化状态数量得到当前任务的起始偏移，再用
    ``seed % trials`` 在当前任务可用状态中稳定选取一个，二者相加得到传给
    ``build_env_cfg`` 的 suite 级 ``specific_reset_id``。

    LiberoEnv 与 benchmark 均在函数内延迟导入。这样 ``main`` 有机会先处理
    ``--cuda-device``，避免 torch、robosuite 或 MuJoCo 在 EGL/CUDA 设备配置完成前
    被传递导入。最终显式使用 ``num_envs=1``，与 facade 的维度转换约定一致。
    """
    # 不要把这两个导入移动到模块顶部；设备选择必须先于它们可能触发的重依赖导入。
    from rlinf.envs.libero.libero_env import LiberoEnv
    from rlinf.envs.libero.utils import benchmark as _bench_mod

    suite = _bench_mod.get_benchmark(suite_name)()
    # 把 task 内局部 trial 编号映射到整个 suite 的连续 reset id 空间。
    first_id = sum(len(suite.get_task_init_states(t)) for t in range(task_id))
    trials = len(suite.get_task_init_states(task_id))
    rid = first_id + (seed % trials)
    cfg = build_env_cfg(
        task_suite_name=suite_name,
        specific_reset_id=rid,
        seed=seed,
        max_episode_steps=max_episode_steps,
    )
    return LiberoEnv(cfg=cfg, num_envs=1, seed_offset=0,
                     total_num_processes=1, worker_info=None)


# ---------------------------------------------------------------------------
# 实现 robots.libero.env_client 协议的 RPC facade
# ---------------------------------------------------------------------------


def _to_numpy_tree(x):
    """递归建立可安全跨 RPC wire 传输的 CPU numpy/Python 对象树。

    torch tensor 可能位于 GPU、带有 autograd 历史，不能直接作为客户端协议对象；
    因此先 ``detach``，再移到 CPU 并转成 numpy。字典、列表和元组会保留容器种类并
    递归处理其值，其他对象原样返回。torch 在此处局部导入，以维持模块级 EGL/CUDA
    初始化顺序，也让只导入本模块而不调用环境接口的路径不提前加载 torch。
    """
    import torch

    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, dict):
        return {k: _to_numpy_tree(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_to_numpy_tree(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_to_numpy_tree(v) for v in x)
    return x


class LiberoEnvFacade(RpcFacade):
    """把 batch-first ``LiberoEnv`` 适配为客户端使用的单环境 RPC 协议。

    RPC 名称以 ``env.`` 为命名空间，由 ``_dispatch`` 去掉前缀后调用同名方法。
    底层环境始终以 ``num_envs=1`` 构造，因此入站单动作/动作块需要增加环境维，
    出站 batch 值则固定取 ``_env_idx`` 对应项。所有可能包含 tensor 的返回结构在
    穿过 agent/env_server wire 前转换为 CPU numpy，使不导入 torch 的客户端也能
    反序列化。

    facade 自身用 ``_done`` 粘滞记录 terminated/truncated。一旦单步或动作块任一
    信号为真，后续推进会被阻止，直到 ``reset`` 成功开始新 episode。
    """

    def __init__(self, env: LiberoEnv, *, meta: dict):
        """绑定单环境实例，并保存供客户端核验的启动 metadata。

        ``meta`` 会复制后保存，避免调用方随后修改原字典影响服务端身份信息；查询时
        还会再次返回副本。客户端在构造阶段把它与期望值严格比较，从而拒绝任务、
        seed 或最大步数不一致的旧服务。
        """
        super().__init__()
        self._env = env
        # 底层只有一个环境，但集中保留索引可让所有 strip/查询路径使用同一约定。
        self._env_idx = 0
        self._done = False
        # 标识此服务启动时绑定的任务与 seed；客户端会与自己的期望配置比较，
        # 从而拒绝连接残留或配置错误的 env_server。
        self._meta = dict(meta)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """把 ``env.<name>`` RPC 请求分发到 facade 的同名方法。

        只有 ``env.`` 命名空间会进入属性查找；调用失败时记录方法名和异常后重新抛出
        同一个异常，让 RPC 层按既有错误协议反馈客户端。其他命名空间直接报告未知
        RPC 方法，不会落到环境对象上。
        """
        if method.startswith("env."):
            attr = method[len("env."):]
            try:
                return getattr(self, attr)(*args, **kwargs)
            except Exception as e:
                logger.warning("run method %s failed: %s", method, e)
                raise e
        raise ValueError(f"unknown RPC method: {method!r}")

    # ---- 单环境 batch 维转换辅助方法 ----

    def _strip(self, v):
        """从 batch 值中取出唯一环境对应的元素。

        ``v`` 可以是形状 ``[B, ...]`` 的 numpy 数组、长度为 B 的列表（例如任务
        描述），也可以是可选图像使用的 ``None``。因为 LiberoEnv 以
        ``num_envs=1`` 运行，``self._env_idx`` 始终指向唯一有效项。这里只移除环境
        维，不递归转换类型；调用点应先经过 ``_to_numpy_tree``。
        """
        if v is None:
            return None
        return v[self._env_idx]

    def _strip_obs(self, obs: dict) -> dict:
        """逐键移除 LIBERO 观测字典每个值的前导环境维。

        字典结构与观测键名保持不变；每个值都遵循 ``_strip`` 的单环境/``None``
        约定。
        """
        return {k: self._strip(v) for k, v in obs.items()}

    def _expand_action(self, action) -> np.ndarray:
        """为形状 ``[action_dim]`` 的单环境动作增加前导环境维。

        ``np.asarray`` 建立明确的 numpy wire/环境边界，新增维后形状为
        ``[1, action_dim]``，符合 batch-first ``LiberoEnv.step`` 的输入约定。
        """
        return np.asarray(action)[None]

    def _expand_chunk(self, actions) -> np.ndarray:
        """为单环境动作块增加前导环境维。

        客户端传入形状 ``[chunk_size, action_dim]``，转换后得到
        ``[1, chunk_size, action_dim]``，供 ``LiberoEnv.chunk_step`` 使用。
        """
        return np.asarray(actions)[None]

    def _record_done(self, *signals: Any) -> None:
        """把任意 terminated/truncated 信号粘滞 OR 到 ``self._done``。

        信号可以是单步标量，也可以是动作块数组。任一信号的任一元素为真即记录
        episode 已结束并提前返回；只有 ``reset`` 才会把状态清零。
        """
        for s in signals:
            if np.asarray(s).any():
                self._done = True
                return

    # ---- 类 Gym 的单环境 RPC 接口 ----

    def reset(self):
        """重置底层环境，移除观测环境维并清除 facade 的结束状态。

        底层 ``reset`` 返回 batch 观测与 info；观测先整体转为 numpy，再逐键取唯一
        环境。info 仅执行递归 numpy 转换并保持原有结构。``_done`` 在底层调用与
        观测转换成功后清零。
        """
        obs, info = self._env.reset()
        obs = self._strip_obs(_to_numpy_tree(obs))
        self._done = False
        return obs, _to_numpy_tree(info)

    def step(self, action):
        """执行一个单环境动作并返回标准五元组。

        发送给底层前把 ``[action_dim]`` 扩展为 ``[1, action_dim]``。返回时观测、
        reward、terminated 与 truncated 去掉唯一环境维，info 只做递归 numpy
        转换；随后把两类结束信号累计到 ``_done``。episode 已结束时会在调用底层
        环境之前由断言阻止重复 step。
        """
        assert not self._done, "step called after episode done"
        obs, rew, term, trunc, info = self._env.step(self._expand_action(action))
        obs = self._strip_obs(_to_numpy_tree(obs))
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        self._record_done(term, trunc)
        return (
            obs,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            _to_numpy_tree(info),
        )

    def chunk_step(self, actions, *, return_all_frames: bool = False):
        """在一次 RPC 中执行完整动作块，并返回固定的五元组协议。

        入站 ``actions`` 形状为 ``[chunk_size, action_dim]``；补环境维后交给底层
        ``chunk_step``。底层逐步观测列表中的每个字典都转为 numpy 并移除环境维。
        ``return_all_frames=True`` 时第一项保留完整 ``list[Obs]``，否则只选最终观测，
        其余四项的位置和结构不变。

        去掉环境维后，terminated/truncated 保留 ``[chunk_size]``，既让客户端可见
        每个 chunk step 的信号，也让 ``_record_done`` 对整块执行 ``any``。reward
        去掉环境维，info 则只递归转换而不重塑。episode 已结束时不会再次调用底层。
        """
        assert not self._done, "chunk_step called after episode done"
        obs_list, rew, term, trunc, info = self._env.chunk_step(
            self._expand_chunk(actions)
        )
        obs_list = [self._strip_obs(_to_numpy_tree(o)) for o in obs_list]
        term = self._strip(_to_numpy_tree(term))
        trunc = self._strip(_to_numpy_tree(trunc))
        self._record_done(term, trunc)
        obs_field = obs_list if return_all_frames else obs_list[-1]
        return (
            obs_field,
            self._strip(_to_numpy_tree(rew)),
            term,
            trunc,
            _to_numpy_tree(info),
        )

    def raw_obs(self) -> dict:
        """返回唯一环境当前保存的原始观测。

        先按 ``_env_idx`` 从底层 ``current_raw_obs`` 选择单环境值，再递归转换其中的
        tensor；该查询不推进环境，也不改变 ``_done``。
        """
        return _to_numpy_tree(self._env.current_raw_obs[self._env_idx])

    def get_env_meta(self) -> dict:
        """返回服务启动 metadata 的副本，供客户端进行防错配校验。"""
        return dict(self._meta)

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        """调用底层相机接口即时渲染指定尺寸的 RGB 或深度结果。

        相机名、分辨率与 ``depth`` 原样传入 LIBERO；返回结构统一经过
        ``_to_numpy_tree``，确保 GPU tensor 不越过 RPC wire。此查询不改变 episode
        推进状态。
        """
        return _to_numpy_tree(
            self._env.render_camera(
                camera_name=camera_name,
                height=height,
                width=width,
                depth=depth,
            )
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        """返回指定相机及目标分辨率对应的标定 metadata。

        底层可返回 ``None``；若包含 tensor 或嵌套容器，则在返回客户端前递归转换为
        CPU numpy/Python 对象。该接口与 ``render_camera`` 分离，使调用方可独立取得
        像素反投影所需信息。
        """
        return _to_numpy_tree(
            self._env.get_camera_meta(
                camera_name=camera_name, height=height, width=width
            )
        )

    def get_task_language(self) -> str | None:
        """返回唯一环境对应的任务自然语言描述。"""
        return self._env.task_descriptions[self._env_idx]

    def cached_image(self) -> np.ndarray | None:
        """读取底层环境最近缓存的完整图像，而不触发新的渲染。

        私有缓存尚不存在或值为空时返回 ``None``。若缓存提供 ``cpu`` 方法则先移到
        CPU 再转 numpy，否则通过 ``np.asarray`` 建立稳定的 wire 表示。
        """
        cached = getattr(self._env, "_cached_full_image", None)
        if cached is None:
            return None
        return cached.cpu().numpy() if hasattr(cached, "cpu") else np.asarray(cached)


# ---------------------------------------------------------------------------
# 服务进程入口
# ---------------------------------------------------------------------------


def main():
    """解析启动参数、按正确顺序初始化 GPU/EGL，并启动 RPC 服务。

    启动流程依次为：解析 transport 与环境身份参数；可选地固定 MuJoCo EGL 和 torch
    使用的物理 GPU；延迟创建单环境；构造包含 suite/task/seed/max steps 的 facade
    metadata；最后调用 ``serve`` 监听请求。``parent_watch`` 原样交给 RPC 层，使该
    独立服务可在父进程 stdin 管道断开时退出。
    """
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", type=str, default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--suite", type=str, default="libero_spatial")
    p.add_argument("--task", type=int, default=9)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max-episode-steps", type=int, default=10000)
    p.add_argument("--parent-watch", action="store_true",
                   help="watch parent process via stdin pipe and exit when it dies")
    p.add_argument("--cuda-device", type=int, default=None,
                   help="GPU device to pin MuJoCo EGL rendering and the torch "
                        "default device to (physical CUDA ordinal).")
    args = p.parse_args()

    if args.cuda_device is not None:
        # 这里刻意不设置 CUDA_VISIBLE_DEVICES。libero 传递导入的 robosuite 会在导入
        # 时用子串方式断言 MUJOCO_EGL_DEVICE_ID 出现在 CUDA_VISIBLE_DEVICES 中；
        # 该假设把 EGL 序号等同于 CUDA 物理序号，在多 GPU 且两套枚举顺序不同时会
        # 错误崩溃。此断言仅在 CUDA_VISIBLE_DEVICES != "" 时启用，因此清除已有值
        # 可让当前进程及继承环境的 multiprocessing render worker 跳过错误检查。
        # 随后分别固定两个后端，且顺序不能后移到 LiberoEnv 导入之后：
        #   - MuJoCo 渲染设备 <- configure_egl_device 设置的 MUJOCO_EGL_DEVICE_ID
        #   - torch 当前设备  <- torch.cuda.set_device(N)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is set; clearing it and pinning via "
                "MUJOCO_EGL_DEVICE_ID + torch.cuda.set_device(--cuda-device=%s) "
                "instead (robosuite's CVD assertion is incompatible with EGL<->CUDA mapping)",
                prev, args.cuda_device,
            )
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        # 先配置 EGL，再导入并设置 torch；make_env 中的 LiberoEnv 导入必须发生在二者之后。
        from rpent.utils.egl import configure_egl_device
        configure_egl_device(args.cuda_device)
        import torch
        torch.cuda.set_device(args.cuda_device)

    # 环境构造会触发此前刻意延迟的重依赖导入，此时所有可选设备设置已经完成。
    raw_env = make_env(args.task, args.seed, suite_name=args.suite,
                       max_episode_steps=args.max_episode_steps)
    # metadata 描述会影响轨迹语义的启动参数；客户端连接后会在 reset 前严格核对。
    facade = LiberoEnvFacade(
        raw_env,
        meta={
            "suite": args.suite,
            "task": args.task,
            "seed": args.seed,
            "max_episode_steps": args.max_episode_steps,
        },
    )
    # transport/监听地址与父进程存活监控全部交给通用 RpcFacade 服务循环。
    facade.serve(transport=args.transport, host=args.host, port=args.port,
                 parent_watch=args.parent_watch)


if __name__ == "__main__":
    main()
