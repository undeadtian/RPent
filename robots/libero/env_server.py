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
