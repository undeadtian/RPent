"""LIBERO 单环境 RPC 客户端适配器。

本模块把调用方看到的类 Gym 环境接口包装成 ``RpcClient.call`` 请求；它并不
在本进程中创建 LIBERO 或导入仿真依赖。这里保留 ``raw_obs``、相机与缓存图像等
LIBERO 专用接口，通用的传输、序列化和请求分发逻辑则由 :mod:`rpent.utils.rpc`
提供。

客户端同时维护两类本地防护状态：构造时核对服务端启动 metadata，避免误连到
任务、seed 或步数配置不同的旧服务；执行 ``step``/``chunk_step`` 后以粘滞 OR
方式累计 terminated 与 truncated，避免 episode 已结束后继续发出推进请求。
"""
from __future__ import annotations

from typing import Any

import numpy as np

from rpent.utils.rpc import RpcClient

# 不同 RPC 的耗时差异很大：reset、动作块和高分辨率渲染需要比轻量查询更宽裕的
# 超时。键名与实际发出的 RPC 名称保持一致，便于逐接口审查超时策略。
_TIMEOUT_S = {
    "default": 30.0,
    "env.reset": 120.0,
    "env.step": 60.0,
    "env.chunk_step": 120.0,
    "env.render_camera": 120.0,
}


class LiberoEnvClient:
    """通过 RPC 暴露 LIBERO 环境协议的客户端代理。

    每个公开方法只负责三件事：把本地参数放入对应的 ``env.*`` RPC 调用、选择
    合适的超时，并原样返回服务端给出的协议结构。类本身不转换观测或动作维度；
    单环境 batch 维与 numpy wire 边界均由服务端 facade 处理。

    ``terminated`` 与 ``truncated`` 是 episode 级粘滞标志。只要任一步或动作块中
    任一位置报告结束，相应标志就保持为真，直到下一次成功 ``reset``。
    """

    def __init__(
        self,
        client: RpcClient,
        *,
        expected_meta: dict,
        return_all_frames: bool = False,
    ):
        """绑定 RPC 客户端、校验远端环境身份并立即 reset。

        Args:
            client: 已连接或可发起调用的通用 RPC 客户端。
            expected_meta: 调用方期望的服务端启动信息。必须与
                ``env.get_env_meta`` 返回的字典完全相等，以同时检查键和值，防止
                复用端口时误连到任务、seed 或 horizon 不同的残留服务。
            return_all_frames: ``chunk_step`` 未显式传参时，是否请求动作块内每一步
                的观测；该默认值只影响 RPC 参数，不改变服务端返回协议。

        构造末尾调用 ``reset``，因此实例成功返回时远端环境已经处于新 episode，
        本地结束标志也已清零。metadata 不匹配时会在 reset 前直接拒绝继续通信。
        """
        self._client = client
        self.return_all_frames = return_all_frames
        self.terminated = False
        self.truncated = False

        # 先查询远端启动参数，再做任何会改变环境状态的调用。严格字典相等检查可
        # 让错误连接尽早失败，而不是在轨迹运行后才暴露任务或 reset 状态错配。
        server_meta = self._client.call(
            "env.get_env_meta", timeout_s=_TIMEOUT_S["default"]
        )
        assert server_meta == expected_meta, (
            f"env_meta mismatch: expected={expected_meta!r} "
            f"actual={server_meta!r}. The env_server was launched with "
            "different args than this client expects — kill the stale "
            "env_server and relaunch."
        )
        self.reset()

    def check_done(self, term, trunc) -> None:
        """把本次返回的结束信号累计到 episode 级本地状态。

        信号既可能是标量，也可能是 ``chunk_step`` 返回的一维数组。先转为 numpy
        再做 ``any``，可用同一条路径处理两者；随后使用原位 OR，确保较早步骤出现
        的结束信号不会被后续假值覆盖。terminated 与 truncated 分开累计，以保留
        两种结束原因。
        """
        self.terminated |= bool(np.asarray(term).any())
        self.truncated |= bool(np.asarray(trunc).any())

    def reset(self) -> tuple[dict, Any]:
        """请求远端重置环境，并在调用成功后清除本地结束状态。

        返回值完全沿用远端 ``env.reset`` 的 ``(obs, info)`` 结构。标志清零发生在
        RPC 返回之后，因此超时或远端异常不会把客户端误标为已经成功开始新 episode。
        """
        ret = self._client.call("env.reset", timeout_s=_TIMEOUT_S["env.reset"])
        self.terminated = False
        self.truncated = False
        return ret

    def step(self, action) -> tuple[dict, Any, np.ndarray, Any, Any]:
        """通过一次 RPC 推进一步，并累计返回的结束信号。

        ``action`` 不在客户端改形或转换，服务端负责为单环境动作补 batch 维。返回值
        保持五元组 ``(obs, reward, terminated, truncated, info)``；若本地已观察到
        episode 结束，则在发送 RPC 前由断言阻止继续推进。
        """
        assert not (self.terminated or self.truncated), (
            "env.step called after the episode signaled term/trunc"
        )
        ret = self._client.call(
            "env.step", args=(action,), timeout_s=_TIMEOUT_S["env.step"]
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def chunk_step(self, actions, *, return_all_frames: bool | None = None) -> tuple[Any, Any, Any, Any, Any]:
        """用一次 RPC 执行完整动作块，并累计块内所有结束信号。

        ``actions`` 以单环境形状 ``[chunk_size, action_dim]`` 传给服务端。返回值固定
        为五元组 ``(obs_or_list, reward, terminated, truncated, info)``：当
        ``return_all_frames`` 为真时，第一项是每个 chunk step 对应的观测列表；否则
        只返回最终观测。服务端移除环境维后，terminated/truncated 的形状为
        ``[chunk_size]``，本方法会跨整个数组做 ``any`` 并粘滞累计。

        参数传入 ``None`` 时使用实例级 ``self.return_all_frames``，使调用方既能设置
        常用默认行为，也能对单次动作块覆盖该选择。
        """
        assert not (self.terminated or self.truncated), (
            "env.chunk_step called after the episode signaled term/trunc"
        )
        if return_all_frames is None:
            return_all_frames = self.return_all_frames
        ret = self._client.call(
            "env.chunk_step",
            args=(actions,),
            kwargs={"return_all_frames": return_all_frames},
            timeout_s=_TIMEOUT_S["env.chunk_step"],
        )
        _, _, term, trunc, _ = ret
        self.check_done(term, trunc)
        return ret

    def raw_obs(self) -> dict:
        """获取服务端当前单环境的原始 LIBERO 观测字典。

        这是只读查询，不推进环境，也不修改客户端的 episode 结束标志；返回值已由
        服务端跨 wire 转换为可序列化的 numpy/Python 对象树。
        """
        return self._client.call("env.raw_obs", timeout_s=_TIMEOUT_S["default"])

    def render_camera(
        self,
        camera_name: str = "agentview",
        height: int = 1024,
        width: int = 1024,
        depth: bool = False,
    ):
        """请求服务端按指定相机和分辨率即时渲染图像。

        ``camera_name``、尺寸与 ``depth`` 作为关键字参数原样进入
        ``env.render_camera``。渲染可能触发较慢的 GPU/EGL 工作，因此使用独立的
        长超时；具体 RGB/深度返回形式由底层 LIBERO 相机接口决定。
        """
        return self._client.call(
            "env.render_camera",
            kwargs={
                "camera_name": camera_name,
                "height": height,
                "width": width,
                "depth": depth,
            },
            timeout_s=_TIMEOUT_S["env.render_camera"],
        )

    def get_camera_meta(
        self,
        camera_name: str = "agentview",
        height: int = 256,
        width: int = 256,
    ) -> dict | None:
        """查询指定相机在目标分辨率下的标定 metadata。

        方法仅包装 ``env.get_camera_meta``，不在客户端解释内外参；远端不提供相机
        信息时允许返回 ``None``。
        """
        return self._client.call(
            "env.get_camera_meta",
            kwargs={"camera_name": camera_name, "height": height, "width": width},
            timeout_s=_TIMEOUT_S["default"],
        )

    def get_task_language(self) -> str | None:
        """获取当前任务的自然语言描述，缺失时返回 ``None``。"""
        return self._client.call(
            "env.get_task_language", timeout_s=_TIMEOUT_S["default"]
        )

    def cached_image(self) -> np.ndarray | None:
        """读取服务端底层环境最近缓存的完整图像，不触发重新渲染。

        缓存尚未建立时返回 ``None``；存在时由服务端保证跨 RPC 边界的是 numpy
        数组，而非 GPU tensor。
        """
        return self._client.call(
            "env.cached_image", timeout_s=_TIMEOUT_S["default"]
        )
