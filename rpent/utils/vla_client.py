"""Pi0.5 VLA RPC 服务的轻量客户端。

服务生命周期由调用方负责：构造本类之前，必须先启动
``robots/libero/vla_server.py``，或连接实现相同 ``predict`` / ``healthz``
协议的远端服务。客户端只负责三件事：

1. 把单环境 NumPy 观测转换成 JSON-safe 的 RPC 负载；
2. 调用与传输无关的 :class:`RpcClient`（HTTP 或 socket）；
3. 把服务端的批量动作 ``[B=1, chunk, action_dim]`` 还原成调用方使用的
   ``[chunk, action_dim]``。

Wire schema（与 ``vla_server.py`` 对应）：

    call("predict", kwargs={
        "instruction": "<task_descriptions>",
        "images": {
            "main":  {"format": "png", "data": "<base64>"},
            "wrist": {"format": "png", "data": "<base64>"},  # optional
            "extra": {"format": "png", "data": "<base64>"},  # optional
        },
        "state": [[s0..sN]],           # shape [B, state_dim]
        "mode":  "eval",
    })
    -> {"actions": [[[a0..a6], ...]],
        "shape": [B, chunk, action_dim], "dtype": "float32"}
"""
from __future__ import annotations

import base64
import io
from typing import Any

import numpy as np

from rpent.utils.rpc import RpcClient


def _png_b64(img: np.ndarray) -> str:
    """把 RGB NumPy 数组编码成可放入 JSON 的 PNG/base64 字符串。

    使用 PNG 而不是直接展开像素列表，可以明显减小 HTTP/socket 消息体，同时
    保持无损图像。服务端 ``_decode_image_block`` 执行严格的逆变换。
    """
    # 延迟导入使不进行 VLA 推理的进程不必加载图像编解码依赖。
    import imageio.v2 as imageio

    arr = np.asarray(img)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8)

    # 在内存中完成编码，不产生临时图片文件。
    buf = io.BytesIO()
    imageio.imwrite(buf, arr, format="png")
    return base64.b64encode(buf.getvalue()).decode("ascii")


class VLAClient:
    """通过任意 ``RpcClient`` 传输调用远端 Pi0.5。

    当前唯一业务调用方是 ``LiberoPrimitives``。本类刻意保持与本地模型相似的
    ``predict_action_batch`` 方法名，使 primitive 不需要知道模型位于独立进程、
    远端机器，还是使用 HTTP/socket 传输。
    """

    def __init__(self, client: RpcClient):
        # 具体传输由 _init_runtime 根据 endpoint 选择：HttpRpcClient 或
        # SocketRpcClient。VLAClient 本身不拥有也不关闭服务进程。
        self._client = client

    def healthz(self, *, timeout_s: float | None = None) -> dict[str, Any]:
        """透传健康检查，主要供 ``wait_for_ready`` 在启动阶段轮询。"""
        return self._client.call("healthz", timeout_s=timeout_s)

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: str = "eval",
        **_kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """发送一个单环境观测并返回 ``float32[chunk, action_dim]``。

        方法名沿用底层模型 API，但 RPent 当前每个 ``LiberoPrimitives`` 只绑定一个
        环境，所以输入故意不带 batch 维；RPC 边界会临时补上 ``B=1``。
        """
        main_images = np.asarray(env_obs["main_images"])
        if main_images.ndim != 3:
            raise ValueError(
                f"main_images expected shape [H,W,3]; got {main_images.shape}"
            )

        # 主相机是必需输入。图像先转 PNG/base64，形成 JSON-safe 字段。
        images: dict[str, Any] = {
            "main": {"format": "png", "data": _png_b64(main_images)},
        }

        # 腕部和额外视角是可选的。只发送非空的 HxWxC 数组；服务端会为缺失视角
        # 填入 None，以满足 OpenPI obs_processor 的固定键约定。
        for src_key, wire_key in (
            ("wrist_images", "wrist"),
            ("extra_view_images", "extra"),
        ):
            view = env_obs.get(src_key)
            if view is None:
                continue
            arr = np.asarray(view)
            if arr.size > 0 and arr.ndim == 3:
                images[wire_key] = {
                    "format": "png",
                    "data": _png_b64(arr),
                }

        states = np.asarray(env_obs["states"]).astype(np.float32)
        if states.ndim != 1:
            raise ValueError(
                f"states must be single-env shape [state_dim]; got {states.shape}"
            )

        # 传输层统一使用 ``call(method, kwargs=...)``。服务端将 instruction、图像和
        # 本体状态重建成 OpenPI 的 env_obs。state 外层列表就是临时 batch 维。
        payload = self._client.call(
            "predict",
            kwargs={
                "instruction": env_obs.get("task_descriptions") or "",
                "images": images,
                # vla_server 的 wire schema 仍要求 [B, state_dim]。
                "state": [states.tolist()],
                "mode": mode,
            },
        )

        # 服务端返回 [B=1, chunk, action_dim]。去掉 B 后，primitive 可直接把整个
        # [chunk, action_dim] 动作块交给 LiberoEnvClient.chunk_step。
        actions = np.asarray(payload["actions"], dtype=np.float32)[0]
        return actions, {}
