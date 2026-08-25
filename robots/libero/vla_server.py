"""Pi0.5 VLA 的独立 RPC 推理服务。

该进程只负责“观测 -> 动作块”推理，不直接推进 LIBERO 环境。完整调用链为：

``LiberoPrimitives._vlm_chunk``
    -> ``VLAClient.predict_action_batch``
    -> RPC ``predict``
    -> ``VLAFacade.predict``
    -> RLinf/OpenPI ``predict_action_batch``
    -> 返回形状为 ``[B, chunk, action_dim]`` 的动作块

主进程随后把去掉 batch 维的动作块交给 ``env.chunk_step`` 执行。把模型放在
独立进程中有三个目的：模型只加载一次；Agent 主进程不直接依赖 RLinf/OpenPI；
以及 VLA 服务可以通过 HTTP/socket 部署到远端 GPU。

RPC wire payload 始终保持 JSON-safe：相机帧以 PNG 字节再经 base64 传输，状态
使用嵌套列表，返回动作则转换为 float32 列表并附带 shape/dtype 元数据。服务启动
时会先按 ``--cuda-device``（如提供）设置 ``CUDA_VISIBLE_DEVICES``，再在 Facade
构造阶段延迟导入 RLinf/OpenPI 并加载模型，使进程内的逻辑 ``cuda:0`` 对应命令行
选择的物理 GPU。
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import sys
import time
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from rpent.utils.config import (
    get_pi05_checkpoint_path,
    get_repo_root,
    get_rlinf_repo_path,
)
from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("vla_server")

# RLinf 通常作为 RPent 旁边的源码仓库存在，而不是安装在当前包中。必须先把其
# 根目录加入 sys.path，后面的延迟 import 才能找到
# ``rlinf.models.embodiment.openpi``。显式配置 RLINF_REPO_PATH 时优先使用配置值。
RPENT_ROOT = get_repo_root()
RLINF_REPO_PATH = get_rlinf_repo_path() or (RPENT_ROOT.parent / "rlinf").resolve()
if str(RLINF_REPO_PATH) not in sys.path:
    sys.path.insert(0, str(RLINF_REPO_PATH))

# RLinf/OpenPI 会读取该变量选择机器人平台相关的观测/动作处理逻辑。setdefault
# 保留了调用方显式设置其他值的能力；正常 LIBERO 启动时这里得到 ``LIBERO``。
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")

# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_model_cfg(model_path: str) -> Any:
    """构造 RLinf ``get_model`` 所需的 OmegaConf 配置。

    这里描述的是推理时模型结构和动作接口，而不是训练任务配置。与本服务数据
    契约直接相关的字段包括：

    - ``action_dim=7``：LIBERO OSC 动作为 3 维平移、3 维旋转和 1 维夹爪；
    - ``action_chunk=5``：一次前向产生一个短动作块，随后由环境批量执行；
    - ``num_images_in_input=2``：策略可使用主相机和腕部相机；
    - ``use_proprio=True``：同时输入机器人本体状态。

    配置分为两层：外层保留 RLinf 通用模型字段，``openpi`` 子表则会覆盖到
    ``OpenPi0Config``，直接控制当前 Pi0.5 推理实现。两层中的重复值应保持一致，
    但在当前 ``get_model`` 路径中，动作裁剪、去噪和值头等 OpenPI 行为以子表为准。

    返回 OmegaConf 而不是普通 dict，是因为 RLinf 的模型工厂按配置对象读取字段。
    """
    return OmegaConf.create(
        {
            # -----------------------------------------------------------------
            # RLinf 通用模型配置层
            # -----------------------------------------------------------------
            # 标识模型后端。通用 RLinf 配置/调度代码可据此识别这是 OpenPI 模型；
            # 本服务已经直接导入 OpenPI get_model，因此不会在这里再次做后端分派。
            "model_type": "openpi",

            # Pi0.5 checkpoint 目录或可由 OpenPI 下载器解析的模型路径。加载器还会
            # 从该目录读取与训练时一致的归一化统计，不能只指向任意权重文件。
            "model_path": model_path,

            # 不在通用层强制指定 fp16/bf16 等精度。当前加载器按 checkpoint 加载，
            # 随后将指定模型参数转为其预期的 bfloat16 表示。
            "precision": None,

            # 通用层保留的动作块配置值。当前 OpenPI 返回多少个连续动作，实际由
            # 下方 openpi.action_chunk 控制；这里保持同值以免其他 RLinf 组件误判。
            "num_action_chunks": 5,

            # LIBERO 环境真正消费的动作维数：XYZ 平移 3 维、旋转 3 维、夹爪 1 维。
            # OpenPI 内部允许使用带 padding 的更宽动作张量，输出变换只保留前 7 维。
            "action_dim": 7,

            # 不加载 LoRA 适配器，直接使用 checkpoint 中的完整模型权重。
            "is_lora": False,

            # LoRA 的低秩维度；仅在 is_lora=True 时有意义，当前值不会参与推理。
            "lora_rank": 32,

            # 声明策略除图像和语言外还使用机器人本体状态。VLA RPC 的 ``state``
            # 字段会被转换成 OpenPI observation/state 输入。
            "use_proprio": True,

            # 通用层的采样步数镜像；当前 OpenPI 去噪循环实际读取子表 num_steps。
            "num_steps": 5,

            # 通用层不请求 critic/value head；与子表同名开关保持一致。
            "add_value_head": False,

            # -----------------------------------------------------------------
            # OpenPI 专属配置层：这些字段覆盖 OpenPi0Config
            # -----------------------------------------------------------------
            "openpi": {
                # 选择 Pi0.5 + LIBERO 的模型/数据配置注册项，决定 LIBERO 图像、状态、
                # prompt、动作反归一化方式以及 checkpoint normalization stats 的布局。
                "config_name": "pi05_libero",

                # 两路有效视觉输入：第三人称主相机和腕部相机。OpenPI 可保留额外
                # 图像槽位，但 LIBERO 配置会对不存在的视角做 padding/mask。
                "num_images_in_input": 2,

                # flow-SDE 采样时的随机扰动强度，参与每步方差计算；选择纯 flow-ODE
                # 路径时该值不产生随机扩散。数值越大，探索性通常越强、轨迹越随机。
                "noise_level": 0.5,

                # 每次模型前向最终交给环境执行的连续动作数。模型可以生成更长的
                # action horizon，但 output_transform 只保留最前面的 5 步。
                "action_chunk": 5,

                # 从初始噪声积分/去噪到动作的迭代次数。更多步通常增加计算延迟，
                # 更少步速度更快；该服务固定 5 步以匹配当前 checkpoint 配置。
                "num_steps": 5,

                # 按“只训练 action expert”的方式构造模型：加载器会冻结 PaliGemma
                # 视觉语言主干。服务处于 eval 模式，但该标志仍需与 checkpoint 兼容。
                "train_expert_only": True,

                # 环境有效动作维数，用于 loss/log-prob 等路径只统计前 7 维，忽略
                # OpenPI 内部 action tensor 为统一模型接口保留的 padding 维度。
                "action_env_dim": 7,

                # flow 采样器类型。``flow_sde`` 允许按 noise_level 注入随机项；RLinf
                # 还支持 flow_ode、flow_noise、flow_cps 等路径。
                "noise_method": "flow_sde",

                # 关闭强化学习 critic/value head；当前服务只需要动作预测结果。
                "add_value_head": False,

                # 若启用 value head，False 表示从 action-expert suffix 特征估值；
                # True 才会从 VLM prefix 特征估值。由于值头关闭，本项当前不生效。
                "value_after_vlm": False,

                # 从 VLM prefix 估值时如何聚合 token；mean_token 表示对选中 token
                # 求均值。仅在 add_value_head 和 value_after_vlm 同时开启时生效。
                "value_vlm_mode": "mean_token",

                # 是否在 critic 输入处截断梯度。None 在当前布尔判断中等同未开启；
                # 且 value head 已关闭，因此不会影响本推理服务。
                "detach_critic_input": None,

                # 关闭 DSRL 扩展，使用标准 OpenPI flow 动作采样，不创建 DSRL 专用
                # 图像/状态编码器、Gaussian noise policy 和多 Q-head critic。
                "use_dsrl": False,
            },
        }
    )


def _decode_image_block(block: dict[str, Any]) -> np.ndarray:
    """把 RPC 中的 PNG/base64 图像块还原为 ``uint8[H,W,3]``。

    RPC 负载必须是 JSON-safe，因此客户端不能直接发送 NumPy 数组，而是先编码
    为 PNG，再编码为 base64 字符串。这里在进入模型前集中校验格式、内容和通道
    数，避免错误输入在 OpenPI 的观测处理器深处才以难理解的方式失败。
    """
    # imageio 延迟导入，健康检查和模块扫描不必提前加载图像编解码依赖。
    import imageio.v2 as imageio

    fmt = (block.get("format") or "png").lower()
    if fmt != "png":
        raise ValueError(f"unsupported image format: {fmt!r} (only 'png')")

    data = block.get("data")
    if not isinstance(data, str) or not data:
        raise ValueError("image block missing base64 'data'")

    # base64 -> PNG bytes -> RGB NumPy 数组。
    raw = base64.b64decode(data)
    img = np.asarray(imageio.imread(io.BytesIO(raw)))
    if img.ndim != 3 or img.shape[-1] != 3:
        raise ValueError(f"image must be HxWx3 RGB; got {img.shape}")
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    return img


def _build_env_obs(
    instruction: str,
    images: dict[str, Any],
    state: list,
) -> dict[str, Any]:
    """将传输层负载重建为 OpenPI 期望的批量观测字典。

    客户端面向单个 LIBERO 环境，发送的图像是 ``[H,W,3]``；模型接口面向批量，
    因而服务端补成 ``[B=1,H,W,3]``。状态已经按 wire contract 发送为
    ``[B,state_dim]``，这里只负责转成 float32 并校验维度。
    """
    if "main" not in images:
        raise ValueError("'images.main' is required")

    main = _decode_image_block(images["main"])
    main_batch = main[None]

    # ``wrist_images`` / ``extra_view_images`` 即使不用也必须显式存在。OpenPI 的
    # obs_processor 会直接用下标访问这些键，而不是通过 ``dict.get`` 读取；缺键
    # 会产生 KeyError，值为 None 才表示当前视角未提供。
    obs: dict[str, Any] = {
        "main_images": main_batch,
        "task_descriptions": [str(instruction)] * main_batch.shape[0],
        "wrist_images": None,
        "extra_view_images": None,
    }

    # 可选视角采用与主相机相同的解码和 batch 维约定。
    if isinstance(images.get("wrist"), dict):
        obs["wrist_images"] = _decode_image_block(images["wrist"])[None]
    if isinstance(images.get("extra"), dict):
        obs["extra_view_images"] = _decode_image_block(images["extra"])[None]

    states = np.asarray(state, dtype=np.float32)
    if states.ndim != 2:
        raise ValueError(f"state must be [B, state_dim]; got shape {states.shape}")
    obs["states"] = states
    return obs


# ---------------------------------------------------------------------------
# Facade implementing the rpent.utils.vla_client protocol
# ---------------------------------------------------------------------------


class VLAFacade(RpcFacade):
    """在 Pi0.5 模型之上实现 ``VLAClient`` 所需的 RPC 协议。

    Facade 的生命周期与服务进程一致：构造时只加载一次模型，随后处理任意多次
    ``predict`` 请求。通用基类 ``RpcFacade`` 负责 ``healthz``、``shutdown``、
    HTTP/socket 监听、父进程存活监控和串行化 dispatch；本类只实现 VLA 业务方法。
    """

    def __init__(self, model_path: str):
        """加载指定 checkpoint 的 Pi0.5，并建立可复用的推理 Facade。

        RLinf/OpenPI 在这里延迟导入，确保调用方已先配置 GPU 可见性。模型仅在服务
        进程启动时构造一次，后续所有 ``predict`` RPC 共用同一份 CUDA 权重。

        Args:
            model_path: 本地 Pi0.5 checkpoint 路径，原样写入 RLinf 模型配置。
        """
        super().__init__()

        # 这里延迟导入 RLinf：普通 CLI 参数解析、环境注册和文档构建都不应因为
        # 导入本模块而立即加载大型模型依赖。
        from rlinf.models.embodiment.openpi import get_model as get_openpi_model

        cfg = build_model_cfg(model_path=model_path)
        t0 = time.time()
        logger.info("loading Pi0.5 (model_path=%s) ...", cfg["model_path"])

        # main() 已在第一次 CUDA 上下文创建前设置 CUDA_VISIBLE_DEVICES。
        # 因而 ``.cuda()`` 使用该服务进程可见的 GPU；``eval`` 关闭训练态行为。
        self._model = get_openpi_model(cfg, torch_dtype=None).cuda().eval()
        logger.info("model ready in %.1fs", time.time() - t0)

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """把通用 RPC 方法名路由到 VLA 业务函数。

        ``healthz`` 和 ``shutdown`` 已由 ``RpcFacade`` 在调用本方法前处理，因此
        此处只允许协议公开的 ``predict``，未知方法立即报错并由传输层包装返回。
        """
        if method == "predict":
            return self.predict(*args, **kwargs)
        raise ValueError(f"unknown RPC method: {method!r}")

    def predict(
        self,
        instruction: str,
        images: dict[str, Any],
        state: list,
        mode: str = "eval",
    ) -> dict[str, Any]:
        """执行一次 Pi0.5 前向推理并返回 JSON-safe 动作块。

        输入是 ``VLAClient`` 定义的 wire schema；输出通常为
        ``[B=1, chunk, action_dim=7]``。NumPy/Torch 张量不能直接进入 JSON，故在
        服务端统一转成 float32 嵌套列表，同时携带 shape/dtype 便于诊断协议问题。
        """
        env_obs = _build_env_obs(instruction, images, state)

        # 推理服务永远不需要构建反向传播图，可显著减少显存和运行时开销。
        with torch.no_grad():
            actions, _ = self._model.predict_action_batch(env_obs, mode=mode)

        # RLinf 实现可能返回 torch.Tensor，也可能返回兼容 NumPy 的对象；这里统一
        # 搬到 CPU，并固定 wire dtype 为 float32。
        actions_np = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        ).astype(np.float32)
        return {
            "actions": actions_np.tolist(),
            "shape": list(actions_np.shape),
            "dtype": "float32",
        }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """解析服务参数、加载一次模型并阻塞运行 RPC 服务。"""
    p = argparse.ArgumentParser()
    p.add_argument("--transport", choices=["socket", "http"], default="http")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=0)
    p.add_argument(
        "--model-path",
        default=None,
        help="Pi0.5 checkpoint (defaults to PI05_CHECKPOINT_PATH env)",
    )
    p.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    p.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    args = p.parse_args()

    # RPent 主进程通过 --cuda-device 把用户选择传给三个子服务。必须在模型构造和
    # 第一次 CUDA 操作之前设置可见设备；子进程内的 ``cuda:0`` 随后对应所选 GPU。
    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logger.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; "
                "overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    # 命令行显式值优先；否则读取 PI05_CHECKPOINT_PATH。尽早失败可以让父进程的
    # wait_for_ready 发现子进程退出，并提示用户检查 vla_server.log。
    model_path = args.model_path or get_pi05_checkpoint_path()
    if not model_path:
        raise RuntimeError(
            "PI05_CHECKPOINT_PATH is not set; provide the Pi0.5 checkpoint "
            "path via --model-path or the environment."
        )

    # 构造 Facade 时同步加载模型。模型就绪后 serve() 才开始响应 healthz，因此
    # 主进程的 readiness 检查同时也是“模型已可推理”的屏障。
    facade = VLAFacade(model_path=model_path)

    # serve() 阻塞到收到 shutdown，或在 --parent-watch 下检测到父进程死亡。
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
