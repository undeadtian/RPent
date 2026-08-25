"""本地 SAM 3.0 图像分割模型的独立 RPC 服务。

该进程持有一份常驻 GPU 的 SAM3 模型，通过 HTTP 或 socket 暴露 ``segment``
方法。一次请求依次经过 ``Sam3Facade.segment`` 的 Pydantic wire schema 校验、
``Sam3Engine.segment`` 的串行推理，以及 ``_select_top`` 的候选筛选；响应最多返回
一个掩码。输入图像以 base64 字符串承载原始图片字节，输出二值掩码则压缩为 PNG
后再次编码为 base64，因而整个 RPC 负载都可安全地放入 JSON。

引擎只在进程启动时加载一次模型，并用锁保护处理器的可变图像状态和 GPU 前向；
最近一张图片的 backbone 特征按原始字节摘要缓存，可供不同文本或点提示复用。
``main`` 会先按 ``--cuda-device``（如提供）设置 ``CUDA_VISIBLE_DEVICES``，
``Sam3Engine.load`` 随后才延迟导入 PyTorch/SAM3 并创建 CUDA 上下文，使进程内
``cuda:0`` 指向所选择的物理设备。

手工启动示例::

    SAM3_CHECKPOINT_PATH=/path/to/sam3.pt \
        python -m robots.libero.sam3_server \
        --transport http --host 127.0.0.1 --port 8114

RPent 通常会自动启动该进程，无需手工执行上述命令。
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
import threading
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field, model_validator

from rpent.utils.logging import get_logger
from rpent.utils.rpc import RpcFacade

logger = get_logger("sam3_server")


class SegmentRequest(BaseModel):
    """``segment`` RPC 的入站 wire schema。

    Pydantic 在调用推理引擎前完成字段类型、点坐标长度和分数范围校验。客户端必须
    在 ``text_prompt`` 与 ``point`` 中恰好提供一种提示：文本会去除首尾空白；点
    使用图像数组坐标 ``[row, col]``。``image_base64`` 承载图片文件的原始字节，
    具体图片格式随后由 Pillow 解码。
    """

    image_base64: str
    text_prompt: str | None = None
    point: list[int] | None = Field(default=None, min_length=2, max_length=2)
    min_score: float = Field(default=0.2, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _exactly_one_prompt(self) -> "SegmentRequest":
        """强制文本提示和单点提示二选一，并规范化非空文本。"""
        has_text = isinstance(self.text_prompt, str) and bool(self.text_prompt.strip())
        has_point = self.point is not None
        if has_text == has_point:
            raise ValueError("provide exactly one of text_prompt or point")
        if has_text:
            self.text_prompt = self.text_prompt.strip()
        return self


class SegmentResponse(BaseModel):
    """``segment`` RPC 的出站 wire schema。

    ``found=False`` 时可返回候选分数、边界框和失败原因；找到有效候选时，掩码以
    灰度 PNG 的 base64 字符串返回，并用 ``mask_shape=[H, W]`` 明确解码后的二维
    尺寸。文本分割可能提供 ``box``，点提示路径则允许该字段为空。
    """

    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask_png_base64: str | None = None
    mask_shape: list[int] | None = None
    reason: str | None = None


def _encode_mask_png(mask: np.ndarray) -> str:
    """把二维二值掩码编码为 JSON-safe 的灰度 PNG/base64 字符串。

    掩码先转成 ``uint8`` 并映射到 0/255，再写入内存中的无损 PNG；最终使用 ASCII
    base64，避免 RPC 层直接传输 NumPy 数组或任意二进制字节。
    """
    image = Image.fromarray(np.asarray(mask, dtype=np.uint8) * 255, mode="L")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class Sam3Engine:
    """封装 SAM3 推理，并串行访问模型和“最近图像”特征缓存。

    官方处理器把当前图片的 backbone 特征保存在可变 ``state`` 中，模型的交互式
    预测器也不保证可并发调用。因此 ``segment`` 用同一把锁覆盖缓存查询、提示处理
    和候选选择，既避免请求间状态串扰，也避免多个线程同时占用同一 GPU 模型。
    缓存只保存最近一张图片，以图片原始字节的 SHA-256 摘要判定是否可复用。
    """

    def __init__(
        self,
        model: Any,
        processor: Any,
        *,
        device: str = "cuda",
        torch_module: Any | None = None,
    ) -> None:
        """保存已构造的模型依赖，并初始化推理锁与单条图像缓存。

        ``torch_module`` 由 ``load`` 注入，用于 CUDA autocast；测试或非 CUDA 替身
        可以传入 ``None``，此时推理上下文退化为 ``nullcontext``。
        """
        self._model = model
        self._processor = processor
        self._device = device
        self._torch = torch_module
        self._lock = threading.Lock()
        self._image_digest: str | None = None
        self._image_state: dict[str, Any] | None = None

    @classmethod
    def load(cls, checkpoint: str) -> "Sam3Engine":
        """从本地 checkpoint 加载官方 SAM 3.0 模型和交互式点提示头。

        PyTorch 与 SAM3 均在此处延迟导入：普通模块扫描、CLI 参数解析不会提前加载
        可选大依赖，更重要的是 ``main`` 能在首次 CUDA 探测及上下文创建前设置设备
        可见性。模型和处理器构造完成后才返回可服务的引擎实例。
        """
        # 延迟导入必须位于 CUDA_VISIBLE_DEVICES 设置之后；缺少可选依赖时，把底层
        # ImportError 转换为带安装提示的服务启动错误。
        try:
            import torch
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError as exc:
            raise RuntimeError(
                "local SAM3 dependencies are missing; install RPent with "
                '`pip install -e ".[sam3]"` (or `.[full]`)'
            ) from exc

        if not torch.cuda.is_available():
            raise RuntimeError("local SAM3 requires a CUDA-capable GPU")

        # TF32 加速支持该格式的矩阵乘和卷积；可见设备已经由 main 固定，因此这里的
        # 逻辑设备 0 对应 --cuda-device 选择的物理 GPU。
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.cuda.set_device(0)

        # 先展开用户目录并解析绝对路径，在进入耗时的模型构造前给出明确文件错误。
        resolved = Path(checkpoint).expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM3 checkpoint not found: {resolved}")
        checkpoint_path = str(resolved)

        logger.info("loading SAM 3.0 checkpoint: %s", checkpoint_path)
        try:
            # 打开 instance interactivity，文本处理器和 predict_inst 点提示头即可共享
            # 同一份图像模型；checkpoint 明确从本地读取，不走 Hugging Face 下载。
            model = build_sam3_image_model(
                device="cuda",
                checkpoint_path=checkpoint_path,
                load_from_HF=False,
                enable_inst_interactivity=True,
            )
        except Exception as exc:
            raise RuntimeError(
                f"failed to load SAM 3.0 checkpoint: {checkpoint_path}"
            ) from exc

        # 处理器内部阈值设为 0，保留全部候选；统一的 min_score 过滤稍后在
        # _select_top 中执行，使文本与点提示遵循同一 RPC 语义。
        processor = Sam3Processor(model, device="cuda", confidence_threshold=0.0)
        return cls(model, processor, device="cuda", torch_module=torch)

    def segment(
        self,
        image_bytes: bytes,
        *,
        text_prompt: str | None,
        point: list[int] | None,
        min_score: float,
    ) -> SegmentResponse:
        """对一张图片执行文本提示或单个前景点提示分割。

        锁内先取得或计算该图片的 backbone ``state``，其原始尺寸用于校验
        ``point=[row, col]``。文本路径由处理器生成候选掩码/分数/框；点路径调用
        交互式预测头生成多候选掩码，二者最终都交给 ``_select_top``。
        """
        # 锁覆盖缓存和完整推理流程，防止其他请求替换处理器当前图像或交叉修改
        # state；缓存命中时只跳过图像 backbone，不跳过本次提示推理。
        with self._lock:
            state = self._state_for_image(image_bytes)
            height = int(state["original_height"])
            width = int(state["original_width"])
            if text_prompt is not None:
                return self._segment_text(state, text_prompt, min_score)

            # Pydantic 已保证两类提示恰好存在一个；assert 同时帮助类型收窄。坐标在
            # 进入模型前按原图 [H, W] 边界检查，避免静默裁剪或错误索引。
            assert point is not None
            row, col = point
            if row < 0 or col < 0 or row >= height or col >= width:
                raise ValueError(
                    f"point [row, col] {point} is outside image shape "
                    f"[{height}, {width}]"
                )
            return self._segment_point(state, row, col, min_score)

    def _inference_context(self):
        """返回当前设备适用的推理精度上下文。

        CUDA 推理采用 bfloat16 autocast 以降低显存和计算开销；测试替身、未注入
        PyTorch 或非 CUDA 设备使用空上下文，不额外改变数值类型。
        """
        if self._torch is None or not self._device.startswith("cuda"):
            return nullcontext()
        return self._torch.autocast("cuda", dtype=self._torch.bfloat16)

    def _state_for_image(self, image_bytes: bytes) -> dict[str, Any]:
        """解码图片并返回可复用的 SAM3 backbone 状态。

        摘要基于收到的原始字节，因此完全相同的 RPC 图片可命中最近一项缓存。缓存
        未命中时，Pillow 从内存字节流解码并统一为 RGB，``set_image`` 再生成含原图
        高宽和图像特征的状态；成功后才替换摘要与状态，异常不会污染旧缓存。
        """
        digest = hashlib.sha256(image_bytes).hexdigest()
        if digest == self._image_digest and self._image_state is not None:
            return self._image_state

        # 输入 base64 已在 Facade 中还原为字节，这里负责真正的图片格式解码和 RGB
        # 归一；损坏或不受支持的数据统一映射为调用方可理解的 ValueError。
        try:
            with Image.open(io.BytesIO(image_bytes)) as source:
                image = source.convert("RGB")
        except Exception as exc:
            raise ValueError(f"invalid image data: {exc}") from exc

        # set_image 是昂贵的 backbone 阶段；文本和点提示都复用其 [H, W] 图像状态。
        with self._inference_context():
            state = self._processor.set_image(image)
        self._image_digest = digest
        self._image_state = state
        return state

    def _segment_text(
        self,
        state: dict[str, Any],
        prompt: str,
        min_score: float,
    ) -> SegmentResponse:
        """用非空文本提示生成候选，并选出满足阈值的最高分实例。"""
        # 文本处理器返回的 masks/scores/boxes 通常带候选维 N；保留原对象交给统一
        # 选择器完成 CPU/NumPy 转换和维度规范化。
        with self._inference_context():
            output = self._processor.set_text_prompt(prompt=prompt, state=state)
        return self._select_top(
            masks=output.get("masks"),
            scores=output.get("scores"),
            boxes=output.get("boxes"),
            min_score=min_score,
        )

    def _segment_point(
        self,
        state: dict[str, Any],
        row: int,
        col: int,
        min_score: float,
    ) -> SegmentResponse:
        """用一个正样本点生成多候选掩码，并返回其中最高分实例。

        RPC 坐标采用数组习惯 ``[row, col]``，而交互式预测头要求几何坐标
        ``[x, y]``，因此传入模型前重排为 ``[[col, row]]``，形状为 ``[1, 2]``；
        ``point_labels=[1]`` 表示该点属于前景。
        """
        if getattr(self._model, "inst_interactive_predictor", None) is None:
            raise RuntimeError("SAM3 instance interactivity is not enabled")
        point_coords = np.asarray([[col, row]], dtype=np.float32)
        point_labels = np.asarray([1], dtype=np.int64)
        with self._inference_context():
            masks, scores, _ = self._model.predict_inst(
                state,
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        return self._select_top(
            masks=masks,
            scores=scores,
            boxes=None,
            min_score=min_score,
        )

    @staticmethod
    def _select_top(
        *,
        masks: Any,
        scores: Any,
        boxes: Any,
        min_score: float,
    ) -> SegmentResponse:
        """规范化 SAM3 候选张量并选择最高分的有效二维掩码。

        ``masks`` 可为 ``[N,1,H,W]``、``[N,H,W]`` 或单个 ``[H,W]``，分数统一
        展平为 ``[N]``。函数校验候选数量一致后取 ``argmax``，可选读取对应边界框，
        再依次应用 ``min_score`` 和非空掩码检查；只有最终成功项会编码为 PNG/base64。
        """
        if masks is None or scores is None:
            return SegmentResponse(found=False, reason="SAM3 returned no candidate")

        # 官方实现可能返回 CUDA Tensor，也允许测试直接传 NumPy；统一转到 CPU
        # NumPy 后再做形状检查。scores 的尾部单例维在此全部压平。
        masks_array = (
            masks
            if isinstance(masks, np.ndarray)
            else masks.detach().float().cpu().numpy()
        )
        scores_array = (
            scores
            if isinstance(scores, np.ndarray)
            else scores.detach().float().cpu().numpy()
        ).reshape(-1)
        if masks_array.size == 0 or scores_array.size == 0:
            return SegmentResponse(found=False, reason="SAM3 returned no candidate")

        # 去除每个候选的单通道维，或为单个二维掩码补候选维，最终固定为 [N,H,W]。
        if masks_array.ndim == 4 and masks_array.shape[1] == 1:
            masks_array = masks_array[:, 0]
        if masks_array.ndim == 2:
            masks_array = masks_array[None]
        if masks_array.ndim != 3 or masks_array.shape[0] != scores_array.shape[0]:
            raise RuntimeError(
                "unexpected SAM3 candidate shapes: "
                f"masks={masks_array.shape}, scores={scores_array.shape}"
            )

        # 所有候选中只返回最高分项。文本路径若带 boxes，则抽取同一索引的前四个
        # 数值作为 wire schema 中的边界框；点提示路径传入 None。
        index = int(np.argmax(scores_array))
        score = float(scores_array[index])
        box: list[float] | None = None
        if boxes is not None:
            boxes_array = (
                boxes
                if isinstance(boxes, np.ndarray)
                else boxes.detach().float().cpu().numpy()
            )
            if boxes_array.ndim >= 2 and index < boxes_array.shape[0]:
                box = [float(value) for value in boxes_array[index].reshape(-1)[:4]]
        if score < min_score:
            return SegmentResponse(
                found=False,
                score=score,
                box=box,
                reason=f"top score {score:.3f} is below min_score {min_score:.3f}",
            )

        # 模型输出以 >0 二值化；掩码必须严格为二维且至少包含一个前景像素。
        mask = np.asarray(masks_array[index]) > 0
        if mask.ndim != 2 or not mask.any():
            return SegmentResponse(
                found=False,
                score=score,
                box=box,
                reason="SAM3 returned an empty mask",
            )
        return SegmentResponse(
            found=True,
            score=score,
            box=box,
            mask_png_base64=_encode_mask_png(mask),
            mask_shape=[int(mask.shape[0]), int(mask.shape[1])],
        )


class Sam3Facade(RpcFacade):
    """通过共享 HTTP/socket 传输层暴露 :class:`Sam3Engine`。

    ``RpcFacade`` 负责监听、JSON 串行化、``healthz``/``shutdown`` 和父进程存活
    监控；本类只实现 ``segment`` 的业务分派、Pydantic 边界校验以及 base64 字节
    转换。引擎自身的锁保证不同传输线程最终串行访问模型。
    """

    def __init__(self, engine: Sam3Engine) -> None:
        """绑定已完成模型加载的引擎；Facade 本身不重复持有或加载模型。"""
        super().__init__()
        self._engine = engine

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """把公开的 ``segment`` 方法路由到业务入口，并拒绝未知 RPC 名称。"""
        if method == "segment":
            return self.segment(*args, **kwargs)
        raise ValueError(f"unknown RPC method: {method!r}")

    def segment(
        self,
        image_base64: str,
        *,
        text_prompt: str | None = None,
        point: list[int] | None = None,
        min_score: float = 0.2,
    ) -> dict[str, Any]:
        """校验 wire 请求、解码图片字节、执行分割并返回 JSON-safe 字典。

        Pydantic 先落实“文本/点二选一”等协议约束；``validate=True`` 严格拒绝非法
        base64 字符，图片文件内容稍后由引擎中的 Pillow 校验。响应通过
        ``exclude_none=True`` 省略不适用于当前结果的可选字段。
        """
        request = SegmentRequest(
            image_base64=image_base64,
            text_prompt=text_prompt,
            point=point,
            min_score=min_score,
        )

        # wire 字符串 -> 原始图片文件字节；二值掩码走相反方向，由
        # _encode_mask_png 生成 PNG 字节 -> base64 ASCII 字符串。
        image_bytes = base64.b64decode(request.image_base64, validate=True)
        if not image_bytes:
            raise ValueError("image_base64 is empty")
        response = self._engine.segment(
            image_bytes,
            text_prompt=request.text_prompt,
            point=request.point,
            min_score=request.min_score,
        )
        return response.model_dump(exclude_none=True)


def _build_argparser() -> argparse.ArgumentParser:
    """构造服务 CLI，描述传输端点、CUDA 设备和父进程监控选项。"""
    parser = argparse.ArgumentParser(description="RPent local SAM 3.0 server")
    parser.add_argument("--transport", choices=["socket", "http"], default="http")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8114)
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=None,
        help="GPU device exposed through CUDA_VISIBLE_DEVICES.",
    )
    parser.add_argument(
        "--parent-watch",
        action="store_true",
        help="watch parent process via stdin pipe and exit when it dies",
    )
    return parser


def main() -> None:
    """配置设备、加载一次 SAM3，并阻塞服务直到关闭或父进程退出。

    生命周期顺序不可颠倒：先解析参数并设置 CUDA 可见设备，再解析 checkpoint、
    延迟导入模型依赖并同步完成加载，最后构造 Facade 并进入 ``serve``。因此只有
    模型可用后端口才会开始服务；``serve`` 负责正常 shutdown 与 parent-watch。
    """
    args = _build_argparser().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s")

    # 必须在 Sam3Engine.load 延迟导入 torch、探测 CUDA 和创建模型前设置。可见设备
    # 会被重新编号，所以 load 内固定选择的 cuda:0 正是这里指定的物理 GPU。
    if args.cuda_device is not None:
        target = str(args.cuda_device)
        prev = os.environ.get("CUDA_VISIBLE_DEVICES")
        if prev is not None and prev != target:
            logging.warning(
                "CUDA_VISIBLE_DEVICES=%s is already set; overriding with --cuda-device=%s",
                prev,
                args.cuda_device,
            )
        os.environ["CUDA_VISIBLE_DEVICES"] = target

    # checkpoint 只从环境变量取得；在耗时加载前尽早失败，便于父进程把启动日志
    # 反馈给调用方。load 返回前会完成文件校验、模型构造和处理器初始化。
    checkpoint = os.environ.get("SAM3_CHECKPOINT_PATH")
    if not checkpoint:
        raise RuntimeError(
            "SAM3_CHECKPOINT_PATH is not set; export the path to sam3.pt "
            "before starting RPent"
        )
    engine = Sam3Engine.load(checkpoint)

    # Facade 仅绑定已就绪引擎；serve 阻塞处理请求，直到收到 shutdown，或启用
    # --parent-watch 时从 stdin 检测到父进程已退出。
    facade = Sam3Facade(engine)
    facade.serve(
        transport=args.transport,
        host=args.host,
        port=args.port,
        parent_watch=args.parent_watch,
    )


if __name__ == "__main__":
    main()
