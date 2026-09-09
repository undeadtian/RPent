# SAM3 Image Model 详解

本文说明当前 RPent 工程中 SAM3 图像模型的组成、构建过程、推理流程、RPC 服务化方式，以及分割结果如何转换为机器人可使用的世界坐标。

> 本文基于当前环境中的 `sam3 0.1.4`、`robots/libero/sam3_server.py` 和 `rpent/utils/sam3_client.py`。这里讨论的是单张图像模型，不是 SAM3 视频跟踪模型。

## 1. 它是什么

`sam3_image_model` 不是 RPent 中某个单独变量的名称，通常指下面这个构造函数返回的完整单图模型：

```python
from sam3.model_builder import build_sam3_image_model

model = build_sam3_image_model(...)
```

它接收图像以及文本或点提示，输出一个或多个候选实例掩码及置信度。RPent 将模型放在独立 GPU 进程中常驻，通过 HTTP/socket RPC 提供 `segment` 方法，避免每次工具调用都重新加载约数 GB 的 checkpoint。

RPent 支持两种提示模式：

1. **文本提示**：例如 `"the black bowl"`；
2. **单点提示**：例如 `[row, col]`，表示目标内部的一个前景点。

两种提示必须且只能选择一种。

## 2. 相关代码

| 文件 | 职责 |
|---|---|
| `sam3/model_builder.py` | 官方模型组件构造和 checkpoint 加载 |
| `robots/libero/sam3_server.py` | GPU 推理引擎、缓存、候选筛选及 RPC 服务 |
| `rpent/utils/sam3_client.py` | 图片编码、RPC 调用及掩码解码 |
| `robots/libero/tools.py` | 智能体 `segment` 工具和世界坐标计算 |
| `robots/libero/__init__.py` | SAM3 daemon 的 Session/任务生命周期 |
| `rpent/utils/rpc.py` | 通用 HTTP/socket RPC 与健康检查 |
## 3. 官方构建函数

当前安装版本的函数签名为：

```python
def build_sam3_image_model(
    bpe_path=None,
    device="cuda" if torch.cuda.is_available() else "cpu",
    eval_mode=True,
    checkpoint_path=None,
    load_from_HF=True,
    enable_segmentation=True,
    enable_inst_interactivity=False,
    compile=False,
): ...
```

### 3.1 参数

- `bpe_path`：文本编码器使用的 BPE 词表。未传入时使用 SAM3 包内的 `assets/bpe_simple_vocab_16e6.txt.gz`。
- `device`：模型最终放置设备，例如 `"cuda"` 或 `"cpu"`。
- `eval_mode`：是否切换为推理模式。默认 `True`。
- `checkpoint_path`：本地 checkpoint 文件路径。
- `load_from_HF`：当没有提供 checkpoint 时，是否从 Hugging Face 的 `facebook/sam3` 下载 `sam3.pt`。
- `enable_segmentation`：是否创建像素解码器和分割头。RPent 使用默认值 `True`。
- `enable_inst_interactivity`：是否创建兼容点提示的交互式实例预测器。
- `compile`：是否启用 PyTorch compile；启用后内部 `compile_mode="default"`。

### 3.2 构建顺序

函数按照以下顺序组装模型：

1. 创建视觉骨干网络；
2. 根据 BPE 词表创建文本编码器；
3. 组合视觉语言 backbone；
4. 创建 SAM3 transformer；
5. 创建点积评分模块；
6. 创建 segmentation head；
7. 创建输入几何编码器；
8. 可选创建 `SAM3InteractiveImagePredictor`；
9. 组合成完整 `Sam3Image` 模型；
10. 加载 checkpoint；
11. 将模型移动到目标设备并设置 eval/train 模式。

简化示意：

```text
RGB Image ──> Vision Encoder ─┐
                             ├─> VL Backbone ─> Transformer ─> Segmentation Head ─> Masks
Text Prompt ─> Text Encoder ─┘                        └───────> Scoring Head ───────> Scores

Point Prompt ─> Geometry Encoder / Interactive Predictor ────────────────────────> Masks
```

### 3.3 RPent 的构建方式

RPent 在 `Sam3Engine.load()` 中使用：

```python
model = build_sam3_image_model(
    device="cuda",
    checkpoint_path=checkpoint_path,
    load_from_HF=False,
    enable_inst_interactivity=True,
)
```

这意味着：

- 强制使用本地 `SAM3_CHECKPOINT_PATH`；
- 不在服务启动期间访问 Hugging Face；
- 同一模型同时支持文本提示和单点提示；
- 必须有 CUDA GPU；
- 模型只加载一次并常驻显存。

随后创建：

```python
processor = Sam3Processor(
    model,
    device="cuda",
    confidence_threshold=0.0,
)
```

处理器阈值设为 0，目的是先保留模型候选，再由 RPent 的 `_select_top()` 统一执行 `min_score` 过滤，使文本提示与点提示具有相同的 API 语义。
## 4. GPU、精度和 checkpoint

服务启动时先处理 `--cuda-device`：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
```

此操作必须发生在导入 PyTorch 和创建 CUDA 上下文之前。物理 GPU 会被重新映射为进程内的逻辑 `cuda:0`，因此引擎固定执行 `torch.cuda.set_device(0)`。

RPent 同时启用 TF32：

```python
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
```

实际推理使用 BF16 autocast：

```python
with torch.autocast("cuda", dtype=torch.bfloat16):
    ...
```

checkpoint 来自：

```bash
export SAM3_CHECKPOINT_PATH=/path/to/sam3.pt
```

服务会先展开 `~`、解析绝对路径并验证文件存在，然后才进入耗时模型构建。当前环境使用的文件为：

```text
/mnt/nvme1n1/model/rlinf/sam3/sam3.pt
```

## 5. RPent 服务架构

SAM3 运行在独立进程中：

```text
Planner / segment tool
        │
        ▼
Sam3Client
        │ JSON + base64
        ▼
HTTP/socket RpcClient
        │
        ▼
Sam3Facade
        │ Pydantic 校验
        ▼
Sam3Engine
        │
        ▼
SAM3 GPU model
```

### 5.1 为什么使用独立进程

- 模型只加载一次，多个 TaskRun 可以复用；
- SAM3 显存与 env_server、Planner 隔离；
- 模型崩溃不会直接破坏环境 worker；
- HTTP/socket 都复用同一业务协议；
- Dashboard Session 可以让 SAM3 跨多个任务持续驻留。

### 5.2 生命周期

启动顺序：

1. 解析 host、port、transport 和 CUDA 参数；
2. 设置可见 GPU；
3. 读取 `SAM3_CHECKPOINT_PATH`；
4. 加载模型和处理器；
5. 构造 `Sam3Facade`；
6. 开始监听 RPC；
7. 响应 `healthz` 后被 RPent 标记为 ready。

关闭条件包括：

- 收到 `shutdown` RPC；
- `--parent-watch` 检测到父进程退出；
- 外部进程管理器发送终止信号。

Dashboard 模式下 SAM3 是 Session 级共享服务；普通单任务 CLI 中则由该任务拥有并在任务结束时关闭。
## 6. RPC 输入输出协议

### 6.1 请求

```json
{
  "image_base64": "<原始 PNG/JPEG 字节的 base64>",
  "text_prompt": "the black bowl",
  "point": null,
  "min_score": 0.2
}
```

或点提示：

```json
{
  "image_base64": "<base64>",
  "point": [150, 200],
  "min_score": 0.2
}
```

字段约束：

- `image_base64` 必须能严格 base64 解码，并且是 Pillow 可读取的图像；
- `text_prompt` 与 `point` 必须恰好提供一个；
- 空白文本视为未提供；
- `point` 必须是两个整数组成的 `[row, col]`；
- `min_score` 必须位于 `[0, 1]`。

### 6.2 响应

成功示例：

```json
{
  "found": true,
  "score": 0.873,
  "box": [120.0, 80.0, 260.0, 230.0],
  "mask_png_base64": "<灰度 PNG 的 base64>",
  "mask_shape": [1024, 1024]
}
```

失败示例：

```json
{
  "found": false,
  "score": 0.13,
  "box": [120.0, 80.0, 260.0, 230.0],
  "reason": "top score 0.130 is below min_score 0.200"
}
```

掩码不直接作为巨大 JSON 数组传输，而是先转成 0/255 灰度 PNG，再做 base64。这样既无损，又兼容所有 JSON RPC transport。

## 7. 文本提示推理

文本路径执行：

```python
state = processor.set_image(image)
output = processor.set_text_prompt(prompt=prompt, state=state)
```

返回对象通常包含：

```python
{
    "masks": ...,   # 候选掩码
    "scores": ...,  # 候选分数
    "boxes": ...,   # 候选边界框
}
```

图片先经视觉 backbone 编码，文本由 tokenizer/text encoder 编码，视觉和语言特征在 VL backbone/transformer 中融合。分割头生成候选掩码，评分模块给出匹配分数。

文本提示建议：

- 描述可见属性和相对位置，如 `"the red mug on the left"`；
- 避免仅使用模型可能不认识的内部资产名；
- 多个同类物体时加入颜色、方位和容器关系；
- 如果文本 grounding 不稳定，可改用单点提示。

## 8. 单点提示推理

RPent 工具使用图像数组坐标：

```text
point = [row, col]
```

交互式预测器使用几何坐标：

```text
[x, y] = [col, row]
```

因此服务端转换为：

```python
point_coords = np.asarray([[col, row]], dtype=np.float32)
point_labels = np.asarray([1], dtype=np.int64)

masks, scores, _ = model.predict_inst(
    state,
    point_coords=point_coords,
    point_labels=point_labels,
    multimask_output=True,
)
```

`point_labels=[1]` 表示前景点。`multimask_output=True` 会生成多个候选，RPent 再选择最高分候选。

点在推理前会根据原始图像尺寸严格检查；越界不会被静默裁剪，而是抛出明确错误。
## 9. 图像特征缓存与线程安全

### 9.1 最近一张图像缓存

`Sam3Engine` 只缓存最近一张图片的 backbone 状态：

```python
self._image_digest: str | None = None
self._image_state: dict[str, Any] | None = None
```

每次请求先对收到的**原始图片文件字节**计算 SHA-256：

```python
digest = hashlib.sha256(image_bytes).hexdigest()
```

如果摘要与最近一次相同，直接复用 `processor.set_image()` 生成的 state；否则使用 Pillow 解码并转成 RGB，再重新计算图像特征。缓存命中只省略图像 backbone 阶段，本次文本或点提示推理仍会执行。

这一策略有几个重要特性：

- 相同图片连续使用不同提示时可以复用图像特征；
- 缓存容量固定为一张图片，不会随 episode 增长；
- 摘要基于文件字节，而不是解码后的像素，因此像素相同但 PNG/JPEG 编码不同的图片不会命中；
- 新图片成功完成 `set_image()` 后才替换旧缓存，解码或推理异常不会污染已有缓存；
- SHA-256 在这里用于可靠识别输入，不承担安全认证用途。

### 9.2 为什么完整推理都在锁内

引擎使用普通 `threading.Lock`：

```python
with self._lock:
    state = self._state_for_image(image_bytes)
    ...
```

锁覆盖缓存查询、图片编码、提示推理和候选选择，而不只是缓存字段。原因是官方处理器和交互式预测器都可能持有与当前图像有关的可变状态。若多个 HTTP 请求并发执行，一个请求可能在另一个请求使用 state 时替换当前图像，导致请求间串扰。

因此，RPC 传输层可以并发接收请求，但同一个 SAM3 模型实例最终会串行执行分割。这样优先保证结果正确性和显存使用稳定性，代价是单实例不能并行提高吞吐量。

## 10. 候选规范化与最高分选择

文本提示和点提示产生候选的接口不同，但最后都进入 `_select_top()`。其处理过程如下。

### 10.1 张量转为 NumPy

来自模型的 CUDA Tensor 会执行：

```python
tensor.detach().float().cpu().numpy()
```

测试或替代实现也可以直接传入 NumPy 数组。分数始终展平为一维 `[N]`。

### 10.2 掩码形状规范化

允许的常见输入形状是：

- `[N, 1, H, W]`：移除单通道维，得到 `[N, H, W]`；
- `[N, H, W]`：保持不变；
- `[H, W]`：补候选维，得到 `[1, H, W]`。

规范化后，掩码必须严格为 `[N,H,W]`，并且候选数必须与分数数目一致。否则抛出 `RuntimeError`，因为这通常意味着 SAM3 版本或返回协议发生了不兼容变化。

### 10.3 分数、box 和掩码规则

1. 使用 `argmax(scores)` 选择全局最高分候选；
2. 文本路径如有 `boxes`，读取相同候选索引的前四个值；
3. 若最高分低于 `min_score`，返回 `found=false`，但保留最高分和可用的 box；
4. 候选掩码按 `> 0` 二值化；
5. 掩码必须为二维且至少有一个前景像素；
6. 成功掩码编码为 0/255 灰度 PNG，再编码为 base64。

空候选、低于阈值和空掩码都是正常的“未找到”结果，而不是进程错误。点提示路径当前不传递 box，因此点提示成功时 `box` 通常为空。

## 11. `Sam3Client`：编码、调用和解码

`Sam3Client` 把传输细节隐藏在 `RpcClient` 后面，同一客户端可连接 HTTP 或 socket 服务。

### 11.1 输入编码

`segment()` 接受：

```python
bytes | bytearray | memoryview | np.ndarray
```

- 如果传入 NumPy 图像，客户端先把数组编码为 PNG；
- 如果传入字节类对象，则直接视为已有 PNG/JPEG 等图片文件字节；
- 图片字节随后编码为 base64 ASCII 字符串；
- 文本会去除首尾空白；
- 点坐标会规范为两个整数；
- 客户端在发送前再次检查提示二选一和 `min_score` 范围。

默认 RPC 超时为 120 秒。首次推理或负载较高时，调用时间可能明显长于缓存命中的后续请求，实际耗时取决于 GPU、图片和运行环境。

### 11.2 `Sam3Result`

客户端将 wire response 解码为不可变 dataclass：

```python
@dataclass(frozen=True)
class Sam3Result:
    found: bool
    score: float | None = None
    box: list[float] | None = None
    mask: np.ndarray | None = None
    mask_shape: tuple[int, int] | None = None
    reason: str | None = None
```

成功响应中的 PNG 掩码会被解码并执行 `decoded > 0`，最终得到布尔 NumPy 数组。客户端还会验证：

- `found` 必须是布尔值；
- box 必须恰好包含四个值；
- 成功响应必须包含非空掩码和两个正整数构成的 shape；
- PNG 实际解码尺寸必须与 `mask_shape` 完全一致。

这些检查能在服务协议损坏或版本不匹配时尽早失败，避免错误掩码进入机器人坐标计算。

## 12. LIBERO `segment` 工具的完整流程

`robots/libero/tools.py::LiberoPrimitives.segment()` 是 Planner 实际看到的只读工具。它不会执行环境 action，也不会推进 episode。

完整流程如下：

```text
选择 EnvState 中的 step
        │
        ▼
选择同相机、同分辨率的 RGB + world map
        │
        ▼
Sam3Client.segment(image, prompt/point, min_score)
        │
        ├── found=false ──> 记录原因和 fallback
        │
        ▼
布尔 mask + 逐像素 world map
        │
        ▼
过滤无效世界点并计算 XYZ 中位数
        │
        ├── 保存 segment_XX.json
        └── 保存 segment_overlay_XX.png
```

### 12.1 step 与相机

- `step=-1` 表示使用 `EnvState` 中最近一步；
- `camera` 只允许 `"agentview"` 或 `"wrist"`；
- 工具读取已有 observation artifact，不重新渲染相机。

### 12.2 RGB 与世界图必须成对

工具按以下优先级选择产物：

1. `<camera>_high.png` + `<camera>_world_high.npz`；
2. `<camera>.png` + `<camera>_world.npz`。

只有两个文件同时出现在 step 的 artifact 记录中且底层文件都存在时才会选中。不能把高分辨率 RGB 与标准分辨率世界图混用，因为这样像素和三维点不会一一对应。

### 12.3 提示调用

文本模式：

```python
segment(prompt="the red mug", camera="agentview", step=-1)
```

点模式：

```python
segment(point=[row, col], camera="agentview", step=-1)
```

这里的 `state` 参数由 RPent 工具运行时注入，Planner 不需要也不应该手工构造。工具调用仍要求文本和点恰好提供一种。

## 13. 从 mask 计算世界坐标

环境预先为每个 RGB 像素计算对应的世界坐标，形成：

```text
world_map.shape == [H, W, >=3]
```

`_mask_to_world(mask, world_map)` 不会自动缩放 mask。如果二者的 `[H,W]` 不一致，直接返回 shape mismatch。这个设计避免插值后产生看似合理、实际错位的机器人坐标。

### 13.1 有效点过滤

先提取所有前景像素的三维点：

```python
ys, xs = np.where(mask)
pts = world_map[ys, xs]
```

然后只保留：

- XYZ 全部为有限值；
- 三轴绝对值之和大于 `1e-6`，即排除近零占位点。

默认至少需要 10 个有效世界点。数量不足时返回 `world_xyz=None`，而不是从极少数深度点估计位置。

### 13.2 中位数聚合

世界坐标按三个轴分别取中位数：

```python
world_xyz = [
    median(valid_x),
    median(valid_y),
    median(valid_z),
]
```

结果保留四位小数。相比均值，中位数对物体边界的背景深度、局部遮挡和少量离群值更稳健。

同时记录：

- `centroid_pixel=[median_x, median_y]`，注意这里是 `[x,y]`；
- `n_pixels`：mask 前景像素总数；
- `n_valid`：过滤后的有效世界点数；
- `mask_resized_to_world_shape=false`：明确表示没有执行尺寸变换。

`world_xyz` 是掩码区域的稳健位置估计，不等同于保证可抓取的接触点。薄物体、透明/反光物体或严重遮挡场景仍需要结合可视化和机器人策略判断。

## 14. 诊断产物

每次 `segment` 调用都在当前 step 寻找首个未占用序号，并使用两位编号：

```text
segment_00.json
segment_overlay_00.png
segment_01.json
segment_overlay_01.png
...
```

### 14.1 `segment_XX.json`

JSON 记录通常包含：

```json
{
  "found": true,
  "mode": "text",
  "camera": "agentview",
  "source_step": 3,
  "segment_index": 0,
  "image_artifact": "agentview_high.png",
  "min_score": 0.2,
  "score": 0.873,
  "box": [120.0, 80.0, 260.0, 230.0],
  "mask_shape": [1024, 1024],
  "prompt": "the red mug",
  "world_xyz": [0.1234, -0.0567, 0.8912],
  "world_artifact": "agentview_world_high.npz",
  "centroid_pixel": [190, 155],
  "n_pixels": 8200,
  "n_valid": 8075,
  "mask_resized_to_world_shape": false
}
```

点提示时写入 `point` 而不是 `prompt`。未找到目标或世界坐标计算失败时，JSON 仍会尽量落盘，并包含 `error` 或 `world_error`，以保留可审计信息。

### 14.2 `segment_overlay_XX.png`

overlay 在原 RGB 图的 mask 区域叠加半透明红色：

```text
55% 原始像素 + 45% 红色
```

只有图片为三通道且尺寸与 mask 完全一致时才生成。overlay 用于人工确认目标边界是否正确，不参与后续世界坐标计算。成功保存后，工具响应还会携带 `_image_bytes`，供 Dashboard 或其他展示层直接显示。

## 15. 错误处理与 fallback

错误被分成三类。

### 15.1 请求错误

常见情况：

- 文本和点同时提供或同时缺失；
- `point` 不是 `[row,col]`；
- 点超出原始图像范围；
- `min_score` 不在 `[0,1]`；
- base64 非法、为空或图片无法解码。

客户端可在本地发现一部分错误，其余由 Pydantic、Facade 或引擎返回。

### 15.2 正常的“未找到”

以下情况返回 `found=false`：

- 模型没有返回候选；
- 最高分低于 `min_score`；
- 最高分候选的掩码为空。

这不表示 SAM3 服务崩溃。Planner 可以改进文本描述、改用点提示、换相机或调整阈值，但降低阈值会增加误分割风险。

### 15.3 服务或 artifact 故障

LIBERO 工具会把 RPC 超时、服务不可达、世界图缺失、shape 不匹配、有效深度过少及产物保存失败转换为结构化字典，而不是让异常终止 episode。服务调用失败时会明确给出：

```text
Use manual visual localization and back_project.
```

该 fallback 表示 Planner 可通过视觉估计像素位置，再调用 `back_project`；它不会自动执行抓取动作，也不会隐式推进环境。

## 16. 使用示例

### 16.1 独立启动 SAM3 HTTP 服务

在 `rpent` Conda 环境中：

```bash
conda activate rpent
export SAM3_CHECKPOINT_PATH=/mnt/nvme1n1/model/rlinf/sam3/sam3.pt
python robots/libero/sam3_server.py \
  --transport http \
  --host 127.0.0.1 \
  --port 8114 \
  --cuda-device 0
```

如果由 RPent 自动启动，本地服务会增加 `--parent-watch`，使父进程退出后 daemon 自动结束。若使用已启动的外部服务，可向 LIBERO 命令传入：

```bash
--sam3-endpoint http://127.0.0.1:8114
```

不传 `--sam3-endpoint` 时，RPent 自动分配本地端口并启动服务。

### 16.2 Python 客户端文本分割

```python
from rpent.utils.http_rpc import HttpRpcClient
from rpent.utils.sam3_client import Sam3Client

rpc = HttpRpcClient("http://127.0.0.1:8114")
client = Sam3Client(rpc)

with open("agentview.png", "rb") as file:
    result = client.segment(
        file.read(),
        text_prompt="the red mug on the left",
        min_score=0.2,
    )

if result.found:
    print(result.score, result.box, result.mask.shape)
else:
    print(result.reason)
```

### 16.3 Python 客户端点提示

```python
result = client.segment(
    image_array,
    point=[240, 380],  # [row, col]
    min_score=0.2,
)
```

不要把 `[x,y]` 直接当成客户端点坐标。客户端和工具统一使用 `[row,col]`，服务端会在调用交互式预测器前转换成 `[x,y]=[col,row]`。

## 17. 性能与设计限制

### 17.1 性能特征

- 模型和 checkpoint 只在 daemon 启动时加载一次；
- 连续对完全相同的图片使用不同提示时可复用 backbone state；
- 缓存只有最近一项，在图片 A、B、A 间切换时第三次 A 需要重新编码；
- 同一引擎的请求由锁串行执行；
- BF16 autocast 和 TF32 的实际收益依赖 GPU 架构与驱动环境；
- PNG/base64 提供无损且稳定的协议，但会产生编码、解码和传输开销；
- 高分辨率 artifact 会被优先选择，通常提供更细边界，同时也增加图像编码和推理工作量。

本文不提供固定延迟或显存数字，因为它们受 GPU、驱动、PyTorch/SAM3 版本、分辨率、提示和并发负载影响，应在目标机器上实测。

### 17.2 当前限制

- 每次只返回最高分实例，不返回全部候选；
- 点模式只支持一个正前景点，不支持负点或多点组合；
- 这里只使用单图模型，不提供跨帧视频跟踪；
- 文本分割质量依赖可见外观描述和场景歧义；
- `score` 适合候选排序和阈值过滤，不应视为严格校准概率；
- 世界坐标质量完全依赖对应 world map 的深度和相机标定；
- 掩码中位数是区域位置估计，不负责碰撞检测、抓取姿态或可达性判断；
- 单实例锁保证一致性，但限制高并发吞吐量。

## 18. 故障排查

### 18.1 `SAM3_CHECKPOINT_PATH is not set`

设置本地 checkpoint：

```bash
export SAM3_CHECKPOINT_PATH=/mnt/nvme1n1/model/rlinf/sam3/sam3.pt
```

服务使用 `load_from_HF=False`，不会在启动时自动下载缺失文件。

### 18.2 `SAM3 checkpoint not found`

检查环境变量指向普通文件，并注意服务会展开 `~` 后解析绝对路径。当前环境已验证的路径是：

```text
/mnt/nvme1n1/model/rlinf/sam3/sam3.pt
```

### 18.3 `local SAM3 requires a CUDA-capable GPU`

确认当前 `rpent` 环境中的 PyTorch 能看到 GPU，并检查 `--cuda-device` 与 `CUDA_VISIBLE_DEVICES`。服务把所选物理 GPU 映射为进程内 `cuda:0`。

### 18.4 tokenizer/BPE 文件缺失

未显式传入 `bpe_path` 时，当前 SAM3 构建器从包相邻的 `assets/bpe_simple_vocab_16e6.txt.gz` 加载词表。在当前安装布局中已补充为：

```text
/home/hirobot/anaconda3/envs/rpent/lib/python3.10/site-packages/assets/bpe_simple_vocab_16e6.txt.gz
```

如果重新安装 SAM3 wheel 后问题复现，应先确认该文件是否仍存在以及当前 `model_builder.py` 实际解析出的路径。

### 18.5 `top score ... is below min_score`

这是正常未找到结果。优先尝试：

1. 使用更明确的颜色、位置和容器关系描述；
2. 切换 `agentview` / `wrist`；
3. 使用目标内部点提示；
4. 最后再谨慎降低 `min_score`。

### 18.6 point 越界或目标错误

确认图像实际 shape，并牢记工具参数是 `[row,col]`。点应落在目标内部，而不是边缘或背景上。

### 18.7 `mask/world shape mismatch`

确认 RGB 和 world map 来自同一个 step、camera 和分辨率。不要手工混用 `_high` 与标准产物。当前实现故意不自动 resize。

### 18.8 `too few valid depth pixels`

说明 mask 内有效世界点少于默认阈值。可能原因包括目标过小、透明/反光材质、遮挡、无效深度或分割落在背景。应检查 overlay，并尝试另一相机或更稳定的目标区域。

### 18.9 RPC 超时或服务退出

依次检查：

1. `sam3_server.log` 中的模型加载异常；
2. checkpoint 和 BPE 文件；
3. GPU 显存、驱动和 PyTorch CUDA 兼容性；
4. endpoint 的协议、host 和 port；
5. `healthz` 是否可达。

Dashboard 模式中 SAM3 是 Session 级共享服务。服务启动失败时 Session 会发布 `sam3=failed` 并逆序停止已拥有的共享 daemon；服务 ready 后，后续顺序 TaskRun 复用同一个 `Sam3Client` 和模型进程。

## 19. 总结

RPent 中的 `sam3_image_model` 不只是一个神经网络构造函数，而是一条完整的机器人感知链路：

```text
本地 checkpoint
  -> CUDA 常驻单图模型
  -> 文本或单点提示
  -> 最高分二维 mask
  -> PNG/base64 RPC
  -> 客户端布尔 mask
  -> 同分辨率 world map
  -> 稳健 XYZ 中位数
  -> JSON + overlay 可审计产物
```

其核心设计原则是：模型进程隔离、输入协议严格、缓存有限可控、推理线程安全、二维与三维数据严格对齐，以及任何失败都尽量转成 Planner 可理解且可回退的结构化结果。