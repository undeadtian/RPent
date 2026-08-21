"""Base class for agent tools.

``Toolkit`` is the agent-facing tool container. Subclasses can register tools
during ``__init__`` via :meth:`Toolkit.add_tool`; the planner calls the tools through :meth:`Toolkit.get_tools_spec` and
:meth:`Toolkit.execute_tool`.
"""
from __future__ import annotations

import base64
import json
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING, Any, ClassVar

from rpent.dashboard.events import DashboardEventSink, StepRecordEvent
from rpent.utils.templates import substitute

if TYPE_CHECKING:
    from rpent.tools.state import EnvState, StepRecord


@dataclass(slots=True)
class _ToolOperation:
    cancel_event: threading.Event = field(default_factory=threading.Event)
    done_event: threading.Event = field(default_factory=threading.Event)


class ToolCancelled(Exception):
    """Raised when an environment reaches a safe cancellation boundary."""


def readonly(func):
    """Mark a tool handler as not advancing environment state.

    Tool handlers capture a fresh observation (:meth:`Toolkit.get_env_state`)
    by default. Apply this marker to observational and file/IO tools that do
    not move the robot or otherwise change the environment.
    """
    func._readonly = True
    return func


def _is_readonly(handler: Callable[..., Any]) -> bool:
    """Whether ``handler`` was marked with :func:`readonly`."""
    target = handler
    while isinstance(target, partial):
        target = target.func
    target = getattr(target, "__func__", target)
    return bool(getattr(target, "_readonly", False))


@dataclass
class ToolResult:
    """Result of executing one tool call.

    Carries the raw result dict (for logging and finish-signal detection)
    alongside the Anthropic-shaped content blocks the LLM consumes.
    """

    name: str
    result: dict[str, Any]
    call_id: str | None = None

    content_blocks: list[dict[str, Any]] = field(
        default_factory=list, init=False, repr=False
    )
    is_finish: bool = field(default=False, init=False)

    #: Max bytes of the text block emitted in :attr:`content_blocks`.
    MAX_TEXT_BYTES_IN_RESULT: ClassVar[int] = 60000

    def __post_init__(self) -> None:
        self.content_blocks = self._build_content_blocks()
        self.is_finish = bool(
            isinstance(self.result, dict) and self.result.get("_finish")
        )

    def _build_content_blocks(self) -> list[dict[str, Any]]:
        """Build Anthropic-shaped content blocks (text + optional images).

        Strips image byte payloads from the text block and emits them as
        separate base64 image blocks so the LLM receives the state images as
        multimodal content.
        """
        result = self.result
        if not isinstance(result, dict):
            return [{"type": "text", "text": str(result)[:self.MAX_TEXT_BYTES_IN_RESULT]}]

        result_for_text = dict(result)
        image = result_for_text.pop("_image_bytes", None)
        image_cam = result_for_text.pop("_image_cam_bytes", None)
        image_wrist = result_for_text.pop("_image_wrist_bytes", None)
        text = json.dumps(result_for_text, indent=2, default=str)
        if len(text) > self.MAX_TEXT_BYTES_IN_RESULT:
            text = text[:self.MAX_TEXT_BYTES_IN_RESULT] + "\n[truncated]"

        blocks: list[dict[str, Any]] = [{"type": "text", "text": text}]

        def _add_image_bytes(data_bytes: bytes) -> None:
            data = base64.b64encode(data_bytes).decode("utf-8")
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": data,
                },
            })

        if image:
            _add_image_bytes(image)
        if image_cam:
            _add_image_bytes(image_cam)
        if image_wrist:
            _add_image_bytes(image_wrist)
        return blocks


class Toolkit:
    """Base toolkit: registers common tools and dispatches tool calls.

    Subclasses extend ``__init__`` (calling ``super().__init__()`` first)
    and register additional tools with :meth:`add_tool`. Env-specific
    subclasses receive their env/model/etc. as constructor arguments and
    build the underlying LiberoPrimitives in ``__init__``; the toolkit
    base class only contributes the common file/IO tools. Override
    :meth:`close` to release env-side primitives / servers at the end of the run.
    """

    def __init__(
        self,
        *,
        dashboard_events: DashboardEventSink,
        state: Any = None,
    ) -> None:
        self._tools: dict[
            str,
            tuple[dict[str, Any], Callable[..., Any]],
        ] = {}
        self._dashboard_events = dashboard_events
        self._state = state
        self._operation_lock = threading.Lock()
        self._active_operation: _ToolOperation | None = None
        self._register_common_tools()

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add_tool(
        self,
        name: str,
        spec: dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Register one tool under ``name`` with its schema and handler.

        Args:
            name: Tool name as the LLM sees it (e.g. ``"read_text_file"``).
            spec: Anthropic-shaped tool schema dict (``name``,
                ``description``, ``input_schema``).
            handler: Callable invoked with the tool's input kwargs; returns
                a result dict. Decorate read-only handlers with
                :func:`readonly`; all other handlers capture state.
        """
        self._tools[name] = (spec, handler)

    def _register_common_tools(self) -> None:
        """Register the file/IO tools shared by every run."""
        from rpent.tools import common

        for spec in common.TOOLS_SPEC:
            name = spec["name"]
            self.add_tool(name, spec, common.TOOL_HANDLERS[name])

    # ------------------------------------------------------------------
    # Planner-facing API
    # ------------------------------------------------------------------

    @property
    def state(self) -> EnvState:
        """Return the run's artifact and step store."""
        if self._state is None:
            raise RuntimeError("toolkit has no environment state")
        return self._state

    def get_tools_spec(self) -> list[dict[str, Any]]:
        """Return the tool schemas the LLM sees."""
        return substitute(
            [spec for spec, _ in self._tools.values()]
        )

    def execute_tool(self, name: str, input_dict: dict[str, Any]) -> ToolResult:
        """执行一个 Planner 工具调用，并统一处理串行化、异常和状态反馈。

        对 VLA 路径而言，handler 最终会进入 ``LiberoPrimitives.pi0_*``，再经
        ``VLAClient`` 调用模型服务。该方法本身与具体环境无关：它只负责找到
        handler，并在动作结束后要求环境 Toolkit 捕获一份新的可观测状态。
        """
        # LLM 只能调用注册表中存在的 schema；未知名称作为普通工具错误返回，
        # 不抛出到 Planner 外层，这样模型有机会修正工具名后继续任务。
        entry = self._tools.get(name)
        if entry is None:
            return ToolResult(name=name, result={"error": f"unknown tool: {name}"})
        _, handler = entry

        # 一个 Toolkit 对应一个物理环境。禁止并发动作可避免两个模型工具同时推进
        # 同一模拟器，也为 Dashboard 的取消操作提供唯一的 active operation。
        with self._operation_lock:
            if self._active_operation is not None:
                return ToolResult(
                    name=name,
                    result={"error": "another tool operation is still active"},
                )
            operation = _ToolOperation()
            self._active_operation = operation

        try:
            started = time.perf_counter()
            failed = False
            try:
                # handler 可能是只读感知函数，也可能同步等待 VLA RPC 和一整个环境
                # 动作块执行完毕；Planner 侧会在独立线程调用本方法，避免阻塞事件循环。
                result = handler(**input_dict)
            except TypeError as e:
                # 参数不符合 JSON schema/函数签名时，把实际输入回显给模型便于修正。
                result = {
                    "error": f"bad arguments for {name}: {e}",
                    "got": input_dict,
                }
                failed = True
            except ToolCancelled as e:
                # 取消是可恢复的工具结果，不应当作未捕获异常终止整个 Planner 会话。
                result = {
                    "error": str(e),
                    "code": "tool_cancelled",
                    "interrupted": True,
                }
                failed = True
            except Exception as e:
                # RPC、环境或 primitive 异常也返回给模型；traceback 同时用于诊断。
                result = {"error": str(e), "traceback": traceback.format_exc()}
                failed = True

            # 非 readonly 工具可能改变机器人/场景。无论 handler 成功还是失败，都要
            # 尝试抓取执行后的真实状态，防止模型基于调用前图像继续规划。
            if not _is_readonly(handler):
                elapsed_s = round(time.perf_counter() - started, 2)
                result_dict = result if isinstance(result, dict) else {"value": result}
                command = {"action": name, **input_dict}
                record: StepRecord | None = None
                try:
                    # 具体保存哪些 RGB-D、世界坐标和机器人状态由环境子类实现。
                    captured = self.get_env_state(
                        command=command,
                        result=result_dict,
                        elapsed_s=elapsed_s,
                    )
                except Exception as e:
                    # 状态采集失败时仍保留 primitive 原结果，避免二次错误完全遮蔽
                    # 首个执行结果。
                    captured = result_dict
                    captured["state_capture_error"] = str(e)
                    captured.setdefault(
                        "error", f"failed to capture state after {name}: {e}"
                    )
                    captured.setdefault("traceback", traceback.format_exc())
                else:
                    record = self._state.latest_record()
                result = captured

                # 若 handler 已失败，把原错误字段合并回状态快照；已有字段优先保留。
                if failed:
                    for key, value in result_dict.items():
                        result.setdefault(key, value)
                if record is not None:
                    self._publish_step(record)

            # ToolResult 把 dict 转成 Planner 可消费的文本块和可选 base64 图像块。
            return ToolResult(name=name, result=result)
        finally:
            # 即使 handler、状态采集或返回格式化异常，也必须释放 active operation，
            # 否则后续所有工具都会被误判为“仍有工具执行中”。
            with self._operation_lock:
                self._active_operation = None
                operation.done_event.set()

    def _publish_step(self, record: StepRecord) -> None:
        """Publish one recorded environment step to the dashboard sink."""
        self._dashboard_events.emit(
            StepRecordEvent(
                record=record,
                env_state=self._state,
                frame_artifacts=dict(getattr(type(self), "_FRAME_ARTIFACTS", {})),
            )
        )

    def get_env_state(
        self,
        *,
        command: dict[str, Any],
        result: dict[str, Any],
        elapsed_s: float,
    ) -> dict[str, Any]:
        """Capture and return the observation produced by a stateful tool."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Server lifecycle hooks (overridden by env toolkits)
    # ------------------------------------------------------------------

    def cancel_active_and_wait(self) -> None:
        """Request cancellation and wait for the active tool to return."""
        with self._operation_lock:
            operation = self._active_operation
            if operation is None:
                return
            operation.cancel_event.set()
        operation.done_event.wait()

    def raise_if_cancelled(self) -> None:
        """Raise at an environment-defined safe cancellation boundary."""
        with self._operation_lock:
            operation = self._active_operation
        if operation is not None and operation.cancel_event.is_set():
            raise ToolCancelled("tool operation interrupted")

    def close(self) -> None:
        """Release the env-side primitives / servers at end of run. Default: no-op."""

    def write_recipe(self, recipe_tag: str) -> str | None:
        """Write a replay recipe for this env, if supported."""
        return None
