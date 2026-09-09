"""In-process streamable-HTTP MCP server that wraps a :class:`Toolkit`.

The Codex CLI accepts streamable-HTTP MCP servers via
``mcp_servers.<name>.url``. This module builds an MCP
:class:`~mcp.server.lowlevel.Server` and serves it through a
:class:`~mcp.server.streamable_http_manager.StreamableHTTPSessionManager`
on a background uvicorn thread — no separate subprocess, no second
``Toolkit`` instance.

Usage::

    server = HttpMcpServer(toolkit)
    server.start()  # binds 127.0.0.1 on a free port
    codex_url = server.url   # e.g. "http://127.0.0.1:54321/mcp/"
    ...
    server.stop()
"""
from __future__ import annotations

import asyncio
import socket
import threading
from typing import Any

import httpx

import uvicorn
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("mcp_http")

# MCP server 名称同时决定 Codex 配置键和工具 namespace：模型侧通常看到
# ``mcp__rpent__<tool>``，Toolkit 内部则始终使用不带 namespace 的短名称。
SERVER_NAME = "rpent"

# =============================================================================
# Codex -> HTTP MCP -> Toolkit 数据与并发边界
# =============================================================================
#
# 本模块不创建第二份 Toolkit，也不启动独立 MCP 子进程。HttpMcpServer 在 RPent
# 当前进程的 daemon thread 中运行 uvicorn；Codex binary 通过 localhost 的
# Streamable HTTP 请求访问它。一次工具调用的执行链如下：
#
#   Codex binary
#       -> HTTP JSON-RPC / MCP call_tool
#       -> uvicorn 线程中的 asyncio event loop
#       -> 默认 executor 工作线程
#       -> 同步 Toolkit.execute_tool(name, arguments)
#       -> ToolResult 文本/图片 block
#       -> MCP CallToolResult
#
# ``run_in_executor`` 只负责避免机器人 RPC、VLA 推理等同步操作阻塞 HTTP event loop；
# 同一个物理环境是否允许并发由 Toolkit 自己的 operation lock 再次约束。
#
# ASGI lifespan 管理 StreamableHTTPSessionManager 的启动和关闭；HttpMcpServer.start()
# 只有在真实 MCP initialize 请求成功后才返回 URL。调用方应始终在 finally 中 stop()，
# daemon thread 只是异常退出时不阻止解释器结束的最后保险。
# =============================================================================


def _toolkit_to_mcp_content(
    tr: Any,
) -> tuple[list[types.TextContent | types.ImageContent], bool]:
    """Translate a :class:`ToolResult` into MCP content blocks + isError."""
    # 标准 Toolkit ToolResult 暴露 Anthropic-shaped content_blocks；保留 getattr
    # fallback 可兼容测试替身或自定义 handler 返回值。
    blocks = getattr(tr, "content_blocks", None)
    if blocks is None:
        # 无结构化 block 时至少向模型返回字符串文本，而不是让整个 MCP 调用失败。
        return [types.TextContent(type="text", text=str(tr))], False

    out: list[types.TextContent | types.ImageContent] = []
    for block in blocks:
        block_type = block.get("type")
        if block_type == "text":
            out.append(types.TextContent(type="text", text=block.get("text", "")))
        elif block_type == "image":
            # Toolkit 图像 source 已是 base64 数据；这里只把 media_type 字段映射为
            # MCP ImageContent 使用的 mimeType，不重新读取或解码 artifact。
            src = block.get("source", {})
            out.append(
                types.ImageContent(
                    type="image",
                    data=src.get("data", ""),
                    mimeType=src.get("media_type", "image/png"),
                )
            )
        # 未知 block 类型不猜测协议含义，避免构造不合法 MCP content。
    result_dict = getattr(tr, "result", None)
    # Toolkit 通常把工具异常编码为 result["error"]，而不是向 MCP handler 抛异常。
    # isError=True 让 Codex 知道调用失败，同时仍可读取文本错误并自行恢复/重试。
    is_error = isinstance(result_dict, dict) and bool(result_dict.get("error"))
    return out, is_error


def _strip_mcp_prefix(name: str) -> str:
    """``mcp__rpent__mcp_list_dir`` -> ``mcp_list_dir`` ; passthrough."""
    prefix = f"mcp__{SERVER_NAME}__"
    if name.startswith(prefix):
        # Codex/SDK 可能把 namespace 前缀带回 call_tool；Toolkit 注册表只接受短名称。
        return name[len(prefix):]
    # 标准 MCP 客户端通常直接传 list_tools 返回的短名，因此保留原值。
    return name


def _build_asgi_app(toolkit: Toolkit) -> Any:
    """Build a raw ASGI3 app wrapping an MCP ``Server`` + streamable HTTP."""
    # Low-level Server 只负责 MCP 方法分派；HTTP session、ASGI 与线程由下方分别包装。
    mcp_app: Server = Server(SERVER_NAME, version="0.1.0")

    @mcp_app.list_tools()
    async def _list_tools() -> list[types.Tool]:
        tools: list[types.Tool] = []
        for spec in toolkit.get_tools_spec():
            # Toolkit spec 使用 input_schema，MCP types.Tool 字段名是 inputSchema。
            tools.append(
                types.Tool(
                    name=str(spec["name"]),
                    description=str(spec.get("description", "")),
                    # 无参数/旧 spec 缺 schema 时降级为任意 object，而不是拒绝注册。
                    inputSchema=spec.get("input_schema", {"type": "object"}),
                )
            )
        return tools

    @mcp_app.call_tool()
    async def _call_tool(
        name: str, arguments: dict[str, Any]
    ) -> types.CallToolResult:
        # 接受 Codex namespace 全名和标准 MCP 短名两种形状，统一映射到 Toolkit。
        lookup = _strip_mcp_prefix(name)
        # Toolkit.execute_tool 是同步阻塞接口，内部可能等待环境/VLA/SAM3 RPC。放入默认
        # executor 后 uvicorn event loop 仍可处理其他 HTTP、MCP session 和 shutdown。
        tr = await asyncio.get_running_loop().run_in_executor(
            None, # None 表示使用 loop 的默认 executor，也就是 ThreadPoolExecutor
            toolkit.execute_tool,
            lookup,
            # MCP 客户端可能对无参数工具传 None；Toolkit 契约始终要求 dict。
            arguments or {},
        )
        # ToolResult 的文本/图片与错误标志在唯一协议边界转换为 MCP Pydantic 类型。
        content, is_error = _toolkit_to_mcp_content(tr)
        return types.CallToolResult(content=content, isError=is_error)

    session_manager = StreamableHTTPSessionManager(
        app=mcp_app,
        # 每个请求独立，不依赖 MCP session ID；适合单个本地 Codex client 的短生命周期。
        stateless=True,
        # 直接返回 JSON response，ready probe 和 Codex 都无需 SSE 响应解析。
        json_response=True,
    )

    # minimal ASGI wrapper
    async def asgi_app(
        scope: dict[str, Any], receive: Any, send: Any
    ) -> None:
        if scope["type"] == "lifespan":
            # uvicorn 会先发送 startup，stop() 设置 should_exit 后再发送 shutdown。
            while True:
                event = await receive()
                if event["type"] == "lifespan.startup":
                    # session_manager.run() 的 context 生命周期必须覆盖全部 HTTP 请求；
                    # 进入成功后才向 uvicorn 确认 startup.complete。
                    async with session_manager.run():
                        await send({"type": "lifespan.startup.complete"})
                        # lifespan scope 独占等待关闭事件，不消费普通 HTTP scope 消息。
                        shutdown_event = await receive()
                        # Use an explicit check rather than ``assert`` —
                        # ``python -O`` strips asserts and would turn a
                        # protocol violation into a silent mis-handle.
                        if shutdown_event["type"] != "lifespan.shutdown":
                            # 非法时序退出 manager context，交由 ASGI server 观察 lifespan 结束。
                            break
                        await send({"type": "lifespan.shutdown.complete"})
                    # 退出 context 后 manager 已释放其 task group/session 资源。
                    break
                elif event["type"] == "lifespan.shutdown":
                    # 兼容尚未收到 startup 就直接关闭的 ASGI 生命周期。
                    await send({"type": "lifespan.shutdown.complete"})
                    break
            return
        if scope["type"] == "http":
            # JSON-RPC 解析、initialize/list_tools/call_tool 分派均由 MCP manager 完成。
            await session_manager.handle_request(scope, receive, send)
        # 该应用只服务 lifespan 与 HTTP；websocket 等 scope 不做处理。

    return asgi_app


def _pick_free_port(host: str) -> int:
    # 绑定端口 0 让内核选择当前可用端口；socket 退出 context 后立即释放该临时占用。
    # start() 到 uvicorn 真正 bind 之间理论上仍有很小 TOCTOU 窗口，适合本地短生命周期服务。
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return int(s.getsockname()[1])


def _wait_for_ready(url: str, *, timeout_s: float) -> None:
    """POST an MCP ``initialize`` request, retrying connection failures."""
    # 不只探测 TCP 端口，而是执行真实 MCP initialize，确保 ASGI lifespan、session
    # manager 和 JSON-RPC 分派都已经可用。
    body = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            # 显式使用当前兼容的 MCP 协议版本，与 Codex 后续协商使用同一服务能力。
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "hc", "version": "0"},
        },
    }
    # retries 只处理连接层启动竞态；非成功 MCP 响应在下方立即作为 ready 失败。
    transport = httpx.HTTPTransport(retries=10)
    with httpx.Client(
        transport=transport,
        timeout=httpx.Timeout(timeout_s, connect=2),
    ) as c:
        resp = c.post(url, json=body, headers={"Accept": "application/json"})
        # 错误只截取前 200 字符，避免服务器 HTML/traceback 充斥日志。
        body_preview = resp.text[:200]
        if not (resp.is_success and "result" in resp.json()):
            raise RuntimeError(
                f"HttpMcpServer not ready: code: {resp.status_code}; content: {body_preview}"
            )


class HttpMcpServer:
    """Run an in-process streamable-HTTP MCP server over a Toolkit.

    The server runs on a background daemon thread with its own asyncio loop;
    callers must invoke :meth:`start` before reading :attr:`url` and
    :meth:`stop` to release the port before the process exits.
    """

    def __init__(
        self,
        toolkit: Toolkit,
        *,
        host: str = "127.0.0.1",
        port: int = 0,
        path: str = "/mcp",
    ) -> None:
        # 保存同一进程内的 Toolkit 引用；HTTP handler 不复制环境客户端或状态。
        self._toolkit = toolkit
        # 默认仅监听 loopback，避免未经认证的机器人工具暴露到外部网络。
        self._host = host
        # port=0 在构造时选择一个临时可用端口；显式端口便于测试/固定配置。
        self._port = port or _pick_free_port(host)
        # 内部统一保存带前导斜杠的 ASGI 路径，url 属性再补结尾斜杠。
        self._path = path if path.startswith("/") else f"/{path}"
        # start 后分别指向 uvicorn 控制对象和承载其独立 asyncio loop 的线程。
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        # Codex config 需要完整 Streamable HTTP endpoint；服务启动前也可计算地址。
        return f"http://{self._host}:{self._port}{self._path}/"

    def start(self, *, ready_timeout_s: float = 30.0) -> str:
        """Launch uvicorn in a background thread and block until it's serving."""
        if self._thread is not None:
            # 同一实例重复 start 是幂等操作，不创建第二个 server/thread。
            return self.url

        # 每次首次启动为当前 Toolkit 构造独立 low-level MCP server 与 session manager。
        app = _build_asgi_app(self._toolkit)

        config = uvicorn.Config(
            app,
            host=self._host,
            port=self._port,
            # 关闭 access log、降低 uvicorn 噪声，工具和生命周期由 RPent 日志记录。
            log_level="warning",
            access_log=False,
            # 强制启用 lifespan，确保 session_manager.run() 覆盖所有请求。
            lifespan="on",
        )
        self._server = uvicorn.Server(config)

        self._thread = threading.Thread(
            target=self._server.run,
            name="mcp-http-server",
            # daemon 防止异常遗漏 stop 时阻塞解释器退出；正常路径仍必须显式 stop。
            daemon=True,
        )
        self._thread.start()

        # 阻塞调用方直到完整 MCP initialize 成功，Codex 才能安全读取该 URL 配置。
        _wait_for_ready(self.url, timeout_s=ready_timeout_s)
        logger.info("HttpMcpServer ready at %s", self.url)
        return self.url

    def stop(self, *, timeout_s: float = 5.0) -> None:
        if self._server is not None:
            # uvicorn 在自己的 loop 中观察 should_exit，发送 lifespan.shutdown 并优雅退出。
            self._server.should_exit = True
        if self._thread is not None:
            # 给活动 HTTP/MCP 请求和 session manager 清理一个有限宽限期。
            self._thread.join(timeout=timeout_s)
            if self._thread.is_alive():
                # Python 无法安全强杀线程；记录警告后依赖 daemon 语义，不执行危险终止。
                logger.warning(
                    "HttpMcpServer thread did not exit within %.1fs; "
                    "leaving references intact",
                    timeout_s,
                )
        # 无论线程是否按时退出，当前实现都会清空控制引用，使重复 stop 成为 no-op；
        # 若上方已告警，后台 daemon 可能仍短暂存活，因此不应立即复用同一端口。
        self._server = None
        self._thread = None
