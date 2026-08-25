"""环境与模型 RPC 边界使用的 pickle 长度前缀 TCP 传输。

每次 RPC 使用一条独立 TCP 连接，线上数据由一个请求帧和一个响应帧组成。帧格式为：

``[4 字节大端无符号 payload 长度][pickle payload]``

请求 payload 是含 ``method``、``args``、``kwargs`` 的字典；响应遵循
:mod:`rpent.utils.rpc` 的统一 envelope：成功为 ``ok/result``，失败为
``ok/error/traceback``。pickle 可以直接传输 NumPy 数组和嵌套 Python 对象，避免
把大数组转换成文本，但反序列化不可信 pickle 会执行任意代码。因此该传输只适用于
同一用户控制的可信本地进程，不能直接暴露给不可信网络或客户端；需要更明确数据边界
时应使用 ``http_rpc.py`` 的 JSON/NumPy 编码。

TCP 是字节流而不是消息协议，一次 ``read`` 不保证得到完整数据。``_read_exact``
负责拼接分段，``_read_frame``/``_write_frame`` 在其上维护请求和响应边界。服务端虽
使用线程化 TCP server 并发接收连接，但上层 :class:`rpent.utils.rpc.RpcFacade`
会用互斥锁串行化环境 step、模型推理等业务调用。
"""
from __future__ import annotations

import pickle
import socket
import socketserver
import struct
from collections.abc import Callable
from typing import Any

from rpent.utils.rpc import check_response, make_error_response

# 连接超时只约束 TCP 建连；请求超时在连接成功后约束发送/接收整个 RPC 帧。
DEFAULT_CONNECT_TIMEOUT_S = 10.0
DEFAULT_REQUEST_TIMEOUT_S = 30.0

# ``>I`` 表示网络字节序（大端）的 4 字节无符号整数。客户端与服务端必须使用同一
# 定义，理论可表示 0..2^32-1 字节的 payload；本可信内部协议不另设帧大小上限。
_LEN_PREFIX = struct.Struct(">I")


def _read_exact(reader, n: int) -> bytes:
    """从 file-like 字节流中精确读取 ``n`` 字节，否则报告中途断连。

    TCP 可能把一个帧拆成任意多个 packet，``reader.read(n)`` 也可能只返回部分数据，
    因此循环累计直到达到目标长度。若在完成前得到空字节串，表示对端已经关闭连接；
    错误消息会包含已读/期望字节数，便于区分长度前缀和 payload 截断。

    Args:
        reader: 提供 ``read(size)`` 的二进制输入流，通常是 socket ``makefile``。
        n: 本次协议阶段要求读取的确切字节数。

    Returns:
        长度严格为 ``n`` 的不可变 ``bytes``。

    Raises:
        ConnectionError: 对端在完整数据到达前关闭连接。
    """
    buf = bytearray()
    while len(buf) < n:
        # 每次只请求剩余字节，避免跨过当前帧边界消费下一段数据。
        chunk = reader.read(n - len(buf))
        if not chunk:
            raise ConnectionError(
                f"socket closed mid-frame (read {len(buf)}/{n} bytes)"
            )
        buf.extend(chunk)
    return bytes(buf)


def _read_frame(reader) -> Any:
    """读取一个长度前缀帧，并反序列化其中的 pickle 对象。

    先精确读取固定 4 字节前缀，再按其中声明的长度读取 payload。这里没有额外的
    大小限制或类型白名单，因为协议假设对端是同一用户启动的可信 RPent 进程；切勿
    对不可信输入调用本函数。

    Args:
        reader: 已连接 socket 对应的二进制输入流。

    Returns:
        pickle payload 还原出的任意 Python 对象。

    Raises:
        ConnectionError: 前缀或 payload 尚未完整到达时连接关闭。
        Exception: 长度解包或 pickle 反序列化失败时传播底层异常。
    """
    # 前缀自身也可能被 TCP 拆分，因此同样通过 _read_exact 读取。
    (length,) = _LEN_PREFIX.unpack(_read_exact(reader, _LEN_PREFIX.size))
    return pickle.loads(_read_exact(reader, length))


def _write_frame(writer, obj: Any) -> None:
    """把 Python 对象序列化成一个完整帧并立即刷新到底层 socket。

    使用当前 Python 支持的最高 pickle 协议，在可信同版本进程间获得较紧凑、高效的
    NumPy/Python 对象传输。长度前缀与 payload 一次交给缓冲流，``flush`` 确保请求或
    响应不会滞留在 ``makefile`` 用户态缓冲区中。

    Args:
        writer: 提供 ``write``/``flush`` 的二进制输出流。
        obj: 任意可被 pickle 序列化的请求、响应或领域数据。
    """
    body = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    writer.write(_LEN_PREFIX.pack(len(body)) + body)
    writer.flush()


class SocketRpcClient:
    """每个方法调用新建一条连接的 pickle-framed RPC 客户端。

    客户端不维护连接池：一次 :meth:`call` 完成建连、发送一个请求帧、读取一个响应
    帧并关闭连接。该模型实现简单，可让服务端按连接隔离超时和损坏帧，也避免跨调用
    残留字节；代价是每次 RPC 都有一次 TCP 握手。具体领域客户端仅依赖
    :class:`rpent.utils.rpc.RpcClient` Protocol，因此可在不改业务代码的情况下切换
    到 HTTP 传输。
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        connect_timeout_s: float = DEFAULT_CONNECT_TIMEOUT_S,
    ):
        """保存服务地址和 TCP 建连超时。

        Args:
            host: 服务端主机名或 IP 地址。
            port: 服务端 TCP 端口；构造时规范化为整数。
            connect_timeout_s: 仅用于 ``socket.create_connection`` 的建连上限，
                不限制服务端业务执行时间。
        """
        self.host = host
        self.port = int(port)
        self.connect_timeout_s = connect_timeout_s

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """通过一条临时 TCP 连接执行远端方法，并返回解包后的结果。

        请求会复制为稳定字典：位置参数规范化为 tuple，``kwargs=None`` 规范化为空
        dict。连接建立后，把 socket 超时切换为本次请求上限，该超时覆盖请求帧写入、
        服务端执行等待和响应帧读取；未指定时使用 30 秒默认值。

        响应交给 :func:`check_response`：远端业务异常转换成 ``RpcError``，成功则只
        返回 ``result``。建连、socket I/O 或 pickle 解码本身的异常按当前实现直接
        传播，调用方可由 :func:`rpent.utils.rpc.wait_for_ready` 在启动阶段统一重试。

        Args:
            method: 远端方法名。
            args: 位置参数。
            kwargs: 关键字参数，``None`` 表示空字典。
            timeout_s: 建连成功后的请求/响应超时；``None`` 使用默认值。

        Returns:
            成功响应 envelope 中的业务结果。
        """
        payload = {
            "method": method,
            "args": tuple(args),
            "kwargs": dict(kwargs or {}),
        }
        request_timeout_s = (
            DEFAULT_REQUEST_TIMEOUT_S if timeout_s is None else timeout_s
        )
        # create_connection 独立使用较短建连超时；成功后再设置业务请求超时。
        with socket.create_connection(
            (self.host, self.port), timeout=self.connect_timeout_s
        ) as sock:
            sock.settimeout(request_timeout_s)
            # makefile 提供带缓冲的二进制读写接口，供统一帧函数使用；嵌套 with
            # 确保文件包装器先关闭，随后关闭底层 socket。
            with sock.makefile("rwb") as f:
                _write_frame(f, payload)
                response = _read_frame(f)
        return check_response(response, method)


class _RequestHandler(socketserver.StreamRequestHandler):
    """处理一条 TCP 连接中的单个请求帧和单个响应帧。

    :class:`socketserver.ThreadingTCPServer` 为每个连接创建一个 handler 实例和请求
    线程。读取失败通常意味着客户端探测后提前断开、帧损坏或超时，此时直接关闭连接；
    成功读取后，payload/dispatch 的任何异常都会被包装成标准失败 envelope，尽量让
    正常等待的客户端获得远端诊断。
    """

    def handle(self) -> None:
        """读取请求、调用 server dispatch，并尽力写回统一响应。"""
        try:
            payload = _read_frame(self.rfile)
        except Exception:
            # 连一个完整请求都没有取得时，无法可靠识别 method，也没有可用的业务
            # envelope 上下文；静默结束连接，由客户端处理 EOF/解码异常。
            return
        try:
            method = payload["method"]
            args = payload.get("args") or ()
            kwargs = payload.get("kwargs") or {}
            # dispatch 属性由 SocketRpcServer.__init__ 注入；通常指向 RpcFacade.serve
            # 创建的包装器，后者处理 healthz/shutdown 并串行化普通业务调用。
            result = self.server.dispatch(method, args, kwargs)  # type: ignore[attr-defined]
            response: dict = {"ok": True, "result": result}
        except Exception as exc:
            # 包括 payload 结构错误、未知方法和业务异常；保留服务端 traceback。
            response = make_error_response(exc)
        try:
            _write_frame(self.wfile, response)
        except Exception:
            # 客户端可能已经超时并关闭连接。响应写失败只影响本次请求，不能终止
            # ThreadingTCPServer 或其他正在处理的连接。
            pass


class SocketRpcServer(socketserver.ThreadingTCPServer):
    """接收 pickle-framed 调用并转交统一 ``dispatch`` 的线程化 TCP server。

    该类刻意提供与 ``HttpRpcServer`` 相同的 ``server_address``、``serve_forever``、
    ``shutdown`` 和 ``server_close`` 接口，使 :class:`RpcFacade` 可只按 transport
    选择类而复用完整生命周期。传输层允许并发连接；是否并行业务执行由上层
    dispatch 决定，当前 Facade 使用互斥锁串行化。
    """

    # 允许服务重启时尽快重新绑定仍处于 TIME_WAIT 相关状态的本地地址。
    allow_reuse_address = True

    # 请求工作线程不会阻止 server 关闭或进程退出，适合由父进程管理的短生命周期服务。
    daemon_threads = True

    def __init__(
        self,
        server_address: tuple[str, int],
        dispatch: Callable[[str, tuple, dict], Any],
    ):
        """绑定监听地址，并保存供请求 handler 调用的业务分派函数。

        Args:
            server_address: ``(host, port)``；端口为 0 时由操作系统自动分配。
            dispatch: 接收 ``method, args, kwargs`` 并返回业务结果的可调用对象。
        """
        super().__init__(server_address, _RequestHandler)
        self.dispatch = dispatch
