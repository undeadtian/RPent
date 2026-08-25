"""RPent 子进程服务共用的传输无关 RPC 协议与服务端 Facade。

本模块不负责 JSON、pickle、socket 或 HTTP 的具体编解码，而是定义两种传输都必须
遵守的上层约定：

* 客户端以 ``method + args + kwargs`` 发起调用，并通过 :class:`RpcClient` Protocol
  让环境/VLA/SAM3 领域客户端不依赖具体传输；
* 服务端统一返回 ``{"ok": True, "result": ...}`` 或
  ``{"ok": False, "error": ..., "traceback": ...}`` envelope；
* :class:`RpcFacade` 提供内建 ``healthz``/``shutdown``、业务分派串行化、父进程
  存活监控和服务清理，子类只实现具体业务方法；
* :func:`wait_for_ready` 把“端口已经可连接”提升为“RPC healthz 已成功”，并可结合
  :class:`~rpent.utils.daemon.ProcessDaemon` 及时发现本地服务启动崩溃。

具体传输位于 ``http_rpc.py`` 与 ``socket_rpc.py``。两者负责请求/响应帧和 NumPy
序列化，但都会调用本模块的错误 envelope 辅助函数，因此上层看到相同的成功结果和
:class:`RpcError`。
"""
from __future__ import annotations

import threading
import time
from typing import TYPE_CHECKING, Any, Literal, Protocol

from rpent.utils.logging import get_logger

if TYPE_CHECKING:
    # 仅用于 wait_for_ready 的类型提示。运行时延迟导入可避免 rpc <-> daemon
    # 模块之间形成不必要的双向导入。
    from rpent.utils.daemon import ProcessDaemon

logger = get_logger("rpc")


class RpcError(RuntimeError):
    """表示远端业务异常、传输错误或非法响应的统一客户端异常。

    具体客户端会把连接失败、解码失败和服务端 ``ok=False`` 都转换到这一类型，
    调用方因此无需了解 HTTP/socket 的底层异常类别。异常文本以 RPC 方法名开头；
    若服务端 envelope 带 traceback，则额外保存在 ``server_traceback`` 中供日志和
    调试使用，但不会自动拼进简短异常消息。
    """

    def __init__(self, method: str, message: str, *, traceback: str | None = None):
        """保存失败方法、可读错误信息及可选的服务端 traceback。"""
        super().__init__(f"{method}: {message}")
        self.method = method
        self.server_traceback = traceback


class RpcClient(Protocol):
    """任意 RPent 方法调用传输必须满足的结构化客户端协议。

    这是静态结构协议，不提供实现，也不要求具体客户端继承。``HttpRpcClient`` 与
    ``SocketRpcClient`` 只要具有同签名的 :meth:`call`，就可注入 ``VLAClient``、
    ``LiberoEnvClient``、``Sam3Client`` 或 :func:`wait_for_ready`。领域层由此只关心
    RPC 方法和 Python 数据结构，不关心 wire format。
    """

    def call(
        self,
        method: str,
        args: tuple = (),
        kwargs: dict | None = None,
        *,
        timeout_s: float | None = None,
    ) -> Any:
        """调用远端方法并返回已解包的 ``result``。

        Args:
            method: 协议方法名，例如 ``healthz``、``env.step`` 或 ``predict``。
            args: 远端调用的位置参数元组。
            kwargs: 远端调用的关键字参数；``None`` 等价于空字典。
            timeout_s: 单次请求超时。``None`` 表示使用具体传输的默认值。

        Returns:
            服务端成功 envelope 中的 ``result``，类型由具体 RPC 方法决定。

        Raises:
            RpcError: 传输失败、响应无法解码、envelope 非法或远端业务调用失败。
        """


def make_error_response(exc: Exception) -> dict:
    """把当前捕获的异常转换成传输层统一的失败响应 envelope。

    HTTP 与 socket handler 都在 ``except`` 块内调用本函数。``format_exc`` 会记录
    当前异常链的完整服务端 traceback，便于客户端诊断模型、环境或参数错误；异常
    对象本身不直接跨进程序列化，只传输其字符串表示。

    Args:
        exc: 服务端 dispatch 或请求处理过程中捕获的异常。

    Returns:
        含 ``ok=False``、简短 ``error`` 和服务端 ``traceback`` 的字典。
    """
    # 局部导入只在错误路径加载 traceback 模块，并明确 format_exc 依赖当前异常上下文。
    import traceback as _tb
    return {"ok": False, "error": str(exc), "traceback": _tb.format_exc()}


def check_response(response: Any, method: str) -> Any:
    """校验并解包服务端响应，失败时统一抛出 :class:`RpcError`。

    合法响应必须首先是字典。``ok`` 缺失或为假时视为失败，并保留服务端 error 与
    traceback；成功时只返回 ``result`` 字段，使领域客户端不必重复处理 envelope。
    成功响应未提供 ``result`` 时按 ``dict.get`` 语义返回 ``None``，适用于无返回值
    的 RPC。

    Args:
        response: 具体传输完成反序列化后的 Python 对象。
        method: 原始方法名，用于给客户端异常补充调用上下文。

    Returns:
        成功 envelope 的 ``result``。

    Raises:
        RpcError: 响应不是字典，或者远端返回了失败 envelope。
    """
    if not isinstance(response, dict):
        # 非 envelope 往往说明协议版本不一致或传输解码到了错误对象。
        raise RpcError(method, f"bad response type: {type(response).__name__}")
    if not response.get("ok"):
        raise RpcError(
            method,
            str(response.get("error", "<no error message>")),
            traceback=response.get("traceback"),
        )
    return response.get("result")


def wait_for_ready(
    client: RpcClient,
    *,
    timeout_s: float = 300.0,
    poll_interval_s: float = 0.5,
    daemon: "ProcessDaemon | None" = None,
) -> None:
    """轮询 ``healthz``，直到服务可用、子进程早退或总等待时间耗尽。

    每次健康探针使用独立的 1 秒请求超时，失败后等待 ``poll_interval_s`` 再重试。
    模型权重加载或仿真初始化期间出现连接拒绝/请求超时是正常现象，函数会保存最近
    一次异常用于最终诊断。

    本地启动服务时传入 ``daemon``：每轮探针前先非阻塞查询子进程退出码，如果服务
    已在 ready 前崩溃，立即携带退出码和最后一次 healthz 错误失败，不必空等完整
    ``timeout_s``。连接外部 endpoint 时没有进程所有权，``daemon=None``，只能依赖
    健康探针和总超时。

    Args:
        client: 已指向目标 endpoint 的任意 :class:`RpcClient` 实现。
        timeout_s: 整体 readiness 等待上限，默认允许大型模型加载 5 分钟。
        poll_interval_s: 探针失败后的重试间隔。
        daemon: 可选的本地子进程句柄，仅用于提前检测退出，不负责停止进程。

    Raises:
        RuntimeError: 本地 daemon 在报告 ready 前已经退出。
        TimeoutError: 截止时间前始终没有成功完成 ``healthz``。
    """
    deadline = time.time() + timeout_s
    last_err: Exception | None = None
    while time.time() < deadline:
        if daemon is not None:
            rc = daemon.poll()
            if rc is not None:
                # 第一次探针前就崩溃时没有 last_err，使用明确占位文本区分该场景。
                detail = last_err if last_err is not None else "no healthz attempt yet"
                raise RuntimeError(
                    f"{daemon.name} exited with code {rc} before becoming "
                    f"ready; check its log. last healthz error: {detail}"
                )
        try:
            # 固定短超时防止单个连接占满整体启动预算；返回内容无需检查，调用成功
            # 即表示传输层和 Facade 内建健康方法都已可用。
            client.call("healthz", timeout_s=1.0)
            return
        except Exception as exc:
            # 启动期间统一容忍连接、超时和 RpcError；最后一次错误会进入超时消息。
            last_err = exc
            time.sleep(poll_interval_s)
    raise TimeoutError(
        f"server did not become ready within {timeout_s:.0f}s: {last_err}"
    )


class RpcFacade:
    """子进程 RPC 服务的传输无关生命周期与分派基类。

    子类只需实现 :meth:`_dispatch`，把业务方法名路由到环境、VLA 或 SAM3 实现。
    基类统一负责：

    * 内建 ``healthz`` 和 ``shutdown``，不让各服务重复实现；
    * 选择 HTTP 或 pickle-framed socket server，并打印实际监听地址；
    * 用一把锁串行化业务调用，保护环境状态、模型缓存和 GPU 推理等非线程安全对象；
    * 可选监听父进程 stdin pipe，在父进程死亡后设置 shutdown event；
    * 在所有退出路径关闭监听循环和 server socket。

    ``healthz`` 不取得业务锁，可用于确认服务循环仍可响应；普通业务调用持有锁直到
    ``_dispatch`` 返回。显式 ``shutdown`` 会先取得同一把锁，因此会等待正在执行的
    primitive/推理完成，而不是在共享状态更新到一半时关闭服务。

    示例::

        class MyFacade(RpcFacade):
            def _dispatch(self, method, args, kwargs):
                if method == "hello":
                    return "world"
                raise ValueError(f"unknown RPC method: {method!r}")

        MyFacade().serve(transport="http", host="127.0.0.1", port=0)
    """

    def __init__(self) -> None:
        """创建控制 ``serve`` 阻塞生命周期的线程安全关闭事件。"""
        self._shutdown_event = threading.Event()

    def _dispatch(self, method: str, args: tuple, kwargs: dict) -> Any:
        """分派一个业务 RPC；具体 Facade 子类必须覆盖本方法。

        ``healthz`` 与 ``shutdown`` 已在 :meth:`serve` 的包装分派器中截获，子类无需
        处理。未知业务方法通常应抛出 ``ValueError``，传输 handler 会把它包装成
        ``ok=False`` envelope 返回客户端。
        """
        raise NotImplementedError

    def serve(
        self,
        *,
        transport: Literal["socket", "http"],
        host: str,
        port: int,
        parent_watch: bool = False,
    ) -> None:
        """绑定传输端口并阻塞服务，直到收到关闭请求或父进程死亡。

        调用顺序为：延迟导入传输实现；创建线程安全 dispatch wrapper；实例化并绑定
        server；公布实际地址；可选启动父进程监控；在 daemon thread 中运行
        ``serve_forever``；当前线程等待 shutdown event；最后关闭服务循环和 socket。

        ``port=0`` 时由操作系统选择实际端口，公布的是 ``server_address`` 中的绑定
        结果。监听 ``0.0.0.0`` 时，日志中的客户端 URL 改写成可连接的
        ``127.0.0.1``，但 server 仍监听全部 IPv4 接口。

        Args:
            transport: ``http`` 或 ``socket``；两种 server 暴露相同生命周期接口。
            host: 服务监听地址。
            port: 服务监听端口，``0`` 表示让操作系统自动分配。
            parent_watch: 是否把 stdin EOF 视为父进程死亡并自动关闭服务。
        """
        # 延迟导入避免 rpc.py 与传输实现之间的模块级循环：传输模块也会导入本文件
        # 的 check_response/make_error_response。
        from rpent.utils.daemon import watch_parent_death
        from rpent.utils.http_rpc import HttpRpcServer
        from rpent.utils.socket_rpc import SocketRpcServer

        # HTTP/socket server 都可能为每个请求创建线程；共享业务对象必须串行访问。
        _lock = threading.Lock()

        def dispatch(method: str, args: tuple, kwargs: dict) -> Any:
            """处理内建控制方法，并在互斥锁内调用子类业务分派器。"""
            if method == "healthz":
                # 健康探针只确认 Facade 已绑定且请求线程可响应，不进入具体模型/环境。
                return {"status": "ok"}
            if method == "shutdown":
                # 等待当前业务调用释放锁后再唤醒 serve 主线程，避免半途拆除服务。
                with _lock:
                    self._shutdown_event.set()
                return {"ok": True}
            with _lock:
                # 环境 step、模型推理和处理器缓存都通过同一临界区顺序执行。
                return self._dispatch(method, args, kwargs)

        # Literal 为静态调用方限定两种值；运行时按现有规则，http 之外选择 socket。
        server_cls = HttpRpcServer if transport == "http" else SocketRpcServer
        # 构造 server 时立即绑定地址；两种实现都保存 dispatch 并提供相同控制接口。
        server = server_cls((host, port), dispatch)
        bound_host, bound_port = server.server_address
        # 0.0.0.0 是监听通配地址，不是客户端应直接连接的目标，因此日志改用回环地址。
        client_host = "127.0.0.1" if bound_host == "0.0.0.0" else bound_host
        url = f"{transport}://{client_host}:{bound_port}"
        # stdout 的 flush 输出会进入 ProcessDaemon 日志，也方便手工启动时立即看到地址。
        print(f"RPC server listening on {url}", flush=True)
        logger.info("RPC server listening on %s", url)

        if parent_watch:
            # ProcessDaemon 以 stdin=PIPE 启动子进程；父进程退出导致 EOF，回调只需设置
            # Event，即可复用与显式 shutdown 相同的 finally 清理路径。
            watch_parent_death(self._shutdown_event.set)
        try:
            # serve_forever 必须在另一线程运行：当前线程等待 Event，随后才能调用
            # server.shutdown()；在 serve_forever 自身线程调用 shutdown 会形成死锁。
            threading.Thread(target=server.serve_forever, daemon=True).start()
            self._shutdown_event.wait()
        finally:
            # 无论显式 shutdown、父进程死亡还是等待路径异常，都停止请求循环并释放端口。
            server.shutdown()
            server.server_close()


def parse_endpoint(endpoint: str) -> tuple[str, str, int]:
    """把 ``[protocol://]host:port`` 解析成 ``(protocol, host, port)``。

    省略协议时默认使用 ``http``。本函数只做轻量语法拆分和端口整数转换，不验证
    protocol 是否属于 ``http``/``socket``；环境 runtime 会根据允许的协议集合给出
    面向具体 CLI 参数的错误。实现按第一个冒号切分 host/port，因此面向当前 IPv4
    主机名格式，不支持未加方括号处理的 IPv6 literal。

    Args:
        endpoint: 例如 ``127.0.0.1:8000``、``http://host:8000`` 或
            ``socket://host:8000``。

    Returns:
        协议字符串、主机字符串和整数端口。

    Raises:
        ValueError: host/port 任一为空，或端口无法转换成整数。
    """
    if "://" in endpoint:
        # partition 只拆第一个协议分隔符，rest 继续交给 host/port 解析。
        protocol, _, rest = endpoint.partition("://")
    else:
        protocol, rest = "http", endpoint
    host, _, port = rest.partition(":")
    if not host or not port:
        raise ValueError(f"endpoint must be [protocol://]host:port, got {endpoint!r}")
    return protocol, host, int(port)


# 限定 ``from rpent.utils.rpc import *`` 的稳定公共接口；具体 HTTP/socket 类由各自
# 模块导出，避免核心协议层绑定某一种传输。
__all__ = [
    "RpcClient",
    "RpcError",
    "RpcFacade",
    "check_response",
    "make_error_response",
    "parse_endpoint",
    "wait_for_ready",
]
