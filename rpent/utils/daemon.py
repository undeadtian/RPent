"""管理 RPent 本地 RPC 服务子进程及其父进程存活通知。

RPent 的 env/VLA/SAM3 服务通常由 CLI 在独立 Python 进程中启动。本模块提供
父子进程两侧互相配合的最小生命周期机制：

* 父进程使用 :class:`ProcessDaemon` 创建服务，把子进程 ``stdin`` 接到一条匿名
  pipe，并将 stdout/stderr 统一写入服务日志；
* 子进程在启用 ``--parent-watch`` 时调用 :func:`watch_parent_death`，后台阻塞读取
  stdin。当父进程正常退出或崩溃、pipe 写端被操作系统关闭后，读取返回 EOF，子进程
  即可执行 shutdown 回调，避免遗留占用端口或 GPU 显存的孤儿服务；
* 服务是否真正 ready 不由本模块判断。调用方启动后通过 RPC ``healthz`` 轮询，
  同时调用 :meth:`ProcessDaemon.poll` 检测模型加载或环境初始化期间的提前崩溃。

``ProcessDaemon`` 只管理由当前运行亲自启动的进程。连接用户提供的外部 endpoint
时不应创建该对象，否则清理阶段可能错误关闭不属于当前运行的服务。
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading
from typing import Callable

from rpent.utils.config import get_repo_root
from rpent.utils.logging import get_logger

logger = get_logger("daemon")


# ---------------------------------------------------------------------------
# 子进程侧：通过 stdin pipe 感知父进程死亡
# ---------------------------------------------------------------------------


def watch_parent_death(on_death: Callable[[], None]) -> None:
    """在后台等待 stdin EOF，并在 EOF 到达后调用一次 ``on_death``。

    :class:`ProcessDaemon` 以 ``stdin=subprocess.PIPE`` 启动服务。父进程持有 pipe
    写端，子进程在这里读取其标准输入：只要父进程仍存活且未关闭 pipe，``read``
    就持续阻塞；父进程退出后，操作系统关闭写端，``read`` 返回 EOF，随后触发关闭
    回调。这一机制不依赖父进程 PID 轮询，也能覆盖崩溃和强制退出。

    读取运行在 daemon thread 中，不阻塞 RPC 服务主线程，且不会单独阻止子进程
    退出。读取异常同 EOF 一样进入回调，以“父连接已不可用”作为保守处理。若服务
    直接从交互终端启动，stdin 通常保持打开并等待用户输入；若 stdin 已关闭或重定向
    到 ``/dev/null``，回调会立即执行。

    Args:
        on_death: 检测到 stdin EOF 或读取失败后执行的无参清理函数。该回调由后台
            线程调用，内部应自行处理线程安全和异常。
    """

    def _watch() -> None:
        """阻塞消费 stdin，直到 pipe 关闭，再把控制权交给服务关闭回调。"""
        try:
            # 不需要解释输入内容；父进程从不写数据，pipe 本身只作为存活租约。
            # 无长度 read 会一直等待 EOF，避免短轮询占用 CPU。
            sys.stdin.buffer.read()
        except Exception:
            # 标准输入不可读与 EOF 的含义相同：父子存活通道已经失效，应触发清理。
            pass
        on_death()

    # daemon=True 确保服务通过其他路径结束时，无需等待这个阻塞读取线程退出。
    threading.Thread(target=_watch, daemon=True).start()


# ---------------------------------------------------------------------------
# 父进程侧：端口选择和服务子进程生命周期
# ---------------------------------------------------------------------------


def pick_free_port(host: str = "127.0.0.1") -> int:
    """让操作系统为 ``host`` 分配一个当前空闲的 IPv4 TCP 端口。

    函数临时绑定端口 ``0``，读取内核选出的实际端口后立即关闭 socket。返回端口
    只是后续启动服务的候选值，并没有被持续预留；从临时 socket 关闭到子进程真正
    bind 之间存在较小的 TOCTOU 竞争窗口。RPent 默认只在本机回环地址上使用它，
    实践中冲突概率很低，真正的 bind 失败仍会通过子进程退出和 readiness 检查暴露。

    Args:
        host: 用于选择端口的 IPv4 监听地址，默认仅本机可访问的 ``127.0.0.1``。

    Returns:
        操作系统分配的 TCP 端口号。
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 端口 0 请求内核自动选择 ephemeral port；离开 with 后立即释放绑定。
        s.bind((host, 0))
        return int(s.getsockname()[1])


class ProcessDaemon:
    """封装一个由当前 RPent 运行拥有的后台服务子进程。

    本类负责进程创建、日志文件句柄和分级终止，不负责 RPC ready 判定。典型调用
    顺序是：构造实例 -> :meth:`start` -> ``wait_for_ready(client, daemon=self)`` ->
    使用服务 -> :meth:`stop`。``wait_for_ready`` 可通过 :meth:`poll` 在健康探针等待
    期间发现子进程已经退出，从而快速报告模型加载、参数或端口绑定错误。

    一个实例预期只调用一次 ``start``。``stop`` 可在正常结束和异常回滚路径调用；
    它只关闭本实例持有的进程与日志，不影响通过外部 endpoint 连接的服务。

    Args:
        name: 用于日志消息和故障定位的服务逻辑名，例如 ``env_server``。
        cmd: 直接传给 :class:`subprocess.Popen` 的 argv 列表，不经过 shell 解释。
        env: 子进程环境变量。未提供或为空时复制当前进程环境。
        log_path: stdout/stderr 的追加写入路径；为 ``None`` 时丢弃到 ``os.devnull``。
            调用方应提前创建日志文件的父目录。
        cwd: 子进程工作目录；缺省使用 RPent 仓库根目录。
    """

    def __init__(
        self,
        name: str,
        cmd: list[str],
        *,
        env: dict[str, str] | None = None,
        log_path: str | None = None,
        cwd: str | None = None,
    ) -> None:
        """保存启动配置，并初始化尚未启动的进程与日志句柄。"""
        self.name = name
        self.cmd = cmd

        # 为子进程保存独立环境字典，调用方可预先叠加 MUJOCO_GL、ROBOT_PLATFORM
        # 等服务专属变量。这里沿用原有 ``env or ...`` 语义：空字典也回退为父环境。
        self.subprocess_env = env or os.environ.copy()
        self.log_path = log_path
        self.cwd = cwd

        # start 前 _proc/_log_f 均为空；stop 会把日志句柄恢复为 None，便于重复清理。
        self._proc: subprocess.Popen | None = None
        self._log_f = None

    def poll(self) -> int | None:
        """非阻塞查询子进程退出码，运行中或尚未启动时返回 ``None``。

        非 ``None`` 值表示子进程已经退出。readiness 轮询用它区分“模型仍在加载”
        和“服务已提前崩溃”，无需一直等到健康检查超时。由于启动前同样返回
        ``None``，调用方应遵循先 :meth:`start`、再轮询的生命周期。
        """
        return self._proc.poll() if self._proc is not None else None

    def start(self) -> None:
        """启动服务子进程并立即返回，不等待 RPC ready。

        stdout 与 stderr 合并到同一追加日志，便于按时间顺序诊断服务启动失败；没有
        ``log_path`` 时仍打开 ``os.devnull``，从而保持统一的文件句柄清理路径。
        stdin 使用 PIPE，但父进程不向其写入业务数据，它只作为
        :func:`watch_parent_death` 的存活信号。工作目录默认固定到仓库根目录，避免
        CLI 从其他目录启动时改变服务对相对路径的解释。

        文件打开或 ``Popen`` 失败会直接向调用方抛出；本方法不执行重试或健康检查。
        """
        # 追加模式保留同一路径中的既有启动诊断；未配置日志时静默丢弃服务输出。
        self._log_f = (
            open(self.log_path, "a") if self.log_path else open(os.devnull, "w")
        )
        self._proc = subprocess.Popen(
            self.cmd,
            # pipe 写端由父进程持有；父进程死亡后，子进程读端收到 EOF，触发
            # watch_parent_death 注册的关闭回调。
            stdin=subprocess.PIPE,
            # 合并两个输出流，避免分别读取 pipe 导致阻塞，也保持日志时序一致。
            stdout=self._log_f,
            stderr=subprocess.STDOUT,
            env=self.subprocess_env,
            cwd=self.cwd or get_repo_root(),
        )
        logger.info("%s spawned (pid=%s)", self.name, self._proc.pid)

    def stop(self, timeout: float = 15.0) -> None:
        """尽力停止子进程，并始终关闭父进程持有的日志文件。

        若进程仍在运行，先发送平台对应的 ``terminate`` 信号并等待最多 ``timeout``
        秒，让 RPC 服务有机会正常退出；超时后发送 ``kill`` 强制终止。尚未启动或已
        自行退出时跳过信号阶段。无论哪种情况，最后都尝试关闭日志句柄；关闭失败被
        忽略，保证异常清理不会遮蔽原始任务错误。

        Args:
            timeout: 发送温和终止信号后等待退出的最长秒数。
        """
        if self._proc is not None and self._proc.poll() is None:
            # 优先温和终止，使服务有机会释放监听 socket、CUDA 上下文等资源。
            self._proc.terminate()
            try:
                self._proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                # 超时说明服务没有响应终止信号，升级为强制 kill，避免清理流程卡住。
                self._proc.kill()
        if self._log_f is not None:
            try:
                self._log_f.close()
            except Exception:
                # 日志关闭失败不应覆盖 Agent/服务本身的失败原因。
                pass
            self._log_f = None
