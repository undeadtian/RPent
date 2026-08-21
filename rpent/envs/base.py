"""环境插件注册层：把 ``--env`` 名称映射到 ``robots.<name>`` 工厂。

具体机器人实现位于仓库顶层 ``robots/``，而不是 ``rpent`` 核心包中。每个环境包
只需公开两个约定函数：

- ``get_env_spec()``：返回 CLI 参数、Prompt 和运行时 hook；
- ``get_toolkit(...)``：用运行时客户端构造环境 Toolkit。

CLI 因而只依赖稳定的 ``EnvSpec``/``Toolkit`` 协议，无需硬编码 LIBERO。导入采用
延迟方式：列出 CLI help 或导入核心模块时，不会提前加载仿真器、CUDA 或模型。
"""
from __future__ import annotations

import importlib
import pkgutil
import sys
from typing import Any

from rpent.envs.env_spec import EnvSpec
from rpent.tools.toolkit import Toolkit
from rpent.utils.config import get_repo_root

# ``robots`` 是 ``rpent`` 的同级顶层包，普通 wheel 配置不一定把它安装进 site-packages。
# 将仓库根目录加入 sys.path 后，无论当前工作目录在哪里都能解析 robots.<name>。
_REPO_ROOT = str(get_repo_root())
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _resolve_env(name: str) -> Any:
    """延迟导入 ``robots.<name>`` 并返回环境模块。

    名称统一转为小写，与 CLI choices 和目录名保持一致。这里把底层
    ModuleNotFoundError 转成面向用户的 ValueError，避免 CLI 暴露导入实现细节。
    """
    if not name:
        raise ValueError("env name must be non-empty")
    env_name = name.lower()
    try:
        return importlib.import_module(f"robots.{env_name}")
    except ModuleNotFoundError as e:
        raise ValueError(f"unknown env: {env_name!r}") from e


def enumerate_envs() -> tuple[str, ...]:
    """扫描 ``robots`` 下可导入的环境子包，供 ``--env`` choices 使用。"""
    import robots

    # 只暴露目录包，忽略 robots/__init__.py 和以下划线开头的内部实现。
    return tuple(
        sorted(
            module.name
            for module in pkgutil.iter_modules(robots.__path__)
            if module.ispkg and not module.name.startswith("_")
        )
    )


def get_env_spec(name: str) -> EnvSpec:
    """调用环境包的 ``get_env_spec`` 工厂，取得静态扩展描述。"""
    return _resolve_env(name).get_env_spec()


def get_toolkit(name: str, **kwargs) -> Toolkit:
    """调用环境包的 Toolkit 工厂，绑定本次运行的客户端和事件 sink。

    Python 会缓存前面导入的环境模块，因此第二次 ``_resolve_env`` 不会重新执行
    环境模块初始化；它只是避免在核心注册层保存额外的全局模块状态。
    """
    return _resolve_env(name).get_toolkit(**kwargs)
