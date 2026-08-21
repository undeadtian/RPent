"""环境插件提供给 CLI 的静态扩展协议。

``EnvSpec`` 只描述环境身份、Prompt 和生命周期 hook，不持有实时环境或模型对象。
这种设计让 ``rpent.cli.main`` 能以同一套流程运行不同机器人环境，同时避免核心
``rpent.envs`` 层反向依赖 Toolkit、RPC 传输或大型仿真依赖。

普通单任务 CLI 使用 ``init_runtime``；Dashboard 为复用重型模型服务而把生命周期
拆成 ``init_shared_runtime``（Session 级）与 ``init_task_runtime``（TaskRun 级）。
"""
from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rpent.dashboard.events import DashboardEventSink
from rpent.envs.prompt_bundle import PromptBundle

if TYPE_CHECKING:
    # 只用于类型提示，运行时导入会不必要地把进程管理层带入环境协议模块。
    from rpent.utils.daemon import ProcessDaemon


@dataclass(frozen=True)
class RunConfig:
    """环境根据最终 CLI 参数派生出的单次运行配置。

    Attributes:
        recipe_tag: 稳定的任务标识，用于 recipe/transcript 文件名。
        output_dir: 本次运行的日志、状态、图片和视频目录。
        prompt_vars: 渲染环境 system/user Prompt 时可替换的变量。
        task_desc: 写入 transcript 顶层、便于聚合评测的任务元数据。
    """

    recipe_tag: str
    output_dir: Path
    prompt_vars: dict[str, Any]
    task_desc: dict[str, Any]


@dataclass(frozen=True)
class EnvSpec:
    """环境级、非工具级扩展点的不可变集合。

    Toolkit schema 和 handler 不放在这里，而由环境 ``get_toolkit`` 工厂负责；
    EnvSpec 只提供 CLI 编排在不同生命周期阶段必须调用的 hook。
    """

    # 环境注册名，通常与 robots/<name> 目录一致。
    name: str

    # Python 定义的 system/user Prompt 工厂。
    prompts: PromptBundle

    # 第一阶段解析出 --env 后，用它向共享 ArgumentParser 注入环境专属参数。
    add_cli_args: Callable[[argparse.ArgumentParser, bool], None]

    # 对完整 argparse Namespace 做环境校验，并生成 RunConfig。
    parse_config: Callable[[argparse.Namespace], RunConfig]

    # Dashboard Session 级服务，例如可跨多个任务复用的 VLA/SAM3。
    init_shared_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]

    # Dashboard 每个 TaskRun 独占的运行时，例如必须全新创建的 env_server。
    init_task_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]

    # 普通 CLI 一次性初始化全部服务，返回 owned daemons 和 primitive 构造参数。
    init_runtime: Callable[
        [argparse.Namespace, Path, DashboardEventSink],
        tuple[list["ProcessDaemon"], dict[str, Any]],
    ]

    # 可选 Dashboard UI 描述；为 None 表示该环境不支持 Dashboard 控制。
    dashboard: dict[str, Any] | None = None
