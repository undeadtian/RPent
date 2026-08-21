"""准备环境运行所需的持久化 resources 目录。

RPent 把任务记忆和参考资源发布为 HuggingFace dataset。普通 CLI 在构造 Planner 前
调用 ``ensure_resources(env_name)``，尝试同步该环境的子目录；同步失败不会阻止任务，
因为本地已有资源仍可能足够，且 memory 本身是可选增强项。
"""
from __future__ import annotations

import os
from pathlib import Path

from rpent.utils.config import get_resources_dir
from rpent.utils.logging import get_logger

# 可通过环境变量切换到私有镜像或兼容数据集，默认使用官方记忆仓库。
RESOURCES_HF_REPO = os.environ.get("RPENT_RESOURCES_HF_REPO", "RLinf/RPent-memory")

logger = get_logger("resources")


def ensure_resources(env_name: str) -> Path:
    """尽力同步指定环境资源并返回其本地目录。

    设置 ``HF_HUB_OFFLINE=1`` 时完全跳过网络请求，直接使用本地副本。在线模式的
    下载、鉴权或连接错误只记录 warning，不会让主任务启动失败。
    """
    resources_dir = get_resources_dir(env_name)

    if os.environ.get("HF_HUB_OFFLINE") == "1":
        return resources_dir

    try:
        # 延迟导入：离线模式以及仅导入 CLI 模块时不要求加载 huggingface_hub。
        from huggingface_hub import snapshot_download

        snapshot_download(
            repo_id=RESOURCES_HF_REPO,
            repo_type="dataset",
            # dataset 内按 <env_name>/... 组织，因此 local_dir 指向 resources 的父层。
            local_dir=str(resources_dir.parent),
            # 只同步当前环境，避免一次下载其他机器人环境的全部资源。
            allow_patterns=[f"{env_name}/**"],
        )
    except Exception as exc:
        logger.warning(
            "could not sync '%s' from '%s': %s; "
            "continuing with local files under %s",
            env_name,
            RESOURCES_HF_REPO,
            exc,
            resources_dir,
        )

    return resources_dir
