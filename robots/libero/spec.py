"""LIBERO 插件供 Dashboard 消费的静态声明。

该模块不创建环境或服务，只描述 Dashboard 可提交的任务参数、运行时状态组件和观测帧
通道。``EnvSpec.dashboard`` 将 :data:`LIBERO_DASHBOARD_SPEC` 交给通用状态机与前端：
任务声明负责命令解析和 TaskRun 目录命名，组件声明约束 ``RuntimeStatusEvent`` 的名称
及作用域，帧声明把环境观测投影到固定相机/腕部相机面板。

字典中的 command、模板、label 和兼容键都是跨模块协议值，而非面向开发者的说明文字；
展示层和状态机按这些值查找数据，因此语义说明放在注释中而不改动其内容。
"""

# suite 顺序同时作为命令补全的展示顺序；解析器也把它当作 suite 参数的允许集合。
LIBERO_SUITE_NAMES = (
    "libero_spatial",
    "libero_object",
    "libero_goal",
    "libero_90",
    "libero_object_task",
    "libero_object_swap",
    "libero_object_lan",
    "libero_goal_task",
    "libero_goal_swap",
    "libero_goal_lan",
    "libero_spatial_task",
    "libero_spatial_swap",
    "libero_spatial_lan",
    "libero_10",
    "libero_10_task",
    "libero_10_swap",
    "libero_10_lan",
)

LIBERO_DASHBOARD_SPEC = {
    # task 描述斜杠命令的语法。fields 的顺序对应三个位置参数；suggestions 既用于
    # 前端补全也用于后端允许值校验，integer/minimum 则由状态机在创建 TaskRun 前校验。
    "task": {
        "command": "/rpent-task",
        "usage": "/rpent-task <suite> <task> <seed>",
        "fields": (
            {"name": "suite", "suggestions": LIBERO_SUITE_NAMES},
            {"name": "task", "kind": "integer", "minimum": 0},
            {"name": "seed", "kind": "integer", "minimum": 0},
        ),
        # display 用于任务选择反馈；output_slug 经安全化后参与递增编号的任务目录名。
        "display": "{suite} / task {task} / seed {seed}",
        "output_slug": "{suite}_t{task}_s{seed}",
    },
    # name 必须与 RuntimeStatusEvent.component 一致。scope="task" 的 env 状态在每次
    # TaskRun 开始/结束时重置；未声明 scope 的 VLA/SAM3 默认为 Session 共享状态。
    "runtime_components": (
        {"name": "env", "label": "ENV", "scope": "task"},
        {"name": "vla", "label": "VLA"},
        {"name": "sam3", "label": "SAM3"},
    ),
    # name 是 Dashboard 内部帧通道；label 仅负责面板展示。若结果尚未提供统一
    # frames 映射，legacy_path_key 允许状态机从旧版结果字段回填对应图像路径。
    "frame_channels": (
        {
            "name": "camera",
            "label": "fixed camera",
            "legacy_path_key": "image_cam_path",
        },
        {
            "name": "wrist",
            "label": "wrist camera",
            "legacy_path_key": "image_wrist_path",
        },
    ),
}
