"""LIBERO system/user Prompt 正文片段的命名空间包。

``system.py`` 保存跨任务共享的角色、运行时契约、感知算法与工作流章节；``user.py``
保存单个 evaluation cell 的变量化摘要和启动指令。该包不重导出正文，也不负责渲染，
调用方应由 :mod:`robots.libero.prompt_bundle` 组装 ``PromptNode``，再通过统一渲染入口
替换运行变量。这样导入命名空间本身不会触发任何 LIBERO 服务或模型初始化。
"""
