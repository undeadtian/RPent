"""LIBERO 机器人原语、感知工具与状态产物实现。

本模块位于 Planner 工具调用与单环境 LIBERO 执行服务之间，主要承担四类职责：

* :class:`LiberoPrimitives` 缓存最近观测，把 Pi0.5/VLA 动作块或脚本化
  OSC 动作推进到环境，并按环境步录制诊断帧；
* 将 RGB-D、相机标定和机器人状态写入 :class:`~rpent.tools.state.EnvState`，
  预计算与图像逐像素对齐的世界坐标图；
* 提供 SAM3 分割、像素反投影和状态查看等只读工具，以及从成功轨迹导出
  可复用 recipe 的逻辑；
* 以 ``TOOLS_SPEC`` 声明 Planner 可见的工具 schema。schema 到 Python handler
  的绑定由 :class:`robots.libero.toolkit.LiberoToolkit` 完成。

除显式标记为 ``@readonly`` 的感知函数外，primitive 会推进模拟器；状态落盘与
Dashboard 发布由 ``LiberoToolkit`` 在动作结束后的统一生命周期中完成。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from robots.libero.env_client import LiberoEnvClient
from rpent.tools.state import EnvState, StepRecord
from rpent.tools.toolkit import readonly
from rpent.utils.logging import get_logger
from rpent.utils.sam3_client import Sam3Client
from rpent.utils.vla_client import VLAClient

logger = get_logger("libero")


def _normalize_xyz(xyz):
    """把 LLM 传入的三维坐标规范化为三个 ``float``。

    工具参数来自 JSON，因而这里只接受长度严格为 3 的列表或元组；尽早抛出
    可读错误可避免形状异常进入后续 NumPy 广播或 OSC 控制计算。
    """
    if not isinstance(xyz, (list, tuple)) or len(xyz) != 3:
        raise ValueError(
            'xyz must be a JSON array of three numbers, e.g. "xyz":[-0.05,0,0.3]'
        )
    return [float(v) for v in xyz]


class LiberoPrimitives:
    """封装单环境 LIBERO 客户端、VLA 策略与视觉分割服务。

    实例始终缓存最近一次策略观测及从 ``states`` 提取的末端位置、高度和夹爪
    开合代理量。所有动作入口共享这份缓存：``pi0_pick``/``pi0_doubled`` 临时把
    顶层任务描述替换为局部指令并执行 VLA chunk；``move_to``、姿态旋转和
    ``release`` 则直接向底层 OSC 控制器发送 7 维动作。

    录制缓冲区按“环境步”保存主相机帧。脚本动作由 :meth:`_step_env` 逐帧追加，
    VLA 动作块则要求环境返回 chunk 内全部观测后逐帧追加，因此 Dashboard 的
    单动作视频和最终 episode 视频共用同一条连续时间线。只读的 :meth:`segment`
    仅消费已落盘产物，不推进环境。

    Args:
        env: 单环境 LIBERO RPC 客户端，负责 reset、step、chunk_step 与渲染。
        model: 输出动作块的 VLA 客户端。
        sam3_client: 对既有 RGB 产物执行文本或点提示分割的客户端。
        check_cancelled: 在模型调用和环境推进的安全边界检查取消状态的回调。
    """

    def __init__(
        self,
        env: LiberoEnvClient,
        model: VLAClient,
        sam3_client: Sam3Client,
        check_cancelled: Callable[[], None],
    ):
        """保存服务依赖，并初始化最近观测缓存与 episode 帧缓冲区。"""
        self.env = env
        self.model = model
        self._sam3_client = sam3_client
        self._check_cancelled = check_cancelled

        # 最近观测是所有 primitive 的单一状态来源；派生字段避免每个控制循环重复
        # 解析 ``states``，同时确保成功判定和脚本控制读取的是同一环境时刻。
        self._last_obs = None
        self._last_obs_eef_pos = None
        self._last_obs_eef_z = None
        self._last_obs_gripper = None

        # 按环境步保存主相机帧。start/stop 控制一次完整 episode 的生命周期，
        # frame_slice 则让 Toolkit 从同一缓冲区截取某次工具调用对应的短片。
        self._recording = False
        self._frames = []

    def start_recording(self):
        """开始新的 episode 录制，并清空此前残留的帧。"""
        self._recording = True
        self._frames = []

    def record_frame(self, obs):
        """从策略观测提取主相机图像，并以连续内存布局追加一帧。

        连续数组可直接交给后续视频编码器，避免引用环境复用的观测缓冲区。
        """
        self._frames.append(np.ascontiguousarray(np.asarray(obs["main_images"])))

    def recorded_frame_count(self) -> int:
        """返回当前 episode 已录制的环境步帧数，供动作片段游标使用。"""
        return len(self._frames)

    def stop_recording(self) -> list[np.ndarray]:
        """停止录制、移交全部帧并清空内部缓冲区。

        返回浅复制的列表后再重置内部列表，使调用方保存 episode 视频时不受后续
        生命周期影响；帧数组本身保持不复制。
        """
        frames = list(self._frames)
        self._recording = False
        self._frames = []
        return frames

    def frame_slice(self, start: int) -> list[np.ndarray]:
        """从帧游标 ``start`` 起复制列表切片，用于生成单次动作视频。"""
        return list(self._frames[int(start):])

    def set_obs(self, obs):
        """更新最近策略观测及控制/成功判定所需的派生状态缓存。

        ``states[:3]`` 是末端世界坐标；夹爪没有直接的开口宽度字段，因此使用
        robosuite 2F-85 两个手指关节绝对值之和作为开合代理量：约 ``0.08`` 为
        张开，接近 ``0`` 为闭合。
        """
        self._last_obs = obs
        states_arr = np.asarray(obs["states"])
        self._last_obs_eef_pos = np.asarray(states_arr[:3], dtype=np.float32)
        self._last_obs_eef_z = float(self._last_obs_eef_pos[2])
        # robosuite 2F-85 的 qpos[6] 约在 [0, 0.04]，qpos[7] 约在
        # [-0.04, 0]；两者绝对值之和可作为手指间距代理量。
        gp = np.asarray(states_arr[6:8], dtype=np.float32)
        self._last_obs_gripper = float(abs(gp[0]) + abs(gp[1]))

    def reset(self):
        """重置远端环境、刷新状态缓存，并返回初始观测与环境信息。"""
        obs, info = self.env.reset()
        self.set_obs(obs)
        return self._last_obs, info

    def _step_env(self, action) -> None:
        """在取消检查之间执行一个环境动作，并同步缓存与录制帧。

        该入口是所有脚本化 OSC primitive 的共同推进路径，保证每一步动作之后的
        最新观测都可用于下一轮闭环控制；启用录制时恰好追加一帧。
        """
        self._check_cancelled()
        obs, _r, _t, _tr, _i = self.env.step(action)
        self.set_obs(obs)
        if self._recording:
            self.record_frame(obs)

    def _vlm_chunk(self, instruction: str):
        """执行一次 VLA 前向，并把返回的整个动作块推进到环境中。

        这是 primitive 层与 VLA RPC 层的唯一直接交界点：输入是当前单环境观测
        和一个局部子指令，``VLAClient`` 返回 ``[chunk, 7]`` 动作，随后
        ``LiberoEnvClient.chunk_step`` 顺序执行这些动作并返回最终观测。

        方法名保留了早期的 ``vlm`` 命名，但这里调用的是能够输出动作的 VLA。
        """
        # 取消只在安全边界检查：不要在模型请求已经发出或环境动作块执行到一半时
        # 无条件破坏共享状态。
        self._check_cancelled()
        original_task = self._last_obs.get("task_descriptions")
        try:
            # 顶层任务语言描述完整任务；每个 VLA primitive 可以临时覆盖为更局部的
            # 接触指令，例如“抓住黑色碗”。finally 会恢复原始任务描述。
            self._last_obs["task_descriptions"] = instruction

            # VLA/OpenPI 的观测处理器要求该键始终存在，即使当前没有额外视角。
            self._last_obs.setdefault("extra_view_images", None)

            # VLAClient 在这里完成图像 PNG/base64 编码和 RPC；服务端执行 Pi0.5
            # 前向，再把 [B=1, chunk, 7] 去掉 batch 维后返回。
            actions, _ = self.model.predict_action_batch(
                self._last_obs,
                mode="eval",
            )
            self._check_cancelled()

            if not self._recording:
                # 非录制模式允许环境客户端按配置只返回最终帧；若服务配置为返回
                # 全部帧，则显式选择动作块执行后的最后一帧作为当前观测。
                chunk_obs, _r, _t, _tr, _i = self.env.chunk_step(actions)
                obs = chunk_obs[-1] if self.env.return_all_frames else chunk_obs
            else:
                # 录制模式必须拿到动作块内每一步的观测，才能生成连续 episode 视频。
                chunk_obs, _r, _t, _tr, _i = self.env.chunk_step(
                    actions,
                    return_all_frames=True,
                )
                for obs in chunk_obs:
                    self.record_frame(obs)
                obs = chunk_obs[-1]

            # 后续成功判定和下一轮 VLA 前向都基于动作块结束后的最新状态。
            self.set_obs(obs)
            return self._last_obs
        finally:
            # 局部 VLA 指令只在这次前向期间生效，不能污染全局任务描述。
            if original_task is not None:
                self._last_obs["task_descriptions"] = original_task

    def pi0_pick(
        self,
        prompt: str,
        *,
        max_chunks: int = 24,
        lift_thresh: float = 0.05,
        gripper_closed_thresh: float = 0.06,
    ) -> dict:
        """用 ``prompt`` 作为局部 VLA 指令执行闭环抓取。

        每轮调用 :meth:`_vlm_chunk` 执行一个完整动作块，再从最新缓存读取末端高度
        与夹爪开口。成功不是简单使用全程最高点与最低点之差，而是要求：机械臂
        先相对起点下降至少 ``0.10`` 米进入抓取阶段，随后从最新最低点上升至少
        ``lift_thresh``，且当前夹爪代理开口小于 ``gripper_closed_thresh``。
        这可避免“仅向下接近物体”被误判成抬升成功。

        环境 ``terminated``/``truncated`` 或 ``max_chunks`` 会提前结束循环；只有
        官方 ``terminated`` 在环境结束分支中计为成功。返回值同时保留高度、夹爪
        和动作块数量诊断，供 Planner 结合图像复核。
        """
        instr = prompt
        start_z = self._last_obs_eef_z
        peak_z = start_z
        min_z = start_z
        # 只跟踪“最近一次最低点之后”的峰值：下降后重新上升才是抓取抬升信号。
        # 若使用全程 |peak - min|，机械臂刚到下降最低点时也可能被错误触发。
        post_min_peak_z = start_z
        min_grip = self._last_obs_gripper
        last_grip = min_grip
        descent_done = False
        success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            z = self._last_obs_eef_z
            grip = self._last_obs_gripper
            peak_z = max(peak_z, z)
            if z < min_z:
                min_z = z
                post_min_peak_z = z  # 出现更低点后，重新开始累计后续上升
            else:
                post_min_peak_z = max(post_min_peak_z, z)
            if (start_z - min_z) >= 0.10:  # 下降至少 10 cm，确认已进入抓取阶段
                descent_done = True
            min_grip = min(min_grip, grip)
            last_grip = grip
            ascended = (post_min_peak_z - min_z) >= lift_thresh
            closed = grip < gripper_closed_thresh
            if descent_done and ascended and closed:
                success = True
                break
            if self.env.terminated or self.env.truncated:
                success = self.env.terminated
                break

        return {
            "name": "pick",
            "instruction": instr,
            "success": success,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "peak_lift_m": post_min_peak_z - min_z,  # 最近最低点后的真实抬升量
            "min_gripper_opening": min_grip,
            "final_gripper_opening": last_grip,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            "diagnostics": {
                "start_eef_z": round(start_z, 4),
                "peak_eef_z": round(peak_z, 4),
                "min_eef_z": round(min_z, 4),
                "post_min_peak_z": round(post_min_peak_z, 4),
                "descent_m": round(start_z - min_z, 4),
                "post_min_ascent_m": round(post_min_peak_z - min_z, 4),
                "descent_done": descent_done,
                "lift_thresh": lift_thresh,
                "gripper_closed_thresh": gripper_closed_thresh,
            },
        }

    def pi0_doubled(
        self,
        prompt: str,
        *,
        max_chunks: int = 20,
    ) -> dict:
        """执行面向非抓取接触动作的闭环 Pi0.5 技能。

        典型用途是旋钮、炉灶开关、按钮或短距离推动。每轮执行一个 VLA chunk，
        直到官方任务谓词使环境 ``terminated``、环境 ``truncated``，或耗尽
        ``max_chunks``。该 primitive 没有访问物体真值位姿的私有成功判定，因此
        中间接触动作即使已经完成，只要整项 LIBERO 任务尚未终止，返回的
        ``success`` 仍为 ``False``；调用方应结合落盘图像和状态判断。
        """
        instr = prompt
        task_success = False
        chunks_used = 0

        for c in range(max_chunks):
            self._vlm_chunk(instr)
            chunks_used = c + 1
            if self.env.terminated or self.env.truncated:
                task_success = self.env.terminated
                break

        return {
            "name": "pi0_doubled",
            "instruction": instr,
            "success": task_success,
            "task_success": task_success,
            "contact_skill_executed": chunks_used > 0,
            "chunks_used": chunks_used,
            "max_chunks": max_chunks,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
            "diagnostics": {
                "mode": "contact_skill_success_by_termination",
                "success_meaning": (
                    "`success` mirrors official LIBERO task termination only; "
                    "for intermediate contact skills, inspect image/state evidence."
                ),
            },
        }

    def move_to(
        self,
        xyz,
        *,
        max_steps: int = 80,
        gripper: float = -1.0,
        step_clip: float = 0.025,
        tol: float = 0.012,
        action_scale: float = 0.05,
        target_yaw: float | None = None,
        yaw_step_clip: float = 0.10,
    ) -> dict:
        """用脚本化 OSC 闭环将末端伺服到世界坐标 ``xyz``。

        每个环境步重新读取缓存的末端位置，将世界坐标误差逐轴裁剪到
        ``step_clip``，再除以 ``action_scale`` 映射到控制器的 ``[-1, 1]``
        平移动作范围。到达 ``tol``、环境结束或耗尽 ``max_steps`` 时停止。

        7 维动作的前三维是位置增量，第 3--5 维是轴角旋转，第 6 维是夹爪命令；
        ``gripper=+1`` 用于持物，``-1`` 用于张开。可选 ``target_yaw`` 会在同一
        控制步内计算世界 Z 轴腕部偏航，但其余姿态保持由 OSC 控制器稳定。
        """
        target = np.asarray(_normalize_xyz(xyz), dtype=np.float32)
        traj = []
        for step in range(max_steps):
            cur = self._last_obs_eef_pos
            diff = target - cur
            dist = float(np.linalg.norm(diff))
            traj.append({
                "step": step,
                "eef_pos": [round(float(x), 4) for x in cur],
                "dist_to_target_m": round(dist, 4),
            })
            if dist < tol:
                break
            step_dxyz = np.clip(diff, -step_clip, step_clip)
            action = np.zeros(7, dtype=np.float32)
            action[:3] = step_dxyz / action_scale  # 映射后通常约落在 [-0.5, 0.5]
            action[:3] = np.clip(action[:3], -1.0, 1.0)
            if target_yaw is not None:
                # action[5] 是绕 Z 轴的轴角分量。世界偏航必须用
                # atan2(R[1, 0], R[0, 0]) 提取；对夹爪朝下的姿态
                # （R[2, 2]≈-1），as_euler('zyx')[0] 会落入另一欧拉角图表并
                # 返回相反符号，进而静默翻转控制方向。
                from scipy.spatial.transform import Rotation as _R
                q = self.env.raw_obs()["robot0_eef_quat"]
                _R_mat = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
                cur_yaw = float(np.arctan2(_R_mat[1, 0], _R_mat[0, 0]))
                err = (float(target_yaw) - cur_yaw + np.pi) % (2 * np.pi) - np.pi
                step_dyaw = float(np.clip(err, -yaw_step_clip, yaw_step_clip))
                action[5] = float(np.clip(step_dyaw / 0.10, -1.0, 1.0))
            action[6] = gripper
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final = self._last_obs_eef_pos
        return {
            "name": "move_to",
            "target_xyz": [float(x) for x in target],
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "steps_used": len(traj),
            "max_steps": max_steps,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    def rotate_wrist(
        self,
        *,
        target_yaw: float | None = None,
        delta_yaw: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """围绕世界坐标 Z 轴旋转腕部，并在旋转期间保持位置与夹爪命令。

        ``target_yaw``（绝对角）和 ``delta_yaw``（相对初始角）至少提供一个；
        相对角只在入口处转换一次固定目标，避免闭环中逐步累加。控制器通过
        ``action[5]`` 接收 Z 轴轴角增量，每步将环绕误差归一化到 ``[-π, π]``，
        再按 ``step_clip`` 裁剪，直到进入 ``tol`` 或达到停止条件。

        世界偏航定义为末端 X 轴在世界 XY 平面的投影角，即
        ``atan2(R[1, 0], R[0, 0])``。不能直接采用 ``as_euler('zyx')[0]``：
        夹爪朝下（``R[2, 2]≈-1``）时，该欧拉分解会选择另一参数图表并翻转首角
        符号，导致腕部朝命令的反方向旋转。
        """
        from scipy.spatial.transform import Rotation as _R

        def _yaw_of(quat_xyzw):
            """从 robosuite 的 xyzw 四元数提取世界坐标偏航角。"""
            q = quat_xyzw
            rot = _R.from_quat([q[0], q[1], q[2], q[3]])
            R = rot.as_matrix()
            # 取末端 X 轴在世界 XY 平面的方向；该定义在夹爪朝下
            # （R[2, 2]≈-1）时仍连续，不受 Z-Y-X 欧拉角图表翻转影响。
            return float(np.arctan2(R[1, 0], R[0, 0]))

        raw = self.env.raw_obs()
        cur_quat = raw["robot0_eef_quat"]
        start_yaw = _yaw_of(cur_quat)
        if target_yaw is None and delta_yaw is None:
            return {"name": "rotate_wrist", "error": "need target_yaw or delta_yaw"}
        if target_yaw is None:
            target_yaw = start_yaw + float(delta_yaw)

        traj = []
        for step in range(max_steps):
            raw = self.env.raw_obs()
            cur_yaw = _yaw_of(raw["robot0_eef_quat"])
            err = float(target_yaw - cur_yaw)
            # 环绕到 [-π, π]，选择最短旋转方向。
            err = (err + np.pi) % (2 * np.pi) - np.pi
            traj.append({"step": step, "yaw": round(cur_yaw, 4), "err": round(err, 4)})
            if abs(err) < tol:
                break
            step_dyaw = float(np.clip(err, -step_clip, step_clip))
            action = np.zeros(7, dtype=np.float32)
            action[5] = step_dyaw / 0.10  # 映射到约 [-1, 1] 的动作范围
            action[5] = float(np.clip(action[5], -1.0, 1.0))
            action[6] = float(gripper)
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final_yaw = _yaw_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_wrist",
            "start_yaw": round(start_yaw, 4),
            "target_yaw": round(float(target_yaw), 4),
            "final_yaw": round(final_yaw, 4),
            "final_err": round(float((target_yaw - final_yaw + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    def rotate_pitch(
        self,
        *,
        target_pitch: float | None = None,
        delta_pitch: float | None = None,
        gripper: float = 1.0,
        max_steps: int = 40,
        tol: float = 0.02,
        step_clip: float = 0.10,
    ) -> dict:
        """围绕世界 X 轴倾斜夹爪，并在旋转期间保持位置、偏航和夹爪命令。

        俯仰角按末端 Z 轴相对世界 ``-Z`` 方向在 YZ 平面中的夹角定义：
        ``pitch = atan2(R[1, 2], -R[2, 2])``。因此 ``0`` 表示常见的夹爪朝下
        姿态，``+π/2`` 表示末端 Z 轴指向世界 ``+Y``，``-π/2`` 则指向
        ``-Y``。OSC 通过 ``action[3]`` 的 X 轴轴角分量驱动该角度。

        ``target_pitch``（绝对角）和 ``delta_pitch``（相对初始角）至少提供一个。
        该 primitive 适合在进入正面法向沿世界 ``±Y`` 的狭窄开口前先调整姿态；
        每步对最短环绕误差裁剪，直到进入 ``tol`` 或达到环境/步数停止条件。
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(quat_xyzw):
            """从 xyzw 四元数计算上述世界 YZ 平面俯仰定义。"""
            q = quat_xyzw
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        raw = self.env.raw_obs()
        start_pitch = _pitch_of(raw["robot0_eef_quat"])
        if target_pitch is None and delta_pitch is None:
            return {"name": "rotate_pitch",
                    "error": "need target_pitch or delta_pitch"}
        if target_pitch is None:
            target_pitch = start_pitch + float(delta_pitch)

        traj = []
        for step in range(max_steps):
            raw = self.env.raw_obs()
            cur_pitch = _pitch_of(raw["robot0_eef_quat"])
            err = float(target_pitch - cur_pitch)
            err = (err + np.pi) % (2 * np.pi) - np.pi
            traj.append({"step": step,
                         "pitch": round(cur_pitch, 4),
                         "err": round(err, 4)})
            if abs(err) < tol:
                break
            step_dpitch = float(np.clip(err, -step_clip, step_clip))
            action = np.zeros(7, dtype=np.float32)
            action[3] = step_dpitch / 0.10
            action[3] = float(np.clip(action[3], -1.0, 1.0))
            action[6] = float(gripper)
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final_pitch = _pitch_of(self.env.raw_obs()["robot0_eef_quat"])
        return {
            "name": "rotate_pitch",
            "start_pitch": round(start_pitch, 4),
            "target_pitch": round(float(target_pitch), 4),
            "final_pitch": round(final_pitch, 4),
            "final_err": round(float(
                (target_pitch - final_pitch + np.pi) % (2 * np.pi) - np.pi), 4),
            "steps_used": len(traj),
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    def move_pose(
        self,
        xyz,
        *,
        target_pitch: float | None = None,
        target_yaw: float | None = None,
        gripper: float = -1.0,
        step_clip: float = 0.02,
        pitch_step: float = 0.08,
        yaw_step: float = 0.08,
        tol: float = 0.012,
        ori_tol: float = 0.05,
        action_scale: float = 0.05,
        max_steps: int = 150,
    ) -> dict:
        """在同一 OSC 闭环中同时伺服位置、俯仰和偏航。

        与先 ``move_to`` 再 ``rotate_pitch`` 的解耦方式不同，本方法在每个环境步
        同时更新平移和旋转动作。位置误差逐轴裁剪，俯仰/偏航误差分别环绕到
        ``[-π, π]`` 并裁剪后写入 ``action[3]``/``action[5]``。只有位置进入
        ``tol`` 且两个已指定姿态目标均进入 ``ori_tol`` 才算到达。

        位姿协同变化可形成类似 Pi0 的弧线接近轨迹，用于柜体正面或低层搁板等
        固定朝下姿态容易把腕部推入逆解奇异点的位置。循环仍通过 :meth:`_step_env`
        逐步刷新状态、响应取消并录制帧。
        """
        from scipy.spatial.transform import Rotation as _R

        def _pitch_of(q):
            """计算末端 Z 轴相对世界 ``-Z`` 的俯仰角。"""
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 2], -R[2, 2]))

        def _yaw_of(q):
            """计算末端 X 轴投影到世界 XY 平面后的偏航角。"""
            R = _R.from_quat([q[0], q[1], q[2], q[3]]).as_matrix()
            return float(np.arctan2(R[1, 0], R[0, 0]))

        target = np.asarray(_normalize_xyz(xyz), dtype=np.float32)
        traj = []
        step = 0
        for step in range(max_steps):
            cur = self._last_obs_eef_pos
            q = self.env.raw_obs()["robot0_eef_quat"]
            diff = target - cur
            dist = float(np.linalg.norm(diff))
            p_err = 0.0 if target_pitch is None else \
                float((target_pitch - _pitch_of(q) + np.pi) % (2 * np.pi) - np.pi)
            y_err = 0.0 if target_yaw is None else \
                float((target_yaw - _yaw_of(q) + np.pi) % (2 * np.pi) - np.pi)
            traj.append({"step": step, "eef": [round(float(x), 4) for x in cur],
                         "dist": round(dist, 4), "p_err": round(p_err, 3)})
            if dist < tol and abs(p_err) < ori_tol and abs(y_err) < ori_tol:
                break
            action = np.zeros(7, dtype=np.float32)
            sd = np.clip(diff, -step_clip, step_clip)
            action[:3] = np.clip(sd / action_scale, -1.0, 1.0)
            action[3] = float(np.clip(np.clip(p_err, -pitch_step, pitch_step) / 0.10, -1.0, 1.0))
            action[5] = float(np.clip(np.clip(y_err, -yaw_step, yaw_step) / 0.10, -1.0, 1.0))
            action[6] = float(gripper)
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        final = self._last_obs_eef_pos
        fq = self.env.raw_obs()["robot0_eef_quat"]
        return {
            "name": "move_pose",
            "final_eef_pos": [round(float(x), 4) for x in final],
            "final_dist_m": round(float(np.linalg.norm(target - final)), 4),
            "final_pitch": round(_pitch_of(fq), 4),
            "steps_used": step + 1,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    def release(
        self,
        *,
        max_steps: int = 20,
    ) -> dict:
        """保持末端位姿不变，连续发送张开夹爪命令。

        每一步仅设置 7 维动作的夹爪分量，并通过 :meth:`_step_env` 推进环境；
        LIBERO 的 ``On``/``In`` 等谓词可能在释放后触发官方终止。循环在环境结束
        或耗尽 ``max_steps`` 时停止，返回起始、过程峰值和最终开口代理量。
        """
        assert max_steps > 0, f"max_steps must be > 0, got {max_steps}"
        start_grip = self._last_obs_gripper
        peak_grip = start_grip
        for step in range(max_steps):
            action = np.zeros(7, dtype=np.float32)
            action[6] = -1.0  # 张开夹爪
            self._step_env(action)
            peak_grip = max(peak_grip, self._last_obs_gripper)
            if self.env.terminated or self.env.truncated:
                break
        return {
            "name": "release",
            "steps_used": step + 1,
            "start_gripper_opening": round(start_grip, 4),
            "peak_gripper_opening": round(peak_grip, 4),
            "final_gripper_opening": round(self._last_obs_gripper, 4),
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    def set_gripper(
        self,
        *,
        gripper: float = -1.0,
        steps: int = 5,
    ) -> dict:
        """保持当前末端位姿，连续 ``steps`` 步发送指定夹爪命令。

        该低层 primitive 常用于搬运途中加固抓持；与 :meth:`release` 一样不发送
        平移或旋转增量，并在环境终止/截断时提前停止。
        """
        g = float(gripper)
        n = int(steps)
        for _ in range(n):
            action = np.zeros(7, dtype=np.float32)
            action[6] = g
            self._step_env(action)
            if self.env.terminated or self.env.truncated:
                break
        return {
            "name": "set_gripper",
            "gripper": g,
            "steps": n,
            "terminated": self.env.terminated,
            "truncated": self.env.truncated,
        }

    # ---- 供 LLM 闭环调用的只读感知辅助工具 ----

    @readonly
    def segment(
        self,
        prompt: str = "",
        camera: str = "agentview",
        step: int = -1,
        point: list[int] | None = None,
        min_score: float = 0.2,
        *,
        state: EnvState,
    ) -> dict:
        """对既有 RGB 产物执行 SAM3 分割，并估计掩码对应的世界坐标。

        这是只读工具：它从 ``EnvState`` 选择指定 step 的主相机或腕部图像及与之
        同分辨率的预计算世界图，不渲染新视角也不推进模拟器。调用必须在文本
        ``prompt`` 与单个正点 ``point=[row, col]`` 中恰好选择一种；优先使用成对
        存在的高分辨率产物，缺失时回退到标准分辨率。

        SAM3 返回首选掩码后，方法以掩码内有效深度点各轴中位数生成 ``world_xyz``，
        保存可复查的 ``segment_XX.json`` 和半透明覆盖图。服务失败、产物缺失、
        掩码为空或世界图不匹配均转换为结构化错误，使 Planner 可回退到手工观察
        与 :func:`back_project`，而不会中断 episode。
        """
        try:
            record = state.get(step)
        except Exception as exc:
            return {"error": f"state step not available: {exc}"}
        nn = record.step_idx

        camera = camera or "agentview"
        prompt = prompt.strip()
        has_prompt = bool(prompt)
        has_point = point is not None
        # SAM3 的两种提示模式互斥；空白文本视为未提供，避免同时把文本和点
        # 传给远端服务造成含糊的分割语义。
        if has_prompt == has_point:
            return {"error": "segment needs exactly one of prompt or point"}
        # 图像与世界图必须来自同一 camera、step 和分辨率；成对选择可避免把
        # SAM3 像素索引投到另一套坐标网格。
        try:
            image_name, world_name, artifact_pairs = _select_segment_artifacts(
                state, record, camera
            )
        except ValueError as e:
            return {"error": str(e)}
        if image_name is None or world_name is None:
            return {
                "error": "complete segment artifacts not found",
                "step": nn,
                "camera": camera,
                "checked_artifacts": [
                    name
                    for image, world in artifact_pairs
                    for name in (image, world)
                    if name
                ],
            }

        try:
            data = self._sam3_client.segment(
                state.load_bytes(image_name, step=nn),
                text_prompt=prompt if has_prompt else None,
                point=point,
                min_score=min_score,
            )
        except ValueError as e:
            return {
                "error": str(e),
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
            }
        except Exception as e:
            return {
                "error": f"segmentation service call failed: {e}",
                "step": nn,
                "camera": camera,
                "image_artifact": image_name,
                "fallback": "Use manual visual localization and back_project.",
            }

        segment_index = _next_segment_index(record)
        segment_name = f"segment_{segment_index:02d}.json"
        overlay_name = f"segment_overlay_{segment_index:02d}.png"
        saved_overlay = None
        mask = data.mask
        # 只有服务明确找到目标且返回 NumPy 掩码时才访问世界图。世界坐标计算与
        # 覆盖图保存彼此独立：深度产物损坏仍可保留分割诊断，覆盖图保存失败也
        # 不会抹掉已计算的坐标。
        if data.found and isinstance(mask, np.ndarray):
            try:
                world_map = state.load(world_name, step=nn)
            except Exception as exc:
                world_result = {
                    "world_xyz": None,
                    "world_error": f"world map artifact not available: {exc}",
                    "expected_world_artifact": world_name,
                }
            else:
                world_result = _mask_to_world(mask, world_map)
                world_result["world_artifact"] = world_name
            overlay = _make_segment_overlay(state.load(image_name, step=nn), mask)
            if overlay is not None and state.save(
                overlay_name,
                overlay,
                step=nn,
            ):
                saved_overlay = overlay_name
        else:
            world_result = {
                "world_xyz": None,
                "world_error": data.reason or "segmentation did not find a mask",
            }

        segment_blob = {
            "found": data.found,
            "mode": "text" if has_prompt else "point",
            "camera": camera,
            "source_step": nn,
            "segment_index": segment_index,
            "image_artifact": image_name,
            "min_score": min_score,
            "score": round(float(data.score), 3) if data.score is not None else None,
            "box": data.box,
            "mask_shape": list(data.mask_shape) if data.mask_shape else None,
        }
        if has_prompt:
            segment_blob["prompt"] = prompt
        else:
            segment_blob["point"] = point
        if not data.found:
            segment_blob["error"] = data.reason or "SAM3 found no mask"
        segment_blob.update(world_result)
        saved_segment = state.save(
            segment_name,
            segment_blob,
            step=nn,
        )

        result = {
            "found": data.found,
            "step": nn,
            "camera": camera,
            "image_artifact": image_name,
            "score": segment_blob["score"],
            "box": segment_blob["box"],
            "world_xyz": segment_blob["world_xyz"],
            "world_error": segment_blob.get("world_error"),
        }
        if saved_segment is None:
            result["error"] = f"failed to persist segment artifact {segment_name}"
            result["code"] = "segment_artifact_save_failed"
            result["attempted_segment_artifact"] = segment_name
            if "error" in segment_blob:
                result["segmentation_error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        else:
            result["segment_artifact"] = saved_segment
        if saved_segment is not None and "error" in segment_blob:
            result["error"] = segment_blob["error"]
            result["fallback"] = "Use manual visual localization and back_project."
        if saved_overlay is not None:
            result["overlay_artifact"] = saved_overlay
            result["_image_bytes"] = state.load_bytes(saved_overlay, step=nn)
        return result


def _is_primitive_action(name: object) -> bool:
    """判断工具名是否对应会推进 LIBERO 状态的 primitive。

    判定依据是 :class:`LiberoPrimitives` 上存在同名方法且未标记 ``@readonly``；
    非字符串、分割和其他只读工具均返回 ``False``，用于 recipe 过滤。
    """
    if not isinstance(name, str):
        return False
    method = getattr(LiberoPrimitives, name, None)
    return method is not None and not bool(getattr(method, "_readonly", False))


def write_recipe_from_states(state: EnvState, recipe_tag: str) -> str:
    """从 ``EnvState`` 轨迹导出可重放的 LIBERO primitive JSONL。

    按 step 扫描状态记录，保留没有结构化错误的状态推进命令；同一步上由只读
    :meth:`LiberoPrimitives.segment` 另存的成功分割也会还原为工具命令。分割事件
    以其 ``segment_index`` 排在所属 step 的动作之后，从而保留原始调用顺序。
    最终只写公开工具参数，不导出观测、结果或内部 ``state`` 对象。

    Args:
        state: 当前运行的完整状态与产物存储。
        recipe_tag: 用于生成 ``recipe_<tag>.jsonl`` 文件名的标签。

    Returns:
        已保存的 recipe 产物名。
    """
    command_events = []
    for record in state.records():
        command = record.command
        result = record.result
        # record.command 只代表会生成新 StepRecord 的动作调用；错误动作不应进入
        # 可重放 recipe，只读工具则由其持久化产物单独恢复。
        if (
            command is not None
            and _is_primitive_action(command.get("action"))
            and not (isinstance(result, dict) and result.get("error"))
        ):
            command_events.append(((record.step_idx, -1), command))

        # segment 是只读调用，不创建新 step；成功调用由当前记录下的编号 JSON
        # 产物恢复，并用 segment_index 保持同一步内的先后次序。
        for name in sorted(record.artifacts):
            if not (name.startswith("segment_") and name.endswith(".json")):
                continue
            segment = state.load(name, step=record.step_idx)
            if segment.get("error"):
                continue
            if segment["mode"] == "text":
                segment_command = {
                    "action": "segment",
                    "prompt": segment["prompt"],
                    "camera": segment["camera"],
                }
            else:
                segment_command = {
                    "action": "segment",
                    "point": segment["point"],
                    "camera": segment["camera"],
                }
            event_order = (record.step_idx, int(segment["segment_index"]))
            command_events.append((event_order, segment_command))

    command_events.sort(key=lambda event: event[0])
    recipe_name = f"recipe_{recipe_tag}.jsonl"
    state.save(
        recipe_name,
        [command for _, command in command_events],
        step=None,
    )
    return recipe_name


def _metric_depth(depth: Any, camera_meta: dict) -> np.ndarray:
    """把环境深度缓冲转换为二维、米制 ``float32`` 深度图。

    robosuite 深度通常是投影空间中的归一化值；当标定包含 ``depth_near`` 与
    ``depth_far`` 时，使用相同投影模型反解相机 Z 深度。若上游已经提供米制值或
    标定缺失，则仅规范化形状和 dtype，不猜测额外尺度。
    """
    d = np.asarray(depth, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    near = camera_meta.get("depth_near")
    far = camera_meta.get("depth_far")
    if near is not None and far is not None:
        d = near / (1.0 - d * (1.0 - near / far))
    return d


def _world_from_depth(depth_metric: np.ndarray, camera_meta: dict) -> np.ndarray:
    """将米制深度图逐像素反投影为世界坐标图。

    对像素 ``(row, col)``，先用内参 ``K`` 恢复相机坐标
    ``[(col-cx)z/fx, (row-cy)z/fy, z]``，追加齐次分量后再乘
    ``extrinsic_cam2world``。返回数组形状为 ``[H, W, 3]``，与输入深度及对应
    RGB 完全逐像素对齐，供分割掩码聚合和 :func:`back_project` 直接索引。
    """
    k_matrix = np.array(camera_meta["intrinsic_K"], dtype=np.float64)
    extrinsic = np.array(camera_meta["extrinsic_cam2world"], dtype=np.float64)
    fx, fy = k_matrix[0, 0], k_matrix[1, 1]
    cx, cy = k_matrix[0, 2], k_matrix[1, 2]
    height, width = depth_metric.shape
    rr, cc = np.mgrid[0:height, 0:width]
    z = depth_metric.astype(np.float64)
    camera_points = np.stack(
        [(cc - cx) * z / fx, (rr - cy) * z / fy, z, np.ones_like(z)],
        axis=-1,
    )
    return (camera_points @ extrinsic.T)[..., :3]


def dump_state(
    primitives: LiberoPrimitives,
    env_state: EnvState,
    log: dict | None = None,
) -> StepRecord:
    """把当前 LIBERO 观测原子化写成一个 ``StepRecord``。

    首先从 ``raw_obs`` 提取末端位姿、夹爪关节与对象名等轻量 JSON 状态，再通过
    :meth:`EnvState.record_step` 建立记录上下文，将任务语言、终止标志以及可选的
    命令/结果/耗时绑定到同一 step。上下文内部调用
    :func:`_save_observation_artifacts` 保存 RGB、深度、相机标定和世界坐标图；
    退出上下文后记录才完整可见并返回。

    ``log=None`` 用于 reset 后的 step 0，此时没有关联命令；动作后的调用则把
    Planner 命令与 primitive 原始结果一起保留下来，供状态查看、Dashboard 和
    recipe 导出复用。
    """
    raw = primitives.env.raw_obs()
    state = {
        "robot0_eef_pos": [float(x) for x in raw["robot0_eef_pos"]],
        "robot0_eef_quat": [float(x) for x in raw["robot0_eef_quat"]],
        "robot0_gripper_qpos": [float(x) for x in raw["robot0_gripper_qpos"]],
        "object_names": sorted(
            k[:-4]
            for k in raw
            if k.endswith("_pos") and "robot0" not in k and "to_robot" not in k
        ),
    }
    log = log or {}
    with env_state.record_step(
        state=state,
        terminated=primitives.env.terminated,
        truncated=primitives.env.truncated,
        command=log.get("command"),
        result=log.get("result"),
        elapsed_s=log.get("elapsed_s"),
        extras={
            "task_language": primitives.env.get_task_language(),
        },
    ) as step_idx:
        _save_observation_artifacts(primitives, env_state, step_idx, raw)
    return env_state.get(step_idx)


def _save_observation_artifacts(
    primitives: LiberoPrimitives,
    env_state: EnvState,
    step_idx: int,
    raw: dict[str, Any],
) -> None:
    """保存一个 step 的多视角 RGB-D、标定与预计算世界坐标产物。

    产物分为策略方向主图、与相机标定/深度严格对齐的标准分辨率主视角和腕部
    视角，以及按需渲染的 1024×1024 高分辨率 RGB/世界图。主相机外参静态，
    腕部相机外参随机械臂运动，因此后者必须逐 step 保存。

    每组可选产物独立捕获异常：单个相机或高分辨率渲染失败不会阻止机器人状态
    记录落盘。调用方应以 ``StepRecord.artifacts`` 判断某项产物是否可用。
    """
    env_state.save(
        "agentview_policy.png",
        primitives._last_obs["main_images"],
        step=step_idx,
    )

    # 主相机标定在 episode 内静态；仍按 step 保存，使每条记录可独立解释其产物。
    agentview_meta = primitives.env.get_camera_meta(
        camera_name="agentview",
        height=256,
        width=256,
    ) or {}
    if agentview_meta:
        cam_meta_out = dict(agentview_meta)
        cam_meta_out["projection"] = (
            "Prefer the back_project(row, col, step=NN) MCP tool; it "
            "uses the 1024x1024 high-resolution world map by default. "
            "Pass resolution='low' only when row/col came from the "
            "256x256 calibration-frame image."
        )
        cam_meta_out["note"] = (
            "The agentview_depth.npz observation is aligned with agentview.png. "
            "agentview_policy.png uses the Pi0 orientation and must not supply "
            "pixels for back-projection."
        )
        env_state.save(
            "agentview_metadata.json",
            cam_meta_out,
            step=step_idx,
        )

    # 保存与深度/K 同坐标系的标准分辨率 RGB：robosuite 原始缓冲区需垂直
    # 翻转。策略图采用 Pi0 方向，只供策略/展示，不能提供反投影像素。
    try:
        ci = raw.get("agentview_image")
        if ci is not None:
            ci = np.asarray(ci)
            if ci.dtype != np.uint8:
                ci = ci.astype(np.uint8)
            env_state.save(
                "agentview.png",
                ci[::-1],
                step=step_idx,
            )
    except Exception as e:
        logger.warning("image_cam dump failed: %s", e)

    # 保存逐 step 的主相机米制深度与世界图，二者均采用标定坐标方向。
    try:
        d = raw.get("agentview_depth")
        if d is not None:
            # 垂直翻转后才与相机矩阵一致：robosuite 的投影
            # M = K_exp @ inv(extrinsic) 以该方向解释深度。实测将对象真值世界坐标
            # 投到图像后，对应像素深度与物体表面深度一致，因此 agentview.png、
            # agentview_depth.npz 和本 step 标定可以逐像素配套使用。
            d = _metric_depth(d, agentview_meta)[::-1]
            env_state.save(
                "agentview_depth.npz",
                d.astype(np.float32),
                step=step_idx,
            )
            world = _world_from_depth(d, agentview_meta).astype(np.float32)
            env_state.save(
                "agentview_world.npz",
                world,
                step=step_idx,
            )
    except Exception as e:
        logger.warning("depth dump failed: %s", e)

    # 腕部相机随末端运动：RGB、深度、世界图和外参必须来自同一 step。
    try:
        wimg = raw.get("robot0_eye_in_hand_image")
        if wimg is None:
            logger.warning("wrist image missing from raw_obs")
        else:
            wimg = np.asarray(wimg)
            if wimg.dtype != np.uint8:
                wimg = wimg.astype(np.uint8)
            env_state.save(
                "wrist.png",
                wimg[::-1],
                step=step_idx,
            )
    except Exception as e:
        logger.warning("wrist image dump failed: %s", e)

    try:
        wdpt = raw.get("robot0_eye_in_hand_depth")
        if wdpt is None:
            logger.warning("wrist depth missing from raw_obs")
        else:
            wdpt_arr = np.asarray(wdpt, dtype=np.float32)
            height, width = wdpt_arr.shape[:2]
            wmeta = primitives.env.get_camera_meta(
                camera_name="robot0_eye_in_hand",
                height=int(height),
                width=int(width),
            )
            if wmeta is None:
                logger.warning("wrist camera meta missing; skipping wrist depth/world")
            else:
                wdpt_metric = _metric_depth(wdpt_arr, wmeta)[::-1]
                env_state.save(
                    "wrist_depth.npz",
                    wdpt_metric.astype(np.float32),
                    step=step_idx,
                )
                world_w = _world_from_depth(wdpt_metric, wmeta).astype(np.float32)
                env_state.save(
                    "wrist_world.npz",
                    world_w,
                    step=step_idx,
                )

                wmeta_out = dict(wmeta)
                wmeta_out["note"] = (
                    "MOVING camera: extrinsic_cam2world is for THIS step "
                    "only. The matching wrist world-map observation gives world "
                    "(x,y,z) for that pixel in the same world frame as the "
                    "agentview world-map artifact."
                )
                env_state.save(
                    "wrist_metadata.json",
                    wmeta_out,
                    step=step_idx,
                )
    except Exception as e:
        logger.warning("wrist depth/world dump failed: %s", e)

    # 额外渲染 1024×1024 主视角供精细定位；世界图降为 float16 以控制
    # EnvState 产物体积，RGB 与世界图仍保持相同翻转和像素索引。
    try:
        rgb_hi, depth_hi = primitives.env.render_camera(
            camera_name="agentview",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_hi = primitives.env.get_camera_meta("agentview", 1024, 1024)
        if meta_hi is None:
            raise RuntimeError("agentview camera metadata missing")
        env_state.save(
            "agentview_high.png",
            np.asarray(rgb_hi)[::-1],
            step=step_idx,
        )
        world_hi = _world_from_depth(
            _metric_depth(depth_hi, meta_hi)[::-1],
            meta_hi,
        ).astype(np.float16)
        env_state.save(
            "agentview_world_high.npz",
            world_hi,
            step=step_idx,
        )
    except Exception as e:
        logger.warning("agentview high-res dump failed: %s", e)

    # 腕部高分辨率视角同样在当前末端姿态下即时渲染，不能复用其他 step 外参。
    try:
        rgb_wrist_hi, depth_wrist_hi = primitives.env.render_camera(
            camera_name="robot0_eye_in_hand",
            height=1024,
            width=1024,
            depth=True,
        )
        meta_wrist_hi = primitives.env.get_camera_meta(
            "robot0_eye_in_hand", 1024, 1024
        )
        if meta_wrist_hi is None:
            raise RuntimeError("robot0_eye_in_hand camera metadata missing")
        env_state.save(
            "wrist_high.png",
            np.asarray(rgb_wrist_hi)[::-1],
            step=step_idx,
        )
        world_wrist_hi = _world_from_depth(
            _metric_depth(depth_wrist_hi, meta_wrist_hi)[::-1],
            meta_wrist_hi,
        ).astype(np.float16)
        env_state.save(
            "wrist_world_high.npz",
            world_wrist_hi,
            step=step_idx,
        )
    except Exception as e:
        logger.warning("wrist high-res dump failed: %s", e)


# ---------------------------------------------------------------------------
# Planner 工具 schema（Anthropic 形状的规范声明）
# ---------------------------------------------------------------------------
# 此列表只描述公开工具名、说明和 JSON 输入约束，不在这里持有运行时对象。
# LiberoToolkit._register_libero_tools 会逐项遍历：状态感知工具通过 partial 注入
# 当前 EnvState，其他同名动作则绑定到 LiberoPrimitives；找不到 handler 的条目
# 不会注册。以下 description/schema 是执行协议的一部分，不应随文档整理而改写。

TOOLS_SPEC = [
    {
        "name": "view_env_state",
        "description": (
            "Read one recorded state and its observation artifacts. Step -1 "
            "selects the latest entry. Embeds policy, agentview, and wrist "
            "images when available. "
            "Use the calibration-frame images for pixel back-projection; JSON "
            "state alone is not enough. Use agentview for global tabletop "
            "layout and object locations; use wrist for close-range details "
            "near the gripper, occlusions, and container/cabinet interiors."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step number; 0 = initial, -1 = latest.",
                },
            },
        },
    },
    {
        "name": "move_to",
        "description": (
            "Scripted EEF servo to a world-frame XYZ target via the OSC "
            "controller. Holds orientation (use rotate_wrist / rotate_pitch "
            "/ move_pose to reorient). gripper: -1 = open, +1 = close. NEVER "
            "command a single move_to with |Δxy| > 0.30 — OSC flips IK and "
            "the run corrupts; split long traversal into 2-3 mid waypoints "
            "at carry z."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "tol": {"type": "number", "description": "Position tolerance, m (default 0.012)"},
                "step_clip": {"type": "number", "description": "Per-step Δxyz cap before action_scale, m (default 0.025)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 80)"},
                "action_scale": {"type": "number", "description": "OSC action scale (default 0.05)"},
                "target_yaw": {
                    "type": ["number", "null"],
                    "description": "Optional world-frame yaw target in radians",
                },
                "yaw_step_clip": {"type": "number", "description": "Per-step yaw clip, rad (default 0.10)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "pi0_pick",
        "description": (
            "Pi0.5 closed-loop pick. Use it for the grasp; YOU then do "
            "every move_to and release. Use modest max_chunks and verify "
            "the grasp from EEF lift, gripper closure, and available images."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Pi0 prompt (e.g. 'pick up the akita black bowl').",
                },
                "max_chunks": {"type": "integer", "description": "Action-chunk budget (default 24)"},
                "lift_thresh": {"type": "number", "description": "EEF post-descent ascent threshold for success, m (default 0.05)"},
                "gripper_closed_thresh": {"type": "number", "description": "Finger-separation closed threshold (default 0.06)"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "pi0_doubled",
        "description": (
            "Pi0.5 closed-loop contact skill for non-pick interactions "
            "(e.g. stove/knob/button/short push). Returned success/task_success "
            "only mirrors official termination; for intermediate contact "
            "skills, success=false does not necessarily mean the contact "
            "interaction failed. Inspect image/state evidence. Do not use it "
            "as a general pick/place shortcut."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Contact-skill prompt, e.g. 'turn on the stove'.",
                },
                "max_chunks": {"type": "integer", "description": "Action-chunk budget (default 20)"},
            },
            "required": ["prompt"],
        },
    },
    {
        "name": "release",
        "description": (
            "Open the gripper for up to max_steps env steps while holding "
            "EEF in place. Triggers libero termination if the matching "
            "On/In predicate is met."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "max_steps": {"type": "integer", "description": "Step budget (default 20)"},
            },
        },
    },
    {
        "name": "set_gripper",
        "description": (
            "Hold the current EEF pose and drive the gripper command for "
            "`steps` env steps. Use to firm up a grip mid-carry."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "gripper": {
                    "type": "number",
                    "description": "Gripper command: -1 open, +1 close (default -1)",
                },
                "steps": {"type": "integer", "description": "Number of env steps (default 5)"},
            },
        },
    },
    {
        "name": "rotate_wrist",
        "description": (
            "Rotate the wrist around the world Z-axis. Provide either "
            "target_yaw (absolute) or delta_yaw (relative). Holds xyz fixed."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_yaw": {"type": ["number", "null"], "description": "Absolute world-frame yaw target, rad"},
                "delta_yaw": {"type": ["number", "null"], "description": "Relative yaw delta, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during rotation (default +1)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 40)"},
                "tol": {"type": "number", "description": "Yaw tolerance, rad (default 0.02)"},
                "step_clip": {"type": "number", "description": "Per-step yaw clip, rad (default 0.10)"},
            },
        },
    },
    {
        "name": "rotate_pitch",
        "description": (
            "Tilt the gripper around the world X-axis. Provide either "
            "target_pitch (absolute) or delta_pitch (relative). Holds xyz "
            "and yaw fixed. Use before threading the gripper into a narrow "
            "opening whose front face normal is along world ±y (e.g. "
            "microwave cavity)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "target_pitch": {"type": ["number", "null"], "description": "Absolute world-frame pitch target, rad"},
                "delta_pitch": {"type": ["number", "null"], "description": "Relative pitch delta, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during rotation (default +1)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 40)"},
                "tol": {"type": "number", "description": "Pitch tolerance, rad (default 0.02)"},
                "step_clip": {"type": "number", "description": "Per-step pitch clip, rad (default 0.10)"},
            },
        },
    },
    {
        "name": "move_pose",
        "description": (
            "Servo position AND orientation (pitch + yaw) SIMULTANEOUSLY. "
            "Unlike move_to (holds orientation) + rotate_pitch (holds xyz), "
            "this co-varies xyz and wrist tilt every env.step. Use to thread "
            "cabinet-front / low-shelf poses where a decoupled position "
            "servo drives the wrist into an IK singularity and stalls."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "xyz": {
                    "type": "array",
                    "description": "World-frame target [x, y, z] in meters",
                    "items": {"type": "number"},
                    "minItems": 3,
                    "maxItems": 3,
                },
                "target_pitch": {"type": ["number", "null"], "description": "Absolute pitch target, rad"},
                "target_yaw": {"type": ["number", "null"], "description": "Absolute yaw target, rad"},
                "gripper": {"type": "number", "description": "Gripper command held during the move (default -1)"},
                "step_clip": {"type": "number", "description": "Per-step Δxyz cap, m (default 0.02)"},
                "pitch_step": {"type": "number", "description": "Per-step pitch clip, rad (default 0.08)"},
                "yaw_step": {"type": "number", "description": "Per-step yaw clip, rad (default 0.08)"},
                "tol": {"type": "number", "description": "Position tolerance, m (default 0.012)"},
                "ori_tol": {"type": "number", "description": "Orientation tolerance, rad (default 0.05)"},
                "action_scale": {"type": "number", "description": "OSC action scale (default 0.05)"},
                "max_steps": {"type": "integer", "description": "Step budget (default 150)"},
            },
            "required": ["xyz"],
        },
    },
    {
        "name": "view_camera_meta",
        "description": (
            "Read per-step camera calibration metadata from recorded artifacts."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera metadata to read (default agentview).",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Metadata step to use; -1 = latest.",
                },
            },
        },
    },
    {
        "name": "segment",
        "description": (
            "SAM3 visual segmentation over an existing run artifact. It never "
            "renders a new camera view. Provide exactly one text prompt or "
            "single positive point. A successful top-ranked mask is projected "
            "through the matching world map to produce world_xyz."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Object/text prompt to segment.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Artifact camera to use (default agentview).",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Step to segment; -1 = latest.",
                },
                "point": {
                    "type": ["array", "null"],
                    "description": (
                        "Optional single positive point as [row, col]. "
                        "Mutually exclusive with prompt."
                    ),
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "min_score": {
                    "type": "number",
                    "description": "Minimum accepted mask score (default 0.2).",
                },
            },
        },
    },
    {
        "name": "back_project",
        "description": (
            "Back-project a pixel (row, col) to a world XYZ point using the "
            "selected camera's precomputed world map. Row 0 = top of image, "
            "col 0 = left. Returns world_xyz in meters.\n\n"
            "USE THIS to find where an object is in the world — look at "
            "the embedded high-resolution image returned by view_env_state "
            "to pick a pixel on the target object, then call back_project. "
            "The default resolution is high (1024x1024). Pass "
            "resolution='low' only for pixels from the embedded/standard "
            "256 image. The pixel coordinates must come "
            "from the same camera and resolution requested here. Use "
            "camera='agentview' for global tabletop layout and object "
            "locations; use camera='wrist' for close-range details near the "
            "gripper, occlusions, and container/cabinet interiors. "
            "Sample several pixels on the object and median their xy for "
            "robustness.\n\n"
            "REGION MODE: pass row_range=[r0,r1] and col_range=[c0,c1] instead "
            "of row/col to get the midpoint of world xy over that pixel window, "
            "with an optional world-z band (z_min, z_max). Use it for the "
            "center of a container cavity or flat region, where a single-pixel "
            "or mask-median estimate is biased toward an edge/rim."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "row": {
                    "type": ["integer", "null"],
                    "description": "Pixel row (0=top) in the selected resolution image.",
                },
                "col": {
                    "type": ["integer", "null"],
                    "description": "Pixel column (0=left) in the selected resolution image.",
                },
                "step": {
                    "type": "integer",
                    "default": -1,
                    "description": "Depth/world-map step; 0 = initial, -1 = latest.",
                },
                "camera": {
                    "type": "string",
                    "enum": ["agentview", "wrist"],
                    "description": "Camera to back-project from (default agentview).",
                },
                "resolution": {
                    "type": "string",
                    "enum": ["high", "low"],
                    "description": (
                        "Coordinate system for row/col (default high). "
                        "Use low only when row/col came from the "
                        "embedded/standard 256 image."
                    ),
                },
                "row_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [r0, r1] pixel row window. Requires col_range.",
                },
                "col_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Region mode: [c0, c1] pixel col window. Requires row_range.",
                },
                "z_min": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only pixels with world z >= z_min.",
                },
                "z_max": {
                    "type": ["number", "null"],
                    "description": "Region mode: keep only pixels with world z <= z_max.",
                },
            },
        },
    },
]


@readonly
def view_env_state(step: int = -1, *, state: EnvState) -> dict:
    """读取一个 ``StepRecord``，并组装 Planner/Dashboard 可消费的状态视图。

    ``step=-1`` 选择最新记录。返回轻量 JSON 状态、终止标志、命令日志和产物清单，
    并为三个约定槽位附加图像字节：策略主图、标定方向主视角、腕部视角。
    标定/腕部图优先读取 1024×1024 版本，缺失时回退到标准分辨率；单个文件在
    清单存在但磁盘缺失时仅跳过该图，不使整次状态查看失败。

    函数标记为 ``@readonly``，因此调用本身不会触发新的环境 step 或状态 dump。
    """
    try:
        record = state.get(step)
    except Exception as exc:
        return {"error": f"state step not available: {exc}"}

    nn = record.step_idx
    extras = record.extras
    out: dict = {
        "step": nn,
        "terminated": record.terminated,
        "truncated": record.truncated,
        "state": record.state,
        "artifacts": sorted(record.artifacts),
    }
    out["task_language"] = extras.get("task_language")
    out["log"] = {
        "command": record.command,
        "result": record.result,
        "elapsed_s": record.elapsed_s,
    }
    for slot, names in (
        ("_image_bytes", ("agentview_policy.png",)),
        ("_image_cam_bytes", ("agentview_high.png", "agentview.png")),
        ("_image_wrist_bytes", ("wrist_high.png", "wrist.png")),
    ):
        name = next((name for name in names if name in record.artifacts), None)
        if name:
            try:
                out[slot] = state.load_bytes(name, step=nn)
            except FileNotFoundError:
                pass
    return out


def _select_segment_artifacts(
    state: EnvState,
    record: StepRecord,
    camera: str,
) -> tuple[str | None, str | None, list[tuple[str | None, str | None]]]:
    """为分割选择同相机、同分辨率且实际存在的 RGB/世界图产物对。

    高分辨率组合优先，标准分辨率作为回退；同时检查记录清单和底层文件，避免
    只凭陈旧 artifact 名称调用 SAM3。第三个返回值保留全部候选，供错误响应说明
    已检查哪些产物。
    """
    if camera not in ("agentview", "wrist"):
        raise ValueError(f"unknown segment camera: {camera}")
    pairs = [
        (f"{camera}_high.png", f"{camera}_world_high.npz"),
        (f"{camera}.png", f"{camera}_world.npz"),
    ]
    for image_name, world_name in pairs:
        if (
            image_name in record.artifacts
            and world_name in record.artifacts
            and state.exists(image_name, step=record.step_idx)
            and state.exists(world_name, step=record.step_idx)
        ):
            return image_name, world_name, pairs
    return None, None, pairs


def _next_segment_index(record: StepRecord) -> int:
    """返回当前 step 中首个未使用的两位分割产物序号。"""
    idx = 0
    while f"segment_{idx:02d}.json" in record.artifacts:
        idx += 1
    return idx


def _mask_to_world(mask: np.ndarray, world_map: np.ndarray,
                   min_valid: int = 10) -> dict:
    """把布尔分割掩码聚合为稳健的世界坐标估计。

    掩码必须与 ``[H, W, >=3]`` 世界图逐像素同形；函数不会自动缩放，以免插值
    或行列错位产生看似合理但错误的坐标。过滤非有限值和近零占位点后，至少需要
    ``min_valid`` 个有效像素，并分别取 XYZ 中位数以减弱边缘深度、遮挡和离群点
    影响。返回值同时携带像素数与有效点数，便于调用方判断置信度。
    """
    if world_map.ndim != 3 or world_map.shape[2] < 3:
        return {
            "world_xyz": None,
            "world_error": f"invalid world map shape: {tuple(world_map.shape)}",
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    if mask.shape != world_map.shape[:2]:
        return {
            "world_xyz": None,
            "world_error": (
                f"mask/world shape mismatch: mask={tuple(mask.shape)}, "
                f"world={tuple(world_map.shape[:2])}"
            ),
            "n_pixels": int(mask.sum()),
            "n_valid": 0,
            "mask_resized_to_world_shape": False,
        }

    ys, xs = np.where(mask)
    if ys.size == 0:
        return {"world_xyz": None, "world_error": "empty mask"}

    pts = world_map[ys, xs].astype(np.float64)
    valid = np.isfinite(pts).all(axis=1) & (np.abs(pts).sum(axis=1) > 1e-6)
    pts = pts[valid]
    result = {
        "centroid_pixel": [
            int(round(float(np.median(xs)))),
            int(round(float(np.median(ys)))),
        ],
        "n_pixels": int(mask.sum()),
        "n_valid": int(pts.shape[0]),
        "mask_resized_to_world_shape": False,
    }
    if pts.shape[0] < min_valid:
        result.update({
            "world_xyz": None,
            "world_error": f"too few valid depth pixels ({int(pts.shape[0])})",
        })
        return result

    result["world_xyz"] = [
        round(float(np.median(pts[:, 0])), 4),
        round(float(np.median(pts[:, 1])), 4),
        round(float(np.median(pts[:, 2])), 4),
    ]
    return result


def _make_segment_overlay(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray | None:
    """在掩码区域叠加半透明红色，生成不修改原图的诊断图。

    图像与掩码尺寸不一致时返回 ``None``，避免为错误配对的产物制造误导性覆盖图。
    """
    if image.ndim != 3 or image.shape[:2] != mask.shape:
        return None
    overlay = image.copy()
    red = np.zeros_like(overlay)
    red[..., 0] = 255
    overlay[mask] = (
        0.55 * overlay[mask].astype(np.float32)
        + 0.45 * red[mask].astype(np.float32)
    ).astype(np.uint8)
    return overlay


@readonly
def view_camera_meta(
    camera: str = "agentview",
    step: int = -1,
    *,
    state: EnvState,
) -> dict:
    """读取指定 step 的相机标定元数据，用于定位和产物解释。

    主相机标定在 episode 内静态；腕部相机外参随末端运动，返回值因此显式携带
    实际 ``step``。函数只读取已保存的 ``<camera>_metadata.json``，不会重新查询
    环境或渲染图像；缺失标定统一返回结构化错误。
    """
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}

    try:
        record = state.get(step)
        metadata_name = f"{camera}_metadata.json"
        if metadata_name not in record.artifacts:
            raise FileNotFoundError(metadata_name)
        meta = state.load(metadata_name, step=record.step_idx)
    except Exception as e:
        return {"error": f"{camera} camera metadata not found: {e}"}

    if camera == "agentview":
        return {"camera": "agentview", "camera_meta": meta}
    return {"camera": "wrist", "step": record.step_idx, "camera_meta": meta}


@readonly
def back_project(
    row: int | None = None,
    col: int | None = None,
    step: int = -1,
    camera: str = "agentview",
    resolution: str = "high",
    row_range: list | None = None,
    col_range: list | None = None,
    z_min: float | None = None,
    z_max: float | None = None,
    *,
    state: EnvState,
) -> dict:
    """从预计算世界图反查像素或图像区域对应的世界坐标。

    单像素模式直接索引 ``world_map[row, col]``；标准分辨率下还读取米制深度并
    验证范围。区域模式要求同时提供半开区间 ``row_range``/``col_range``，先将
    边界裁到图像范围，再过滤非有限/近零世界点和可选 Z 高度带。其 ``center_xyz``
    使用有效点 XYZ 包围范围的 XY 中点与 Z 中位数，适合容器空腔或平面区域；
    ``median_xyz`` 另供稳健比较。

    ``resolution='high'`` 与 ``'low'`` 必须匹配像素来源；主相机和腕部相机也不能
    混用。函数只消费当前 ``StepRecord`` 的世界图，不执行在线投影、不渲染相机，
    因而标记为只读。
    """
    if camera not in ("agentview", "wrist"):
        return {"error": f"bad camera '{camera}' (use 'agentview' or 'wrist')"}
    if resolution not in ("high", "low"):
        return {"error": f"bad resolution '{resolution}' (use 'high' or 'low')"}

    # 任一范围参数出现即进入区域模式；此时单像素 row/col 不参与计算。
    region_mode = row_range is not None or col_range is not None
    if not region_mode and (row is None or col is None):
        return {
            "error": (
                "provide either (row, col) for a single pixel, or "
                "row_range=[r0,r1] and col_range=[c0,c1] for a region center"
            )
        }

    try:
        record = state.get(step)
    except Exception as e:
        return {"error": f"state step not available: {e}"}
    nn = record.step_idx

    # 直接选择与调用方像素坐标系一致的预计算世界图；不在高低分辨率之间
    # 自动缩放坐标，以免舍入与方向差异造成静默定位偏差。
    hi_artifact = f"{camera}_world_high.npz"
    low_artifact = f"{camera}_world.npz"
    source_artifact = hi_artifact if resolution == "high" else low_artifact
    if source_artifact not in record.artifacts:
        return {
            "error": (
                f"{camera} {resolution}-resolution world map not recorded "
                f"for step {nn}"
            )
        }

    try:
        world_map = state.load(str(source_artifact), step=nn)
    except Exception as e:
        return {
            "error": (
                f"{camera} {resolution}-resolution artifact not found "
                f"for step {nn}: {e}"
            )
        }

    height, width = world_map.shape[:2]

    if region_mode:
        if row_range is None or col_range is None:
            return {
                "error": "region mode needs BOTH row_range=[r0,r1] and col_range=[c0,c1]"
            }
        try:
            r0, r1 = int(row_range[0]), int(row_range[1])
            c0, c1 = int(col_range[0]), int(col_range[1])
        except Exception:
            return {"error": "row_range/col_range must each be [min, max] integers"}
        r0, r1 = sorted((max(0, r0), min(height, r1)))
        c0, c1 = sorted((max(0, c0), min(width, c1)))
        if r1 <= r0 or c1 <= c0:
            return {
                "error": (
                    f"empty region after clamping to image {height}x{width}: "
                    f"rows [{r0},{r1}] cols [{c0},{c1}]"
                )
            }
        # 区间按 Python 切片的半开语义裁剪到图像边界；世界图中的非有限值和
        # 全零占位点先剔除，再施加可选 Z 带，避免背景无效深度影响区域中心。
        window = world_map[r0:r1, c0:c1].reshape(-1, world_map.shape[2]).astype(
            np.float64
        )
        finite = np.isfinite(window).all(axis=1) & (
            np.abs(window[:, :3]).sum(axis=1) > 1e-6
        )
        pts = window[finite]
        n_total = int(pts.shape[0])
        if z_min is not None:
            pts = pts[pts[:, 2] >= float(z_min)]
        if z_max is not None:
            pts = pts[pts[:, 2] <= float(z_max)]
        if pts.shape[0] < 8:
            return {
                "error": (
                    f"too few valid pixels in region after z-filter "
                    f"({int(pts.shape[0])}); widen the window or the z band"
                ),
                "n_valid_before_zfilter": n_total,
            }
        xs, ys, zs = pts[:, 0], pts[:, 1], pts[:, 2]
        # XY 用世界点云包围范围中点表达区域几何中心；Z 使用中位数抵抗边缘和
        # 遮挡离群值。返回的 median_xyz 让调用方也能查看稳健统计中心。
        center = [
            round(float((xs.min() + xs.max()) / 2.0), 4),
            round(float((ys.min() + ys.max()) / 2.0), 4),
            round(float(np.median(zs)), 4),
        ]
        return {
            "camera": camera,
            "resolution": resolution,
            "mode": "region",
            "row_range": [r0, r1],
            "col_range": [c0, c1],
            "z_band": [z_min, z_max],
            "center_xyz": center,
            "median_xyz": [
                round(float(np.median(xs)), 4),
                round(float(np.median(ys)), 4),
                round(float(np.median(zs)), 4),
            ],
            "n_valid": int(pts.shape[0]),
            "step": nn,
            "image_size": [height, width],
            "source_artifact": source_artifact,
        }

    if row < 0 or row >= height or col < 0 or col >= width:
        return {
            "error": (
                f"pixel ({row},{col}) out of bounds; {camera} image is "
                f"{height}x{width}"
            )
        }

    depth_m = None
    if source_artifact == low_artifact:
        # 标准分辨率会同时保存独立米制深度，可在返回前做物理范围校验；高分辨率
        # 只持久化世界图以节省空间，因此直接验证其中的 XYZ。
        try:
            depth_artifact = f"{camera}_depth.npz"
            if depth_artifact not in record.artifacts:
                raise FileNotFoundError(depth_artifact)
            depth = state.load(depth_artifact, step=nn)
            if depth.ndim == 3:
                depth = depth[..., 0]
        except Exception as e:
            return {"error": f"{camera} depth not found for step {nn}: {e}"}
        depth_m = float(depth[row, col])
        if not np.isfinite(depth_m) or depth_m <= 0 or depth_m > 10:
            return {
                "error": (
                    f"invalid {camera} depth {depth_m:.3f}m at pixel "
                    f"({row},{col}); pick a different pixel"
                )
            }
    world_xyz_raw = world_map[row, col]
    if (
        not np.isfinite(world_xyz_raw).all()
        or float(np.abs(world_xyz_raw[:3]).sum()) <= 1e-6
    ):
        return {"error": f"invalid {camera} world xyz at pixel ({row},{col})"}
    world_xyz = [round(float(v), 4) for v in world_xyz_raw[:3]]

    out = {
        "camera": camera,
        "resolution": resolution,
        "pixel": [row, col],
        "world_xyz": world_xyz,
        "step": nn,
        "image_size": [height, width],
        "source_artifact": source_artifact,
    }
    if depth_m is not None:
        out["depth_m"] = round(depth_m, 4)
    return out
