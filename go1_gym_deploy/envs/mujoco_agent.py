import numpy as np
import torch
import mujoco

from go1_gym_deploy.envs.lcm_agent import LCMAgent


class MujocoAgent(LCMAgent):
    def __init__(
        self,
        cfg,
        se,
        command_profile,
        model,
        data,
        actuator_names=None,
    ):
        super().__init__(cfg, se, command_profile)
        self.model = model
        self.data = data

        self.actuator_names = actuator_names or [
            "FL_hip", "FL_thigh", "FL_calf",
            "FR_hip", "FR_thigh", "FR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
        ]
        self.actuator_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in self.actuator_names
        ]
        if any(aid < 0 for aid in self.actuator_ids):
            missing = [n for n, aid in zip(self.actuator_names, self.actuator_ids) if aid < 0]
            raise ValueError(f"Actuators not found in model: {missing}")

        sim_dt = float(self.model.opt.timestep)
        self.sim_steps_per_control = max(1, int(round(self.dt / sim_dt)))

    def _set_base_state(self):
        if self.model.nq < 7:
            return
        pos = self.cfg["init_state"]["pos"]
        rot = self.cfg["init_state"]["rot"]
        self.data.qpos[0:3] = np.array(pos)
        if len(rot) == 4:
            x, y, z, w = rot
            self.data.qpos[3:7] = np.array([w, x, y, z])

    def _set_joint_state(self):
        for idx, adr in enumerate(self.se.joint_qposadr):
            self.data.qpos[adr] = self.default_dof_pos[idx]
        for adr in self.se.joint_qveladr:
            self.data.qvel[adr] = 0.0

    def reset(self):
        self.actions = torch.zeros(12)
        self.timestep = 0
        mujoco.mj_resetData(self.model, self.data)
        self._set_base_state()
        self._set_joint_state()
        mujoco.mj_forward(self.model, self.data)
        return self.get_obs()

    def publish_action(self, action, hard_reset=False):
        self.joint_pos_target = (
            action[0, :12].detach().cpu().numpy() * self.cfg["control"]["action_scale"]
        ).flatten()
        self.joint_pos_target[[0, 3, 6, 9]] *= self.cfg["control"]["hip_scale_reduction"]
        self.joint_pos_target += self.default_dof_pos

        self.joint_vel_target = np.zeros(12)
        self.torques = (
            (self.joint_pos_target - self.dof_pos) * self.p_gains
            + (self.joint_vel_target - self.dof_vel) * self.d_gains
        )

        for i, act_id in enumerate(self.actuator_ids):
            self.data.ctrl[act_id] = self.torques[i]

    def step(self, actions, hard_reset=False):
        clip_actions = self.cfg["normalization"]["clip_actions"]
        self.last_actions = self.actions[:]
        self.actions = torch.clip(actions[0:1, :], -clip_actions, clip_actions)
        self.publish_action(self.actions, hard_reset=hard_reset)

        for _ in range(self.sim_steps_per_control):
            mujoco.mj_step(self.model, self.data)

        obs = self.get_obs()

        frequencies = self.commands[:, 4]
        phases = self.commands[:, 5]
        offsets = self.commands[:, 6]
        if self.num_commands == 8:
            bounds = 0
            durations = self.commands[:, 7]
        else:
            bounds = self.commands[:, 7]
            durations = self.commands[:, 8]
        self.gait_indices = torch.remainder(self.gait_indices + self.dt * frequencies, 1.0)

        if "pacing_offset" in self.cfg["commands"] and self.cfg["commands"]["pacing_offset"]:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + phases]
        else:
            self.foot_indices = [self.gait_indices + phases + offsets + bounds,
                                 self.gait_indices + offsets,
                                 self.gait_indices + bounds,
                                 self.gait_indices + phases]
        self.clock_inputs[:, 0] = torch.sin(2 * np.pi * self.foot_indices[0])
        self.clock_inputs[:, 1] = torch.sin(2 * np.pi * self.foot_indices[1])
        self.clock_inputs[:, 2] = torch.sin(2 * np.pi * self.foot_indices[2])
        self.clock_inputs[:, 3] = torch.sin(2 * np.pi * self.foot_indices[3])

        infos = {"joint_pos": self.dof_pos[np.newaxis, :],
                 "joint_vel": self.dof_vel[np.newaxis, :],
                 "joint_pos_target": self.joint_pos_target[np.newaxis, :],
                 "joint_vel_target": self.joint_vel_target[np.newaxis, :],
                 "body_linear_vel": self.body_linear_vel[np.newaxis, :],
                 "body_angular_vel": self.body_angular_vel[np.newaxis, :],
                 "contact_state": self.contact_state[np.newaxis, :],
                 "clock_inputs": self.clock_inputs[np.newaxis, :],
                 "body_linear_vel_cmd": self.commands[:, 0:2],
                 "body_angular_vel_cmd": self.commands[:, 2:],
                 "privileged_obs": None,
                 "camera_image_front": None,
                 "camera_image_bottom": None,
                 "camera_image_rear": None,
                 "camera_image_left": None,
                 "camera_image_right": None,
                 }

        self.timestep += 1
        return obs, None, None, infos
