import numpy as np
import mujoco


def _rpy_from_quaternion(q):
    w, x, y, z = q
    r = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x ** 2 + y ** 2))
    p = np.arcsin(2 * (w * y - z * x))
    y = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y ** 2 + z ** 2))
    return np.array([r, p, y])


class MujocoStateEstimator:
    def __init__(
        self,
        model,
        data,
        body_name="trunk",
        joint_names=None,
        foot_body_names=None,
    ):
        self.model = model
        self.data = data

        self.joint_names = joint_names or [
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        ]
        self.foot_body_names = foot_body_names or [
            "FL_calf", "FR_calf", "RL_calf", "RR_calf",
        ]

        self.body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if self.body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model.")

        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in self.joint_names
        ]
        if any(jid < 0 for jid in self.joint_ids):
            missing = [n for n, jid in zip(self.joint_names, self.joint_ids) if jid < 0]
            raise ValueError(f"Joints not found in model: {missing}")

        self.joint_qposadr = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        self.joint_qveladr = [self.model.jnt_dofadr[jid] for jid in self.joint_ids]

        self.foot_body_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in self.foot_body_names
        ]
        self.foot_body_ids = [bid for bid in self.foot_body_ids if bid >= 0]

        self.joint_idxs = list(range(len(self.joint_names)))
        self.contact_idxs = list(range(len(self.foot_body_names)))

        self.world_lin_vel = np.zeros(3)
        self.world_ang_vel = np.zeros(3)
        self.euler = np.zeros(3)
        self.R = np.eye(3)

        self.mode = 0
        self.ctrlmode_left = 0
        self.ctrlmode_right = 0
        self.left_stick = [0, 0]
        self.right_stick = [0, 0]
        self.left_upper_switch = 0
        self.left_lower_left_switch = 0
        self.left_lower_right_switch = 0
        self.right_upper_switch = 0
        self.right_lower_left_switch = 0
        self.right_lower_right_switch = 0
        self.left_upper_switch_pressed = 0
        self.left_lower_left_switch_pressed = 0
        self.left_lower_right_switch_pressed = 0
        self.right_upper_switch_pressed = 0
        self.right_lower_left_switch_pressed = 0
        self.right_lower_right_switch_pressed = 0

    def _update_body_state(self):
        self.R = self.data.xmat[self.body_id].reshape(3, 3)
        self.world_lin_vel = self.data.xvelp[self.body_id].copy()
        self.world_ang_vel = self.data.xvelr[self.body_id].copy()
        quat = self.data.xquat[self.body_id]
        self.euler = _rpy_from_quaternion(quat)

    def get_body_linear_vel(self):
        self._update_body_state()
        return self.R.T.dot(self.world_lin_vel)

    def get_body_angular_vel(self):
        self._update_body_state()
        return self.R.T.dot(self.world_ang_vel)

    def get_gravity_vector(self):
        self._update_body_state()
        return self.R.T.dot(np.array([0.0, 0.0, -1.0]))

    def get_contact_state(self):
        contact_state = np.zeros(4, dtype=np.float32)
        if not self.foot_body_ids:
            return contact_state

        for i in range(self.data.ncon):
            con = self.data.contact[i]
            body1 = self.model.geom_bodyid[con.geom1]
            body2 = self.model.geom_bodyid[con.geom2]
            if body1 in self.foot_body_ids:
                idx = self.foot_body_ids.index(body1)
                contact_state[idx] = 1.0
            if body2 in self.foot_body_ids:
                idx = self.foot_body_ids.index(body2)
                contact_state[idx] = 1.0
        return contact_state

    def get_rpy(self):
        self._update_body_state()
        return self.euler

    def get_yaw(self):
        self._update_body_state()
        return self.euler[2]

    def get_dof_pos(self):
        return np.array([self.data.qpos[adr] for adr in self.joint_qposadr])

    def get_dof_vel(self):
        return np.array([self.data.qvel[adr] for adr in self.joint_qveladr])

    def get_command(self):
        return np.zeros(3)

    def get_buttons(self):
        return [0, 0, 0, 0]

    def get_camera_front(self):
        return None

    def get_camera_bottom(self):
        return None

    def get_camera_rear(self):
        return None

    def get_camera_left(self):
        return None

    def get_camera_right(self):
        return None
