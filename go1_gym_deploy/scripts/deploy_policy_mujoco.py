import argparse
import glob
import pickle as pkl
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
import torch

from go1_gym_deploy.envs.mujoco_agent import MujocoAgent
from go1_gym_deploy.envs.history_wrapper import HistoryWrapper
from go1_gym_deploy.utils.mujoco_state_estimator import MujocoStateEstimator
from go1_gym_deploy.utils.command_profile import MujocoKeyboardProfile


def class_to_dict(obj):
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


class FixedCommandProfile:
    def __init__(self, dt, num_commands, command):
        self.dt = dt
        self.num_commands = num_commands
        self.command = command

    def get_command(self, t, probe=False):
        return self.command, False

    def get_buttons(self):
        return [0, 0, 0, 0]


def load_policy(logdir):
    body = torch.jit.load(logdir + "/checkpoints/body_latest.jit")
    adaptation_module = torch.jit.load(logdir + "/checkpoints/adaptation_module_latest.jit")

    def policy(obs, info):
        latent = adaptation_module.forward(obs["obs_history"].to("cpu"))
        action = body.forward(torch.cat((obs["obs_history"].to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def load_cfg(label):
    dirs = glob.glob(f"../../runs/{label}/*")
    if not dirs:
        raise FileNotFoundError(f"no run dirs found for label: {label}")
    logdir = sorted(dirs)[0]

    with open(logdir + "/parameters.pkl", "rb") as file:
        pkl_cfg = pkl.load(file)
        cfg = pkl_cfg["Cfg"]
        if not isinstance(cfg, dict):
            cfg = class_to_dict(cfg)
    return logdir, cfg


def make_command(num_commands, x_vel, y_vel, yaw_vel):
    command = np.zeros(num_commands, dtype=np.float32)
    command[0] = x_vel
    if num_commands > 1:
        command[1] = y_vel
    if num_commands > 2:
        command[2] = yaw_vel
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", default="gait-conditioned-agility/pretrain-v0/train")
    default_xml = (Path(__file__).resolve().parent.parent.parent / "resources/robots/go1/xml/go1.xml")
    parser.add_argument("--xml", default=str(default_xml))
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--x", type=float, default=0.0)
    parser.add_argument("--y", type=float, default=0.0)
    parser.add_argument("--yaw", type=float, default=0.0)
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--x-scale", type=float, default=1.0)
    parser.add_argument("--y-scale", type=float, default=1.0)
    parser.add_argument("--yaw-scale", type=float, default=1.0)
    args = parser.parse_args()

    logdir, cfg = load_cfg(args.label)

    model = mujoco.MjModel.from_xml_path(args.xml)
    data = mujoco.MjData(model)

    se = MujocoStateEstimator(model, data)
    dt = cfg["control"]["decimation"] * cfg["sim"]["dt"]
    if args.keyboard:
        viewer = mujoco.viewer.launch_passive(model, data)
        command_profile = MujocoKeyboardProfile(
            dt=dt,
            viewer=viewer,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
            yaw_scale=args.yaw_scale,
        )
    else:
        command = make_command(cfg["commands"]["num_commands"], args.x, args.y, args.yaw)
        command_profile = FixedCommandProfile(dt=dt, num_commands=cfg["commands"]["num_commands"], command=command)

    agent = MujocoAgent(cfg, se, command_profile, model, data)
    agent = HistoryWrapper(agent)
    policy = load_policy(logdir)

    obs = agent.reset()
    for _ in range(args.steps):
        info = {}
        action = policy(obs, info)
        obs, _, _, _ = agent.step(action)
        if args.keyboard:
            viewer.sync()


if __name__ == "__main__":
    main()
