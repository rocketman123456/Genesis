import argparse
import os
import pickle

import torch
from go2_env import Go2Env
from rsl_rl.runners import OnPolicyRunner

import genesis as gs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="go2-walking")
    parser.add_argument("--ckpt", type=int, default=100)
    parser.add_argument("-v", "--vis", action="store_true", default=False)
    args = parser.parse_args()

    gs.init()

    log_dir = f"logs/{args.exp_name}"
    env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(open(f"logs/{args.exp_name}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}

    env = Go2Env(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        # device="cuda:0",
        device="mps",
        show_viewer=True,
    )

    runner = OnPolicyRunner(
        env,
        train_cfg,
        log_dir,
        # device="cuda:0",
        device="mps",
    )
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    # policy = runner.get_inference_policy(device="cuda:0")
    policy = runner.get_inference_policy(device="mps")

    gs.tools.run_in_another_thread(fn=run_sim, args=(env, policy, args.vis))
    if args.vis:
        env.scene.viewer.start()

    # obs, _ = env.reset()
    # with torch.no_grad():
    #     while True:
    #         actions = policy(obs)
    #         obs, _, rews, dones, infos = env.step(actions)


def run_sim(env, policy, enable_vis):
    obs, _ = env.reset()
    with torch.no_grad():
        i = 0
        while True:
            actions = policy(obs)
            obs, _, rews, dones, infos = env.step(actions)

            i += 1
            if i > 50 * 100:
                break

    if enable_vis:
        env.scene.viewer.stop()


if __name__ == "__main__":
    main()

"""
# evaluation
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100
"""
