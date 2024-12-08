import os
import sys
import time

script_path = os.path.dirname(os.path.realpath(__file__))
track_path = os.path.abspath(os.path.join(script_path, ".."))
repo_path = os.path.abspath(os.path.join(track_path, ".."))
sys.path.append(script_path)
sys.path.append(track_path)
sys.path.append(repo_path)

import gymnasium as gym
import ruamel.yaml as yaml
import torch
from path import Path

from solutions.policies import TD3PolicyForPegInsertionV2
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import (CallbackList,
                                                CheckpointCallback,
                                                EvalCallback)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from utils.common import get_time
from wandb.integration.sb3 import WandbCallback

from loguru import logger as log

import wandb
from arguments import *

algorithm_aliases = {
    "TD3": TD3,
}
TD3.policy_aliases["TD3PolicyForPegInsertionV2"] = TD3PolicyForPegInsertionV2


def make_env(env_name, seed=0, i=0, **env_args):
    wp_device = f"cuda:0"
    def _init():
        env = gym.make(env_name, device=wp_device, **env_args)
        return env

    set_random_seed(seed)

    return _init


if __name__ == "__main__":

    # 准备训练参数
    parser = get_parser()
    # 从命令行获取的参数
    args = parser.parse_args()
    with open(args.cfg, "r") as f:
        cfg = yaml.YAML(typ='safe', pure=True).load(f)

    # solve argument conflict
    # 解决参数覆盖冲突
    cfg = solve_argument_conflict(args, cfg)
    exp_start_time = get_time()
    exp_name = f"{cfg['train']['name']}_{exp_start_time}"
    cfg["train"]["emp"] = {}
    log_dir = os.path.join(track_path, f"training_log/{exp_name}")
    Path(log_dir).makedirs_p()

    with open(os.path.join(log_dir, "cfg.yaml"), "w") as f:
        yaml.YAML(typ='unsafe', pure=True).dump(cfg, f)

    env_name = cfg["env"].pop("env_name")
    params = cfg["env"].pop("params")
    params_lb, params_ub = parse_params(env_name, params)

    if "max_action" in cfg["env"].keys():
        cfg["env"]["max_action"] = np.array(cfg["env"]["max_action"])

    specified_env_args:dict = copy.deepcopy(cfg["env"])
    specified_env_args.update(
        {
            "params": params_lb,
            "params_upper_bound": params_ub,
            "log_folder": log_dir,

        }
    )
    with open(Path(log_dir) / "params_lb.txt", "w") as f:
        f.write(str(params_lb))
    with open(Path(log_dir) / "params_ub.txt", "w") as f:
        f.write(str(params_ub))

    if cfg["train"]["seed"] >= 0:
        seed = cfg["train"]["seed"]
    else:
        seed = int(time.time())
        cfg["train"]["seed"] = seed
    parallel_num = cfg["train"]["parallel"]

    # 创建环境
    env = SubprocVecEnv( 
        [
            make_env(
                env_name,
                seed,
                i,
                **specified_env_args,
            )
            for i in range(parallel_num)
        ]
    )
    eval_env = gym.make(env_name, **specified_env_args)

    # 训练设备
    device = "cpu"
    if torch.cuda.is_available():
        if torch.cuda.device_count() > cfg["train"]["gpu"]:
            device = f"cuda:{cfg['train']['gpu']}"
        else:
            device = "cuda"

    cfg["train"]["device"] = device
    set_random_seed(seed)
    policy_name = cfg["policy"].pop("policy_name")
    # 处理策略参数，这里env.action_space看出action的输出从环境配置中来
    cfg = handle_policy_args(cfg, log_dir, action_dim=env.action_space.shape[0])

    # 加载算法，创建模型
    algorithm_class = algorithm_aliases[cfg["train"]["algorithm_name"]]
    model = algorithm_class(
        policy_name,
        env,
        verbose=1,
        **cfg["policy"],
    )

    # 权重文件
    weight_dir = os.path.join(log_dir, "weights")
    Path(weight_dir).makedirs_p()

    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg["train"]["checkpoint_every"] // parallel_num, 1),
        save_path=weight_dir,
        name_prefix="rl_model",
        save_replay_buffer=False,
        save_vecnormalize=True,
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=max(cfg["train"]["eval_freq"] // parallel_num, 1),
        deterministic=True,
        render=False,
        n_eval_episodes=cfg["train"]["n_eval"],
    )

    WANDB = False
    if WANDB:
        wandb_run = wandb.init(
            project=cfg["train"]["wandb_name"],
            name=f"{cfg['train']['name']}_{exp_start_time}",
            entity="openlock",
            config=cfg,
            sync_tensorboard=True,
            monitor_gym=False,
            save_code=True,
        )
        wandb_callback = WandbCallback(
            verbose=2,
        )
        callback = CallbackList([checkpoint_callback, eval_callback, wandb_callback])
    else:
        callback = CallbackList([checkpoint_callback, eval_callback])

    # 训练模型
    model.learn(
        total_timesteps=cfg["train"]["total_timesteps"], callback=callback, log_interval=cfg["train"]["log_interval"]
    )
    if WANDB:
        wandb_run.finish()
    model.save(os.path.join(log_dir, "rl_model_final.zip"))
    env.close()
