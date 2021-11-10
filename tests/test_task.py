from maml_env import HalfCheetahDirecBulletEnv
import random
from policy import *
from baseline import VFBaseLine
from task_worker import TaskWorker

def test_task_worker():
    env = HalfCheetahDirecBulletEnv(True)
    env._max_episode_steps = 200
    baseline = VFBaseLine(env.observation_space.shape[0], output_size=1, hidden_sizes=(256, 256))
    policy = get_policy(env, hidden_sizes=(256, 512))
    policy.share_memory()
    train_episodes = []
    test_episodes = []
    worker = TaskWorker(policy, env, baseline, train_episodes, test_episodes, inner_loop_steps=2)
    worker.run()
    pass