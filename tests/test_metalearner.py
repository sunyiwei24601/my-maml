from maml_env import HalfCheetahDirecBulletEnv
from policy import *
from baseline import VFBaseLine
from task_worker import TaskWorker
from meta_learner import TRPOMetaLearner
from task_sampler import Tasks

def test_metalearner():
    tasks = Tasks(("Forward", True), ("Backward", False))
    env = HalfCheetahDirecBulletEnv()
    env._max_episode_steps = 200
    policy = get_policy(env, hidden_sizes=(256, 512))
    # share the memory in different processes, so we can work in parallel
    policy.share_memory()
    baseline = VFBaseLine(env.observation_space.shape[0], output_size=1, hidden_sizes=(256, 256))
    metalearner = TRPOMetaLearner(policy)

    # Outer loop
    for meta_iter in range(2):
        train_episodes = []
        test_episodes = []
        for task_config in tasks.sample_tasks(3):
            # Inner loop
            task_name, env_args = task_config[0], task_config[1:]
            env = HalfCheetahDirecBulletEnv(*env_args)
            worker = TaskWorker(policy, env, baseline, train_episodes, test_episodes)
            worker.run()

            # Adaptation
        metalearner.step(train_episodes, test_episodes)