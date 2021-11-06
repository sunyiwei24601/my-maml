from maml_env import HalfCheetahDirecBulletEnv
import random
from policy import *

class Tasks:
    def __init__(self, *task_configs):
        self.tasks = [i for i in task_configs]

    def sample_tasks(self, batch_size):
        return random.choices(self.tasks, k=batch_size)


def main(args):
    tasks = Tasks(("Forward", True), ("Backward", False))
    env = HalfCheetahDirecBulletEnv()
    policy = get_policy(env, hidden_sizes=(256, 512))
    # share the memory in different processes, so we can work in parallel
    policy.share_memory()




    # Outer loop
    for meta_iter in range(args.meta_iteration):
        for task_config in tasks.sample_tasks(args.meta_batch_size):
            # Inner loop
            task_name, env_args = task_config[0], task_config[1:]
            env = HalfCheetahDirecBulletEnv(*env_args)

            # Adaptation

            # Run adapted policy

        # Meta Optimization


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_iteration", default=500, type=int)
    parser.add_argument("--meta_batch_size", default=40, type=int)
    parser.add_argument("--horizon", "-H", default=200, type=int)
    parser.add_argument("--num_adapt_steps", default=1, type=int)

    args = parser.parse_args()

    main(args)
