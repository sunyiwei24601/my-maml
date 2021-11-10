from maml_env import HalfCheetahDirecBulletEnv
import random

# class Tasks:
#     def __init__(self, *task_configs):
#         self.tasks = [("Forward", True), ("Backward", False)]
# 
#     def sample_tasks(self, batch_size):
#         """
#         given a batch size, return the task env list according to task config distribution
#         :param batch_size:
#         :return:
#         """
#         task_configs = random.choices(self.tasks, k=batch_size)
#         tasks = []
#         for task_config in task_configs:
#             task_name, env_args = task_config[0], task_config[1:]
#             env = HalfCheetahDirecBulletEnv(*env_args)
#             tasks.append(env)
#         return tasks

class Tasks:
    def __init__(self, *task_configs):
        self.tasks = [i for i in task_configs]

    def sample_tasks(self, batch_size):
        return random.choices(self.tasks, k=batch_size)