from maml_env import HalfCheetahDirecBulletEnv
from policy import *
from baseline import VFBaseLine
from task_worker import TaskWorker
from meta_learner.trpo_learner import TRPOMetaLearner
from meta_learner.ppo_learner import PPOMetaLearner
from task_sampler import Tasks
import time
from multiprocessing import Pool
from torch.utils.tensorboard import SummaryWriter
import os

def main(args):
    dir_name = os.path.join("result", "{}_{}_{}_{}.log".format(args.meta_iteration, args.algo, args.meta_batch_size, time.time()))

    writer = SummaryWriter(log_dir=dir_name)

    tasks = Tasks(("Forward", True), ("Backward", False))
    env = HalfCheetahDirecBulletEnv()
    env._max_episode_steps = 200
    policy = get_policy(env, hidden_sizes=(256, 512))
    # share the memory in different processes, so we can work in parallel
    policy.share_memory()
    baseline = VFBaseLine(env.observation_space.shape[0], output_size=1, hidden_sizes=(256, 256))
    if args.algo == "TRPO":
        metalearner = TRPOMetaLearner(policy)
    elif args.algo == "PPO":
        metalearner = PPOMetaLearner(policy)

    policy_filename = os.path.join("saved_policy", "{}_{}_{}_{}.th".format(args.meta_iteration, args.algo, args.meta_batch_size, time.time())) 
    previous_loss = 1e20

    # Outer loop
    for meta_iter in range(args.meta_iteration):
        train_episodes = []
        test_episodes = []
        start = time.time()
        # p = Pool(args.multi_process)
        results = []
        for task_config in tasks.sample_tasks(args.meta_batch_size):
            # Inner loop
            task_name, env_args = task_config[0], task_config[1:]
            env = HalfCheetahDirecBulletEnv(*env_args)
            worker = TaskWorker(policy, env, baseline, train_episodes, test_episodes)
            worker.run()
            # results.append(p.apply_async(worker.run))
        # p.close()
        # p.join()
        # 
        # for res in results:
        #     train_episode_pairs, test_episode = res.get()
        #     train_episodes.append(train_episode_pairs)
        #     test_episodes.append(test_episode)

        end = time.time()
        print("Sample Trajectories cost: ", end-start)

        # Meta Optimization
        loss = metalearner.step(train_episodes, test_episodes)
        print("{}  loss: {}, Updating MetaLearner Cost time: {}".format(time.ctime(), loss, time.time() - end))
        writer.add_scalar("meta_learning_curve", loss, meta_iter)

        # only save policy if it have a better result
        if loss < previous_loss:
            previous_loss = loss
            with open(policy_filename, 'wb') as f:
                torch.save(policy.state_dict(), f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--meta_iteration", default=500, type=int)
    parser.add_argument("--meta_batch_size", default=20, type=int)
    parser.add_argument("--horizon", "-H", default=200, type=int)
    parser.add_argument("--num_adapt_steps", default=1, type=int)
    parser.add_argument("--algo", default="PPO", type=str)
    parser.add_argument("--multi_process", default=1, type=int)

    args = parser.parse_args()

    main(args)
