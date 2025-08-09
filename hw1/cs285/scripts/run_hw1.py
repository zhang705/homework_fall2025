"""
Runs behavior cloning and DAgger for homework 1

Functions to edit:
    1. run_training_loop
"""

import pickle  # 导入 pickle，用于读取/保存二进制数据（如专家数据）
import os  # 操作系统相关功能（如路径处理、创建目录）
import time  # 计时与时间戳
import gym  # OpenAI Gym 环境接口

import numpy as np  # 数值计算库
import torch  # PyTorch 深度学习框架

from cs285.infrastructure import pytorch_util as ptu  # PyTorch 工具封装（设备/GPU相关）
from cs285.infrastructure import utils  # 作业提供的常用工具函数
from cs285.infrastructure.logger import Logger  # 日志记录器（标量与视频）
from cs285.infrastructure.replay_buffer import ReplayBuffer  # 经验回放缓存
from cs285.policies.MLP_policy import MLPPolicySL  # 监督学习用 MLP 策略（行为克隆）
from cs285.policies.loaded_gaussian_policy import LoadedGaussianPolicy  # 预训练的高斯策略（专家）


# 保存到 TensorBoard 的视频数量上限  # 注：仅用于可视化抽样的少量轨迹
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # 视频最大长度（稍后会被设为 episode 长度）

# MuJoCo 相关环境名称（作业中可用）
MJ_ENV_NAMES = ["Ant-v4", "Walker2d-v4", "HalfCheetah-v4", "Hopper-v4"]


def run_training_loop(params):  # 训练主循环（支持 BC 与 DAgger）
    """
    Runs training with the specified parameters
    (behavior cloning or dagger)

    Args:
        params: experiment parameters
    """

    #############
    ## INIT
    #############

    # 初始化日志记录器（创建 logdir 并写入标量/视频）
    logger = Logger(params['logdir'])

    # 设置随机种子，保证实验可复现
    seed = params['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    ptu.init_gpu(  # 初始化 GPU/CPU 设备
        use_gpu=not params['no_gpu'],  # 若未指定 no_gpu，则使用 GPU
        gpu_id=params['which_gpu']  # 指定 GPU ID
    )

    # 是否记录视频与标量（按频率控制）
    log_video = True
    log_metrics = True

    #############
    ## ENV
    #############

    # 创建 Gym 环境；训练时 render_mode=None（不渲染）
    env = gym.make(params['env_name'], render_mode=None)
    env.reset(seed=seed)  # 设置环境的随机种子

    # Episode 最大步长（若未显式传入则用环境默认值）
    params['ep_len'] = params['ep_len'] or env.spec.max_episode_steps
    print("params['ep_len']",params['ep_len'])
    MAX_VIDEO_LEN = params['ep_len']  # 将视频长度上限设为单个 episode 的最大步数

    assert isinstance(env.action_space, gym.spaces.Box), "Environment must be continuous"  # 仅支持连续动作空间
    # 观测与动作维度
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.shape[0]

    # 仿真时间步长（用于视频帧率）
    if 'model' in dir(env):  
        fps = 1/env.model.opt.timestep
    else:  # 否则从环境元数据中读取渲染帧率
        fps = env.env.metadata['render_fps']

    #############
    ## AGENT
    #############

    # TODO: Implement missing functions in this class.
    actor = MLPPolicySL(
        ac_dim,  # 动作维度
        ob_dim,  # 观测维度
        params['n_layers'],  # 隐层层数
        params['size'],  # 每层宽度
        learning_rate=params['learning_rate'],  # 学习率
    )

    # 经验回放缓存（用于存放收集的轨迹数据）
    replay_buffer = ReplayBuffer(params['max_replay_buffer_size'])

    #######################
    ## LOAD EXPERT POLICY
    #######################

    print('Loading expert policy from...', params['expert_policy_file'])  # 打印专家策略加载路径
    expert_policy = LoadedGaussianPolicy(params['expert_policy_file'])  # 加载已训练好的专家策略
    expert_policy.to(ptu.device)  # 将专家策略放置到相同设备
    print('Done restoring expert policy...')  # 加载完成

    #######################
    ## TRAINING LOOP
    #######################

    # 训练循环初始变量：环境步数累计与起始时间
    total_envsteps = 0
    start_time = time.time()

    for itr in range(params['n_iter']):  # 迭代轮数（BC: 1，DAgger: >1）
        print("\n\n********** Iteration %i ************"%itr)  # 当前迭代提示

        # 是否在本轮记录视频（按频率）
        log_video = ((itr % params['video_log_freq'] == 0) and (params['video_log_freq'] != -1))
        # 是否在本轮记录标量（按频率）
        log_metrics = (itr % params['scalar_log_freq'] == 0)

        print("\nCollecting data to be used for training...")  # 开始数据收集（用于训练）
        if itr == 0:  # 第 0 轮：仅使用专家离线数据（行为克隆）
            # 从磁盘加载专家演示轨迹（paths 列表）
            paths = pickle.load(open(params['expert_data'], 'rb'))
            envsteps_this_batch = 0  # 未进行环境交互
        else:
            # DAGGER training from sampled data relabeled by expert
            assert params['do_dagger']
            # TODO: collect `params['batch_size']` transitions
            # HINT: use utils.sample_trajectories
            # TODO: implement missing parts of utils.sample_trajectory
            paths, envsteps_this_batch = utils.sample_trajectories(env, actor, params['batch_size'], MAX_VIDEO_LEN)

            # 使用专家策略对收集到的观测进行动作重标注
            if params['do_dagger']:
                print("\nRelabelling collected observations with labels from an expert policy...")

                # TODO: relabel collected obsevations (from our policy) with labels from expert policy
                # HINT: query the policy (using the get_action function) with paths[i]["observation"]
                # and replace paths[i]["action"] with these expert labels
                for path in paths:
                    path['action'] = expert_policy.get_action(path['observation'])

        total_envsteps += envsteps_this_batch  # 累计环境交互步数
        # 将收集的数据加入回放缓存
        replay_buffer.add_rollouts(paths)

        # 训练智能体（从回放缓存中采样小批量进行监督学习）
        print('\nTraining agent using sampled data from replay buffer...')
        training_logs = []  # 存放每次 update 的训练日志（如 loss）
        for _ in range(params['num_agent_train_steps_per_iter']):  # 每轮的梯度更新步数

          # TODO: sample some data from replay_buffer
          # HINT1: how much data = params['train_batch_size']
          # HINT2: use np.random.permutation to sample random indices
          # HINT3: return corresponding data points from each array (i.e., not different indices from each array)
          # for imitation learning, we only need observations and actions.  
          random_indices = np.random.permutation(len(replay_buffer.obs))[:params['train_batch_size']]
          ob_batch, ac_batch = replay_buffer.obs[random_indices], replay_buffer.acs[random_indices]

          # 使用采样的数据更新策略参数
          train_log = actor.update(ob_batch, ac_batch)
          training_logs.append(train_log)  # 记录训练日志

        # 日志与保存
        print('\nBeginning logging procedure...')  # 开始记录日志
        if log_video:
            # 采样评估用的视频轨迹（用于可视化，不参与训练）
            print('\nCollecting video rollouts eval')
            eval_video_paths = utils.sample_n_trajectories(
                env, actor, MAX_NVIDEO, MAX_VIDEO_LEN, True)  # True 表示渲染以保存视频

            # 保存视频到日志
            if eval_video_paths is not None:
                logger.log_paths_as_videos(
                    eval_video_paths, itr,
                    fps=fps,
                    max_videos_to_save=MAX_NVIDEO,
                    video_title='eval_rollouts')

        if log_metrics:
            # 评估指标（采样一批评估轨迹）
            print("\nCollecting data for eval...")
            eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
                env, actor, params['eval_batch_size'], params['ep_len'])  # 评估不做重标注

            # 计算训练/评估相关指标
            logs = utils.compute_metrics(paths, eval_paths)
            # 追加最近一次训练的日志项（例如 loss）
            logs.update(training_logs[-1]) # 目前仅记录最后一次 update 的日志
            logs["Train_EnvstepsSoFar"] = total_envsteps  # 训练至今的环境步数
            logs["TimeSinceStart"] = time.time() - start_time  # 训练耗时
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]  # 初始数据的平均回报

            # 输出并写入日志文件
            for key, value in logs.items():
                print('{} : {}'.format(key, value))  # 控制台打印
                logger.log_scalar(value, key, itr)  # 写入 TensorBoard 标量
            print('Done logging...\n\n')  # 日志完成

            logger.flush()  # 刷新到磁盘

        if params['save_params']:
            print('\nSaving agent params')  # 保存策略参数
            actor.save('{}/policy_itr_{}.pt'.format(params['logdir'], itr))  # 每轮保存一次


def main():  # 命令行入口
    import argparse  # 解析命令行参数
    parser = argparse.ArgumentParser()  # 创建参数解析器
    parser.add_argument('--expert_policy_file', '-epf', type=str, required=True)  # 专家策略文件路径（相对运行目录）
    parser.add_argument('--expert_data', '-ed', type=str, required=True) # 专家数据（演示轨迹）文件路径
    parser.add_argument('--env_name', '-env', type=str, help=f'choices: {", ".join(MJ_ENV_NAMES)}', required=True)  # 环境名称
    parser.add_argument('--exp_name', '-exp', type=str, default='pick an experiment name', required=True)  # 实验名
    parser.add_argument('--do_dagger', action='store_true')  # 是否启用 DAgger（否则仅做 BC）
    parser.add_argument('--ep_len', type=int)  # 每个 episode 的最大步数（可选，不填用默认）

    parser.add_argument('--num_agent_train_steps_per_iter', type=int, default=500)  # 每轮训练中的梯度更新步数
    parser.add_argument('--n_iter', '-n', type=int, default=1)  # 训练轮数（BC=1，DAgger>1）

    parser.add_argument('--batch_size', type=int, default=1000)  # 每轮在环境中收集的训练数据量（步数）
    parser.add_argument('--eval_batch_size', type=int,
                        default=5000)  # 每轮评估时收集的数据量（步数）
    parser.add_argument('--train_batch_size', type=int,
                        default=100)  # 每次 update 使用的样本数

    parser.add_argument('--n_layers', type=int, default=2)  # 策略网络的层数（深度）
    parser.add_argument('--size', type=int, default=64)  # 策略网络每层的宽度
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)  # 监督学习的学习率

    parser.add_argument('--video_log_freq', type=int, default=5)  # 视频记录频率（迭代为单位）
    parser.add_argument('--scalar_log_freq', type=int, default=1)  # 标量记录频率
    parser.add_argument('--no_gpu', '-ngpu', action='store_true')  # 若指定则只用 CPU
    parser.add_argument('--which_gpu', type=int, default=0)  # 指定使用的 GPU ID
    parser.add_argument('--max_replay_buffer_size', type=int, default=1000000)  # 回放缓存最大容量
    parser.add_argument('--save_params', action='store_true')  # 是否保存策略参数文件
    parser.add_argument('--seed', type=int, default=1)  # 随机种子
    args = parser.parse_args()  # 解析命令行参数

    # 将参数对象转换为字典，便于传递
    params = vars(args)

    ##################################
    ### CREATE DIRECTORY FOR LOGGING
    ##################################

    if args.do_dagger:
        # 提交作业时使用该前缀（自动评分器依赖此命名）
        logdir_prefix = 'q2_'
        assert args.n_iter>1, ('DAGGER needs more than 1 iteration (n_iter>1) of training, to iteratively query the expert and train (after 1st warmstarting from behavior cloning).')  # DAgger 需多轮
    else:
        # 提交作业时使用该前缀（自动评分器依赖此命名）
        logdir_prefix = 'q1_'
        assert args.n_iter==1, ('Vanilla behavior cloning collects expert data just once (n_iter=1)')  # 纯 BC 只允许一轮

    # 日志根目录：脚本同级目录的 ../../data
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data')
    if not (os.path.exists(data_path)):
        os.makedirs(data_path)  # 若不存在则创建
    # 日志目录名：前缀_实验名_环境名_时间戳
    logdir = logdir_prefix + args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join(data_path, logdir)  # 拼接成完整路径
    params['logdir'] = logdir  # 保存到参数字典
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)  # 创建日志目录

    ###################
    ### RUN TRAINING
    ###################

    run_training_loop(params)  # 启动训练主循环


if __name__ == "__main__":  # 脚本入口
    main()  # 调用主函数
