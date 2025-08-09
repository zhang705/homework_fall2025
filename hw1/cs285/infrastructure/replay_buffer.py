from cs285.infrastructure.utils import *  # 导入工具函数（例如 convert_listofrollouts）


class ReplayBuffer(object):  # 经验回放缓存，用于存储轨迹与拼接后的数组表示

    def __init__(self, max_size=1000000):  # 初始化回放缓存，指定最大容量

        self.max_size = max_size  # 最大存储条目数（按步计）

        # store each rollout  # 存储每条完整的 rollout（轨迹）
        self.paths = []  # 以列表形式保存每次采样得到的轨迹字典

        # store (concatenated) component arrays from each rollout  # 存储拼接后的组件数组
        self.obs = None  # 所有观测拼接后的数组（按时间步展开）
        self.acs = None  # 所有动作拼接后的数组
        self.rews = None  # 所有奖励拼接后的数组（或列表，取决于 concat_rew）
        self.next_obs = None  # 所有下一时刻观测拼接后的数组
        self.terminals = None  # 所有终止标记拼接后的数组（done 标志）

    def __len__(self):  # 返回当前缓存中样本（时间步）的数量
        if self.obs:  # 若已有观测数组
            return self.obs.shape[0]  # 返回观测的第一维长度（样本数）
        else:  # 否则为空
            return 0  # 样本数为 0

    def add_rollouts(self, paths, concat_rew=True):  # 向缓存添加若干条轨迹，并更新拼接数组

        # add new rollouts into our list of rollouts  # 将新轨迹追加到 paths 列表
        for path in paths:  # 遍历本次收集的每条轨迹
            self.paths.append(path)  # 存入原始轨迹字典（不改变结构）

        # convert new rollouts into their component arrays, and append them onto
        # our arrays  # 将新轨迹转换为组件数组，并准备与已有数组拼接
        observations, actions, rewards, next_observations, terminals = (  # 调用工具函数做结构化转换
            convert_listofrollouts(paths, concat_rew))  # 根据 concat_rew 决定奖励拼接方式

        if self.obs is None:  # 若是第一次写入（尚无历史数据）
            self.obs = observations[-self.max_size:]  # 仅保留最近 max_size 条样本
            self.acs = actions[-self.max_size:]  # 同上
            self.rews = rewards[-self.max_size:]  # 同上
            self.next_obs = next_observations[-self.max_size:]  # 同上
            self.terminals = terminals[-self.max_size:]  # 同上
        else:  # 已有历史数据，需与新数据拼接
            self.obs = np.concatenate([self.obs, observations])[-self.max_size:]  # 观测拼接并截断容量
            self.acs = np.concatenate([self.acs, actions])[-self.max_size:]  # 动作拼接并截断
            if concat_rew:  # 若奖励按时间步拼接为单一数组
                self.rews = np.concatenate(  # 直接数组拼接
                    [self.rews, rewards]
                )[-self.max_size:]  # 截断容量
            else:  # 若奖励按 episode 维度保留（列表），则按列表处理
                if isinstance(rewards, list):  # 若新奖励为列表（多条）
                    self.rews += rewards  # 列表扩展
                else:  # 单条则追加
                    self.rews.append(rewards)  # 追加到列表
                self.rews = self.rews[-self.max_size:]  # 截断容量
            self.next_obs = np.concatenate(  # 下一观测拼接
                [self.next_obs, next_observations]
            )[-self.max_size:]  # 截断容量
            self.terminals = np.concatenate(  # 终止标志拼接
                [self.terminals, terminals]
            )[-self.max_size:]  # 截断容量

