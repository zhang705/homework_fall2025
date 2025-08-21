import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

sns.set(style="whitegrid")

# =========================
# Ant-v4 DAgger 数据
# =========================
iterations_ant = np.arange(9)
eval_avg_ant = np.array([
    2784.6467, 3630.6462, 4136.1963, 4591.9321,
    3975.7888, 4540.2837, 4544.5977, 4450.4521, 4719.4033
])
eval_std_ant = np.array([
    1078.4607, 986.5129, 245.2837, 75.1123,
    1025.3670, 126.4263, 126.2622, 66.0279, 108.2885
])
bc_perf_ant = 288.4923
expert_perf_ant = 4712.6003

# =========================
# HalfCheetah-v4 DAgger 数据
# =========================
iterations_cheetah = np.arange(9)
eval_avg_cheetah = np.array([
    3157.0459, 3779.9155, 3869.2258, 4020.7695,
    3954.8457, 4027.5640, 4051.7332, 4044.0938, 4035.1270
])
eval_std_cheetah = np.array([
    69.4558, 133.2185, 109.0602, 96.8649,
    118.9075, 59.0362, 57.9587, 48.8558, 50.5744
])
bc_perf_cheetah = 2103.0227
expert_perf_cheetah = 4067.6677

# =========================
# 绘图函数
# =========================
def plot_dagger_curve(iterations, avg, std, bc, expert, env_name):
    plt.figure(figsize=(10,6))
    plt.plot(iterations, avg, marker='o', label='DAgger')
    plt.fill_between(iterations, avg - std, avg + std, alpha=0.3)
    plt.axhline(bc, color='orange', linestyle='--', label='BC agent')
    plt.axhline(expert, color='green', linestyle='--', label='Expert')
    plt.xlabel('DAgger Iteration')
    plt.ylabel('Eval Average Return')
    plt.title(f'DAgger Learning Curve on {env_name}')
    plt.legend()
    plt.tight_layout()
    plt.show()

# =========================
# 画图
# =========================
plot_dagger_curve(iterations_ant, eval_avg_ant, eval_std_ant, bc_perf_ant, expert_perf_ant, 'Ant-v4')
plot_dagger_curve(iterations_cheetah, eval_avg_cheetah, eval_std_cheetah, bc_perf_cheetah, expert_perf_cheetah, 'HalfCheetah-v4')
