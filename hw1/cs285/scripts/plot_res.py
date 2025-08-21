import matplotlib.pyplot as plt
import seaborn as sns

# 启用 seaborn 风格
sns.set_theme(style="whitegrid", font_scale=1.2)

# 数据
steps = [500, 1000, 1500, 2000, 5000, 8000, 10000, 12000]
eval_avg = [288, 1175, 3143, 4271, 4546, 4700, 4669, 4669]
eval_std = [182, 105, 507, 68, 130, 114, 142, 68]

# 创建画布
plt.figure(figsize=(8,6))

# 平均曲线
plt.plot(steps, eval_avg, marker="o", color="tab:blue", linewidth=2, label="Eval_AverageReturn")

# 方差区域 (均值 ± Std)
plt.fill_between(steps,
                 [a-s for a,s in zip(eval_avg, eval_std)],
                 [a+s for a,s in zip(eval_avg, eval_std)],
                 color="tab:blue", alpha=0.2, label="± StdReturn")

# 细节优化
plt.title("Evaluation Performance on Ant-v4 (BC)", fontsize=14, fontweight="bold")
plt.xlabel("Training Steps")
plt.ylabel("Return")
plt.legend()
plt.tight_layout()
plt.show()
