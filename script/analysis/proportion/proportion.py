import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据
data = {
    "Percentage": ["0%", "25%", "50%", "75%", "100%"],
    "Former": [35.53, 21.32, 46.64, 56.41, 61.86],
    "Latter": [35.53, 30.58, 33.65, 45.88, 61.86]
}
df = pd.DataFrame(data)

# 设置论文风格
sns.set_theme(style="whitegrid", font="serif", font_scale=1.2)

# 绘制折线图
plt.figure(figsize=(6,4))
sns.lineplot(data=df, x="Percentage", y="Former", marker="o", label="Former", linewidth=2)
line = sns.lineplot(data=df, x="Percentage", y="Latter", marker="s", label="Latter", linewidth=2)
# 添加baseline水平虚线，使用与折线相同的颜色
plt.axhline(y=71.38, color=line.get_lines()[0].get_color(), linestyle='--', linewidth=2, label='Sharer Model')

# 美化
plt.xlabel("Proportion of C2C Fused KV-Cache", fontsize=13)
plt.ylabel("Accuracy", fontsize=13)
plt.legend(title="", fontsize=11, frameon=False)
plt.tight_layout()

# 保存高分辨率图片
plt.savefig("./proportion.pdf", dpi=300)
plt.show()