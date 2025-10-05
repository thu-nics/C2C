import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.ticker import MaxNLocator
from matplotlib import transforms as mtransforms
import matplotlib.pyplot as plt

# # 定义 base 模型
# base_models = ["Pure Teacher", "0.6B", "1.7B", "4B", "8B", "14B"]

# # 定义 teacher size
# teacher_sizes = ["0", "0.5B", "1.5B", "7B", "14B"]

# # 数据表（先空着，后续你填真实 acc 数据）
# # 每个 base 对应一个列表，长度等于 teacher_sizes
# data = {
#     "Pure Teacher": [None,36.49, 58.79, 72.25, 77.98],
#     "0.6B": [35.03, 47.19, 57.62, 69.67, 65.75],
#     "1.7B": [58.26, None,61.43, 71.22, 74.20],
#     "4B":   [71.47, None, None, 73.67, 75.73],
#     "8B":   [75.53, None, None, 75.83, 77.45],
#     "14B":  [79.49, None, None, None, 80.33],
# }

# 定义 base 模型
base_models = ["Pure Teacher", "0.6B", "1.7B", "4B", "8B", "14B"]

# 定义 teacher size
teacher_sizes = ["0", "0.5B", "1.5B", "3B", "7B", "14B"]

# 数据表（先空着，后续你填真实 acc 数据）
# 每个 base 对应一个列表，长度等于 teacher_sizes
data = {
    "Pure Teacher": [None,38.42, 58.79, 63.32, 72.25, 77.98],
    "0.6B": [35.53, 47.19, 57.62, 59.13, 68.43, 67.49],
    "1.7B": [58.26, None,61.43, 63.42, 71.22, 74.20],
    "4B":   [71.47, None, None, 71.22, 73.67, 75.73],
    "8B":   [75.53, None, None, None, 75.83, 77.45],
    "14B":  [79.49, None, None, None, None, 80.33],
}

# data = {
#     "Pure Teacher": [None,38.42, 58.79, 63.32, 72.25, 77.98],
#     "0.6B": [35.53, 41.51, 43.22, 43.32, 43.50, 42.74],
#     "1.7B": [58.26, None,60.00, 60.94, 60.46, 60.40],
#     "4B":   [71.47, None, None, 72.67, 72.66, 73.28],
#     "8B":   [75.53, None, None, None, 76.83, 76.74],
#     "14B":  [79.49, None, None, None, None, 75.41],
# }

# 使用 seaborn 美化风格（学术风格）
sns.set_theme(context="paper", style="whitegrid", font_scale=1.0)

# 统一风格设置（参考 style.md）
FIGSIZE = (6, 3.2)
BASE_FONT_SIZE = 14
LEGEND_FONT_SIZE = 12
TITLE_FONT_SIZE = 14
TEXT_COLOR = "#000000"  # Black
plt.rcParams.update({
    "axes.titlesize": TITLE_FONT_SIZE,
    "axes.labelsize": BASE_FONT_SIZE,
    "xtick.labelsize": BASE_FONT_SIZE,
    "ytick.labelsize": BASE_FONT_SIZE,
    "legend.fontsize": LEGEND_FONT_SIZE,
    "legend.title_fontsize": LEGEND_FONT_SIZE,
    # 文本与坐标轴颜色
    "text.color": TEXT_COLOR,
    "axes.labelcolor": TEXT_COLOR,
    "xtick.color": TEXT_COLOR,
    "ytick.color": TEXT_COLOR,
    # Matplotlib >=3.6 支持 legend.labelcolor
    "legend.labelcolor": TEXT_COLOR,
})

# 为当前绘图排除 teacher-size 为 0 的数据，但保留原始数据以备后用
plot_indices = list(range(1, len(teacher_sizes)))
teacher_sizes_plot = [teacher_sizes[i] for i in plot_indices]

# 将教师规模标签转换为数值（单位：B）以在 x 轴上使用真实间距
def _parse_size(label: str) -> float:
    if label.endswith("B"):
        label = label[:-1]
    return float(label)

numeric_teacher_sizes = [_parse_size(s) for s in teacher_sizes]
x_plot = [numeric_teacher_sizes[i] for i in plot_indices]
pure_teacher_ticks = [data["Pure Teacher"][i] for i in plot_indices]

# 颜色配置：非 Pure Teacher 使用调色板，Pure Teacher 使用灰色虚线
non_pure_models = [m for m in base_models if m != "Pure Teacher"]
palette = sns.color_palette("tab10", n_colors=len(non_pure_models))
model_to_color = {m: palette[i] for i, m in enumerate(non_pure_models)}

# plt.figure(figsize=(6.4, 4), dpi=300)
# plt.xscale('log', base=10)
# ax = plt.gca()
# trans_data_axes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

# for model in base_models:
#     y = data[model]
#     valid_orig_indices = [i for i in plot_indices if y[i] is not None]
#     if not valid_orig_indices:
#         continue
#     plot_x = [numeric_teacher_sizes[i] for i in valid_orig_indices]
#     plot_y = [y[i] for i in valid_orig_indices]

#     if model == "Pure Teacher":
#         sns.lineplot(x=plot_x, y=plot_y, marker="o", color="0.4", linestyle="--", linewidth=2, label=model)
#     else:
#         sns.lineplot(x=plot_x, y=plot_y, marker="o", color=model_to_color[model], linewidth=2, label=model)

    

# plt.xticks(ticks=x_plot, labels=teacher_sizes_plot)
# plt.gca().set_xticks(x_plot)
# plt.gca().set_xticklabels(teacher_sizes_plot)
# plt.xlabel("Teacher Size (B)")
# plt.ylabel("Performance (Acc)")
# plt.title("Performance vs Teacher Size")
# plt.legend(title="Base Model")
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# ax = plt.gca()
# ax.xaxis.set_label_coords(0.5, -0.18)
# plt.gcf().subplots_adjust(bottom=0.26)
# sns.despine()

# plt.tight_layout()
# plt.savefig("scaling_curve.png", bbox_inches="tight")
# plt.show()

# 第二张图：展示相对 base-only 的提升（Δ Accuracy）
plt.figure(figsize=FIGSIZE, dpi=300)
plt.xscale('log', base=10)
ax = plt.gca()
trans_data_axes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

baseline_dot_y = 0
label_y_for_baseline = -0.7

# 使用蓝色渐变为不同曲线着色（最浅 #8FAADC → 最深 #203864）
blue_cmap = mcolors.LinearSegmentedColormap.from_list("blue_grad", ["#8FAADC", "#203864"]) 
blue_shades = [blue_cmap(x) for x in np.linspace(0, 1, len(non_pure_models))]
improv_color_map = {m: blue_shades[i] for i, m in enumerate(non_pure_models)}

for model in non_pure_models:
    y = data[model]
    baseline = y[0]
    valid_orig_indices = [i for i in plot_indices if y[i] is not None]
    if not valid_orig_indices:
        continue
    plot_x = [numeric_teacher_sizes[i] for i in valid_orig_indices]
    plot_y = [y[i] - baseline for i in valid_orig_indices]

    # legend: model size 正常，括号里的 baseline 用斜体
    legend_label = f"{model} $\\mathit{{({baseline:.2f})}}$"
    sns.lineplot(
        x=plot_x, y=plot_y, marker="o", 
        color=improv_color_map[model], linewidth=2,
        label=legend_label
    )

    # 在每条曲线的起点正下方标记空心点
    # first_x = plot_x[0]
    # plt.scatter(
    #     first_x, baseline_dot_y, s=20,
    #     facecolors='none', edgecolors=improv_color_map[model],
    #     linewidths=1.5, zorder=3
    # )

plt.axhline(0, color="0.7", linewidth=1)
plt.ylim(bottom=label_y_for_baseline - 5)

# xticks: size 正常，括号里的 acc 用斜体
_xtick_labels = [
    f"{size}\n${{({acc:.2f})}}$"
    for size, acc in zip(teacher_sizes_plot, pure_teacher_ticks)
]
plt.xticks(ticks=x_plot, labels=_xtick_labels)
plt.gca().set_xticks(x_plot)
plt.gca().set_xticklabels(_xtick_labels)

# plt.xlabel(r"Sharer Model Size $\mathit{(Accuracy)}$")
plt.xlabel("Sharer Model Size (Accuracy)")
plt.ylabel("Δ Accuracy")
# plt.legend(title="Reciever Model Size\n" + r"       $\mathit{(Accuracy)}$")
plt.legend(
    title="Reciever Model Size (Accuracy)",
    ncol=5,
    loc='lower center',
    bbox_to_anchor=(0.5, 1.02),
    frameon=False,
    columnspacing=0.8,
    handlelength=1.2,
    handletextpad=0.4,
    borderpad=0.2,
    labelspacing=0.2,
    fontsize=10,
    title_fontsize=12,
)
plt.gca().set_yticks([0, 10, 20, 30])

ax = plt.gca()
ax.xaxis.set_label_coords(0.5, -0.30)
plt.gcf().subplots_adjust(bottom=0.66, top=0.86)
sns.despine()

plt.tight_layout()
plt.savefig("scaling_improvement_T2T.pdf", bbox_inches="tight")
plt.show()

# # 第三张图：基于 base 错误率的相对提升（Relative Error Reduction）
# plt.figure(figsize=(6.4, 4), dpi=300)
# plt.xscale('log', base=10)
# ax = plt.gca()
# trans_data_axes = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)

# baseline_dot_y = -0.65
# label_y_for_baseline = baseline_dot_y - 0.30

# for model in non_pure_models:
#     y = data[model]
#     baseline = y[0]
#     err_base = max(1e-6, 100.0 - baseline)  # 基线错误率（百分比），防止除零
#     valid_orig_indices = [i for i in plot_indices if y[i] is not None]
#     if not valid_orig_indices:
#         continue
#     plot_x = [numeric_teacher_sizes[i] for i in valid_orig_indices]
#     # 相对错误率降低（百分比）：(y - baseline) / (100 - baseline) * 100
#     plot_y = [((y[i] - baseline) / err_base) * 100.0 for i in valid_orig_indices]
#     legend_label = f"{model} $({baseline:.2f})$"
#     sns.lineplot(x=plot_x, y=plot_y, marker="o", color=model_to_color[model], linewidth=2, label=legend_label)

#     # 在每条曲线的起点下方放置空心圆（y<0）并标注基线绝对值
#     first_x = plot_x[0]
#     plt.scatter(first_x, baseline_dot_y, s=40, facecolors='none', edgecolors=model_to_color[model], linewidths=1.5, zorder=3)
#     ax.text(first_x, -0.08, f"({baseline:.2f})", transform=trans_data_axes, ha="center", va="top", fontsize=BASE_FONT_SIZE, color=TEXT_COLOR, fontstyle="italic", clip_on=False)

# plt.axhline(0, color="0.7", linewidth=1)
# plt.ylim(bottom=label_y_for_baseline - 1)
# plt.xticks(ticks=x_plot, labels=teacher_sizes_plot)
# plt.gca().set_xticks(x_plot)
# plt.gca().set_xticklabels(teacher_sizes_plot)
# plt.xlabel("Teacher Size (B)")
# plt.ylabel("Relative Error Reduction vs Base-only (%)")
# plt.title("Normalized Improvement by Base Error Rate")
# plt.legend(title="Base Model")
# plt.gca().yaxis.set_major_locator(MaxNLocator(nbins=4))
# ax = plt.gca()
# ax.xaxis.set_label_coords(0.5, -0.18)
# plt.gcf().subplots_adjust(bottom=0.28)
# sns.despine()

# plt.tight_layout()
# plt.savefig("scaling_error_reduction.png", bbox_inches="tight")
# plt.show()