import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgba

# 自定义颜色（统一边框色和填充色，带 alpha=0.4 透明度）
custom_colors = {
    "Deep Learning": "#FF9B45",
    "Machine Learning": "#EE99C2",
    "Statistic Method": "#608BC1",
    "GEG2P": "#686D76"
}

# 创建 legend 元素：填充和边框用相同颜色，透明度为 0.4
legend_elements = [
    Patch(
        facecolor=to_rgba(color, alpha=0.4),
        edgecolor=to_rgba(color, alpha=1.0),  # 边框同色，但不透明
        linewidth=1.5,
        label=label
    )
    for label, color in custom_colors.items()
]

# 创建空白图，仅展示图例
fig, ax = plt.subplots(figsize=(6, 2))
ax.axis('off')  # 关闭坐标轴

# 图例：无外部边框，仅色块有边框
legend = ax.legend(
    handles=legend_elements,
    loc='center',
    ncol=len(custom_colors),
    frameon=False,      # 不显示图例整体框
    fontsize=12
)

plt.tight_layout()
plt.savefig("legend_only.png", dpi=300, bbox_inches='tight')
plt.show()
