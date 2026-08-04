import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

from matplotlib.colors import to_rgba

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False

df_raw = pd.read_csv("1.csv", skip_blank_lines=True)


methods = ["SoyDNGP","Cropformer","WheatGP","EGGPT" ,"GEG2P(v3)"]

class_labels = [
    "1",
    "Deep Learning",
    "Machine Learning",
    "Statistic Method",
     "GEG2P"]
custom_colors = {
    "1":"#DA0037",
    "Deep Learning": "#1FAB89",
    "Machine Learning": "#205295",
    "Statistic Method": "#52057B",
    "GEG2P": "#686D76"
}


colors = [custom_colors[label] for label in class_labels]

def get_dynamic_ylim(data, margin_ratio):
    data_min, data_max = min(data), max(data)
    margin = (data_max - data_min) * margin_ratio
    ymin = data_min - 2 * margin
    ymax = data_max + 0.6 * margin

    range_span = ymax - ymin
    if range_span == 0:
        step = 1
    else:
        raw_step = range_span / 6
        exponent = math.floor(math.log10(raw_step))
        fraction = raw_step / 10 ** exponent
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
        step = nice_fraction * 10 ** exponent

    ymin = math.floor(ymin / step) * step
    ymax = math.ceil(ymax / step) * step

    return ymin, ymax, step
def plot_bar_manual_x(data,  methods, title, ylabel, ylim, colors, save_path=None,bar_width=0.4):

    fig, ax = plt.subplots(figsize=(4, 4))
    ax.set_facecolor("white")
    x_positions = [
        0.4, 0.9, 1.4, 1.9, 2.4,


    ]



    for i, (x, val) in enumerate(zip(x_positions, data)):
        ax.bar(x, val, width=bar_width, color=to_rgba(colors[i], alpha=0.4), label=methods[i], edgecolor=colors[i],
               linewidth=1)
        #ax.errorbar(x, val, yerr=err, fmt="none", ecolor=colors[i], capsize=6, elinewidth=1, capthick=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['' for _ in methods])
    ax.set_xlim(0, x_positions[-1] + bar_width)
    for i, method in enumerate(methods):
        ax.text(x_positions[i] + 0.15, -0.025, method,
                rotation=45, ha='right', va='top', fontsize=12, fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.set_ylim(ylim[0], ylim[1])
    plt.yticks(np.arange(ylim[0], ylim[1] + ylim[2], ylim[2]))
    plt.title(title, fontsize=14, fontweight="bold", pad=10)
    plt.ylabel(ylabel, fontsize=14, labelpad=10, fontweight="bold")

    ax.tick_params(axis='y', direction='out', length=5, width=2, color='black', labelsize=14)
    ax.tick_params(axis='x', direction='out', length=5, width=2, color='black', labelsize=14)

    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    sns.despine()
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    #plt.show()

# 拆分数据块（空行为界）
blocks = []
current_block = []
for _, row in df_raw.iterrows():
    if pd.isna(row['Trait']) and pd.isna(row['Way']):
        if current_block:
            blocks.append(pd.DataFrame(current_block))
            current_block = []
    else:
        current_block.append(row)
if current_block:
    blocks.append(pd.DataFrame(current_block))

# 设置物种名称
species_list = ["Wheat", "Rice", "Chickpea", "Soybean","Maize"]

print(blocks)
# 主循环
for i, block in enumerate(blocks):
    species = species_list[i]
    print(species)
    block = block.reset_index(drop=True)
    traits = block['Trait'].unique()

    for trait in traits:
        avg_row = block[(block['Trait'] == trait) & (block['Way'] == 'avg')]
        #se_row = block[(block['Trait'] == trait) & (block['Way'] == 'SE')]
        if avg_row.empty :
            continue

        values = avg_row.iloc[0, 2:].values.astype(float).tolist()
        #errors = se_row.iloc[0, 2:].values.astype(float).tolist()
        ylim = get_dynamic_ylim(values, margin_ratio=0.05)

        save_path = f"{species}-{trait}.png"
        title = f"{trait} ({species})"
        plot_bar_manual_x(values,  methods, title, ylabel="PCC", ylim=ylim, colors=colors, save_path=save_path)
