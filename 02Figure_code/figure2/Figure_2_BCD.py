import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import math

from matplotlib.colors import to_rgba

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
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

def calculate_fig_width(n_methods, bar_width):
    return max(4, n_methods * bar_width * 1.5)

def plot_bar_manual_x(data, errors, methods, title, ylabel, ylim, colors, save_path=None,bar_width=0.4):

    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.set_facecolor("white")
    x_positions = [
        0.4, 0.9, 1.4, 1.9, 2.4,
        2.9, 3.4, 3.9, 4.4, 4.9,
        5.4, 5.9, 6.4, 6.9, 7.4,
        7.9, 8.4, 8.9, 9.4, 9.9,
        10.4, 10.9, 11.4
    ]

    y_se = errors

    for i, (x, val, err) in enumerate(zip(x_positions, data, y_se)):
        ax.bar(x, val, width=bar_width, color=to_rgba(colors[i], alpha=0.4), label=methods[i], edgecolor= colors[i], linewidth=1)
        ax.errorbar(x, val, yerr=err, fmt="none", ecolor=colors[i], capsize=6, elinewidth=1, capthick=1)

    ax.set_xticks(x_positions)
    ax.set_xticklabels(['' for _ in methods])
    ax.set_xlim(0, x_positions[-1] + bar_width)
    for i, method in enumerate(methods):
        ax.text(x_positions[i] + 0.15, -0.04, method,
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
    plt.show()

methods = ["LCNN","DNNGP", "DeepGS", "DLGWAS", "gMLP", "GEG2P(DL)", "KNN", "RF", "XGBoost", "MLP", "SVR"
           , "GEG2P(ML)","BayesA", "BayesB", "BayesC", "BL", "BRR", "rrBLUP", "LASSO", "SPLS", "RR", "BRNN",
            "GEG2P(SS)"]

class_labels = [
    "Deep Learning", "Deep Learning", "Deep Learning", "Deep Learning", "Deep Learning","GEG2P",
    "Machine Learning", "Machine Learning", "Machine Learning", "Machine Learning", "Machine Learning", "GEG2P",
    "Statistic Method", "Statistic Method", "Statistic Method", "Statistic Method", "Statistic Method",
    "Statistic Method", "Statistic Method", "Statistic Method", "Statistic Method","Statistic Method",  "GEG2P"]


custom_colors = {
    "Deep Learning": "#FF9B45",
    "Machine Learning": "#EE99C2",
    "Statistic Method": "#608BC1",
    "GEG2P": "#686D76"
}

colors = [custom_colors[label] for label in class_labels]
pcc_df = pd.read_csv("pcc.csv")
se_df = pd.read_csv("se.csv")


pcc_df.set_index("Trait", inplace=True)
se_df.set_index("Trait", inplace=True)


for trait in pcc_df.index:
    values = pcc_df.loc[trait, methods].values
    errors = se_df.loc[trait, methods].values
    ylim = get_dynamic_ylim(values, margin_ratio=0.05)

    plot_bar_manual_x(
        data=values,
        errors=errors,
        methods=methods,
        title=trait,
        ylabel="PCC",
        ylim=ylim,
        colors=colors,
        save_path=f"{trait}.png"
    )

