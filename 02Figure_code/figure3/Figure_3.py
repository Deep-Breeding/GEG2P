import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import to_rgba

# 解决中文和负号显示问题
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
maize_data = {
    'Flowering Time': {
        "SVR": [0.650117, 0.015044899],
        "GEG2P(ML)": [0.661406, 0.014709335],
        "GEG2P(v1)": [0.670427, 0.015241889],
        "GEG2P(v2)": [0.671527, 0.015121217],
        "GEG2P(v3)": [0.672603, 0.015121217],
    },
    'Plant Architecture': {
        "SVR": [0.691429, 0.014264408],
        "GEG2P(ML)": [0.701468, 0.014574608],
        "GEG2P(v1)": [0.707282, 0.01436176],
        "GEG2P(v2)": [0.708169, 0.014271882],
        "GEG2P(v3)": [0.708585, 0.014271882],
    },
    'Yield': {
        "SVR": [0.57464, 0.02017711],
        "GEG2P(ML)": [0.589557, 0.018490029],
        "GEG2P(v1)": [0.595739, 0.019183909],
        "GEG2P(v2)": [0.5976, 0.01913599],
        "GEG2P(v3)": [0.598253, 0.01913599],
    }
}
wheat_data = {
    "Maturity": {
        "SVR": [0.797771581,0.003127],
        "GEG2P(ML)": [0.816573414, 0.002799],
        "GEG2P(v1)": [0.822392459, 0.002846],
        "GEG2P(v2)": [0.82235763, 0.002646],
        "GEG2P(v3)": [0.822978792, 0.002736],
    },
    "Yield": {
        "BayesB": [0.598046443,0.004324667],
        "GEG2P(ML)": [0.592402805, 0.00507],
        "GEG2P(v1)": [0.615447092, 0.0045235],
        "GEG2P(v2)": [0.616158158, 0.004527],
        "GEG2P(v3)": [0.616158158, 0.004527],
    }
}
rice_data = {
    "Heading_date": {
        "SVR": [0.907992,0.008009605],
        "GEG2P(ML)": [0.91644, 0.007762376],
        "GEG2P(v1)": [0.918822, 0.007953119],
        "GEG2P(v2)": [0.919648, 0.008061058],
        "GEG2P(v3)": [0.919755608, 0.008060327],
    },
    "Height": {
        "SVR": [0.867154,0.008027768],
        "GEG2P(ML)": [0.880642, 0.006977273],
        "GEG2P(v1)": [0.884453, 0.006422698],
        "GEG2P(v2)": [0.884852, 0.006515382],
        "GEG2P(v3)": [0.884852, 0.006515382],
    },
    "Yield_per_plant": {
        "SVR": [0.398885,0.020059008],
        "GEG2P(ML)": [0.432309, 0.019257376],
        "GEG2P(v1)": [0.44931, 0.019390187],
        "GEG2P(v2)": [0.449091, 0.019372691],
        "GEG2P(v3)": [0.449091, 0.019372691],
    }
}
chickpea_data = {
    "Days_to_0.5_flowering": {
        "BayesC": [0.490883,0.011861001],
        "GEG2P(SS)": [0.502982, 0.011766775],
        "GEG2P(v1)": [0.511143, 0.011132012],
        "GEG2P(v2)": [0.51173, 0.010954477],
        "GEG2P(v3)": [0.512469909, 0.01095],
    },
    "Plant_height": {
        "RF": [0.589329,0.01251158],
         "GEG2P(ML)": [0.601056, 0.012394106],
        "GEG2P(v1)": [0.602732, 0.011959948],
        "GEG2P(v2)": [0.603802, 0.012066699],
        "GEG2P(v3)": [0.604053363, 0.012425],
    },
    "Yield": {
        "rrBLUP": [0.468636,0.014810577],
        "GEG2P(SS)": [0.473851, 0.015167523],
        "GEG2P(v1)": [0.481985, 0.015382829],
        "GEG2P(v2)": [0.482265, 0.015301352],
        "GEG2P(v3)": [0.482265, 0.015301352],
    }
}
soy_data = {
    "BBD": {
        "BayesB": [0.841435885,0.00589414],
        "GEG2P(SS)": [0.848956424, 0.005308419],
        "GEG2P(v1)": [0.854975094, 0.004610713],
        "GEG2P(v2)": [0.855219531, 0.004613],
        "GEG2P(v3)": [0.855219531, 0.004613],
    },
    "PLH": {
        "SVR": [0.781202398,0.011195744],
        "GEG2P(ML)": [0.800374598, 0.012394106],
        "GEG2P(v1)": [0.809278552, 0.011959948],
        "GEG2P(v2)": [0.809167, 0.012066699],
        "GEG2P(v3)": [0.809167, 0.012425],
    },
    "Yield": {
        "LASSO": [0.670714235,0.015751724],
        "GEG2P(SS)": [0.69683089, 0.013712793],
        "GEG2P(v1)": [0.705123316, 0.012325856],
        "GEG2P(v2)": [0.706459147, 0.012197728],
        "GEG2P(v3)": [0.706820163,0.012239909]
    }
}



#colors = ["#FFE299", "#CAF4FF", "#A0DEFF", "#5AB2FF", "#30E3CA"]
colors = ["#F16767", "#686D76", "#AEEA94", "#5DB996", "#118B50"]


sns.set_style("white")


def calculate_fig_width(num_bars, bar_width=0.6, spacing=0.2):
    return 2 * (num_bars * (bar_width + spacing) + spacing)

import math

def get_dynamic_ylim(data, margin_ratio):
    data_min, data_max = min(data), max(data)
    margin = (data_max - data_min) * margin_ratio
    ymin = data_min - 2 * margin
    ymax = data_max + 0.6*margin


    range_span = ymax - ymin
    if range_span == 0:
        step = 1
    else:
        raw_step = range_span / 6
        exponent = math.floor(math.log10(raw_step))
        fraction = raw_step / 10**exponent
        if fraction < 1.5:
            nice_fraction = 1
        elif fraction < 3:
            nice_fraction = 2
        elif fraction < 7:
            nice_fraction = 5
        else:
            nice_fraction = 10
        step = nice_fraction * 10**exponent


    ymin = math.floor(ymin / step) * step
    ymax = math.ceil(ymax / step) * step

    return ymin, ymax,step




def plot_bar_manual_x(data, errors, methods, title, ylabel, ylim, colors, bar_width=0.4, save_path=None):
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.set_facecolor("white")

    x_positions = [0.5,1.0,1.5,2.0,2.5]


    y_se = errors
    for i, (x, val, err) in enumerate(zip(x_positions, data, y_se)):
        ax.bar(x, val, width=bar_width, color=to_rgba(colors[i], alpha=0.4), label=methods[i], edgecolor=colors[i],
               linewidth=1)
        ax.errorbar(x, val, yerr=err, fmt="none", ecolor=colors[i], capsize=6, elinewidth=1, capthick=1)


    ax.set_xticks(x_positions)
    ax.set_xticklabels(['' for _ in methods])


    for i, method in enumerate(methods):
        ax.text(x_positions[i]+0.1, -0.025, method,
                rotation=45,
                ha='right', va='top',
                fontsize=12,
                fontweight='bold',
                transform=ax.get_xaxis_transform())

    ax.set_ylim(ylim[0], ylim[1])
    plt.yticks(np.arange(ylim[0], ylim[1] + ylim[2], ylim[2]))
    plt.title(title, fontsize=12, fontweight="bold", loc='center', pad=10)
    #plt.xlabel("", fontsize=12, labelpad=5)
    plt.ylabel(ylabel, fontsize=12, labelpad=10, fontweight="bold")

    ax.tick_params(axis='y', which='both', direction='out', bottom=True, top=False, left=True, right=False, length=5, width=2, color='black', labelsize=12)
    ax.tick_params(axis='x', which='both', direction='out', bottom=True, top=False, left=True, right=False, length=5, width=2, color='black', labelsize=12)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    sns.despine()
    plt.tight_layout()
    #plt.show()

    plt.savefig(save_path, dpi=300, bbox_inches="tight")

plant_datasets = {
    "Wheat": wheat_data,
    "Rice": rice_data,
    "Chickpea": chickpea_data,
    "Soy": soy_data,
    "Maize":maize_data
}
plant_datasets = {
    "Maize":maize_data
}
for plant, plant_data in plant_datasets.items():
    for trait, results in plant_data.items():
        methods = list(results.keys())
        pcc_values = [results[m][0] for m in methods]
        pcc_errors = [results[m][1] for m in methods]

        pcc_ylim = get_dynamic_ylim(pcc_values, margin_ratio=2)
        plot_bar_manual_x(pcc_values, pcc_errors, methods,
                          #title=f"{trait} ({plant})",
                          title=f"{trait} ",
                          ylabel=f"PCC",
                          ylim=pcc_ylim, colors=colors[:len(methods)],
                          save_path=f"{plant}_{trait}_PCC.png")