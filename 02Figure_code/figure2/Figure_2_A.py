import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches
from matplotlib.colors import to_rgba

plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False


file_path = "weight.csv"
df = pd.read_csv(file_path, delimiter=",")


df.columns = df.columns.str.strip()
df["Trait"] = df["Trait"].str.strip()


df = df.set_index("Trait")


flower_traits = {'DTA', 'DTS', 'DTT', 'ATI', 'SAI', 'STI'}
plant_traits = {'EH', 'ELL', 'ELW', 'LNAE', 'LNBE', 'PH', 'TBN', 'TL'}
yield_traits = {'EL', 'ERN', 'EW', 'KNPE', 'KNPR', 'KWPE', 'LBT', 'CW', 'ED'}


models = [  'LCNN','DNNGP', 'DeepGS', 'DLGWAS', 'gMLP', "DL",'KNN', 'Random Forest', 'XGBoost', 'MLP', 'SVR',
          "ML",'BayesA', 'BayesB', 'BayesC', 'BL', 'BRR', 'rrBLUP', 'LASSO', 'SPLS', 'RR', 'BRNN',
           "SS", ]


df_long = df[models].reset_index().melt(id_vars="Trait", var_name="Model", value_name="Value")


df_long["Value"] = pd.to_numeric(df_long["Value"], errors="coerce").fillna(0)


trait_order = list(df_long["Trait"].unique())

plt.figure(figsize=(20, 6.5))

for trait in trait_order:
    trait_data = df_long[df_long["Trait"] == trait]

    for _, row in trait_data.iterrows():
        plt.scatter(
            x=row["Model"], y=row["Trait"],
            s=row["Value"] * 250,
            c="royalblue",
            marker="o",
            alpha=0.7, linewidths=0, edgecolors=None
        )
leftx=-3

GEG2P=0.5
for i, trait in enumerate(trait_order):
    plt.hlines(y=i - 0.5, xmin=-1.5, xmax=len(models) - 0.5, color='black', linewidth=1)
    plt.hlines(y=i + 0.5, xmin=-1.5, xmax=len(models) - 0.5, color='black', linewidth=1)

plt.hlines(y=23 + 0.5+GEG2P, xmin=-0.5, xmax=23 - 0.5, color='black', linewidth=1)
plt.hlines(y=24 + 0.5+GEG2P, xmin=leftx, xmax=23 - 0.5, color='black', linewidth=1)

plt.hlines(y=-0.5, xmin=leftx, xmax=-1.5, color='black', linewidth=1)
plt.hlines(y=8.5, xmin=leftx, xmax=-1.5, color='black', linewidth=1)
plt.hlines(y=16.5, xmin=leftx, xmax=-1.5, color='black', linewidth=1)
plt.hlines(y=22.5, xmin=leftx, xmax=-1.5, color='black', linewidth=1)

for j in range(len(models)):
    plt.vlines(x=j + 0.5, ymin=-0.5, ymax=len(trait_order) + 0.5+GEG2P, color='black', linewidth=1)
    plt.vlines(x=j - 0.5, ymin=-0.5, ymax=len(trait_order) + 0.5+GEG2P, color='black', linewidth=1)

plt.vlines(x=-1.5, ymin=-0.5, ymax=23 - 0.5, color='black', linewidth=1)
plt.vlines(x=leftx, ymin=-0.5, ymax=25 - 0.5+GEG2P, color='black', linewidth=1)

plt.vlines(x=-0.5, ymin=23.5, ymax=24.5+GEG2P, color='black', linewidth=1)
plt.vlines(x=5.5, ymin=23.5, ymax=24.5+GEG2P, color='black', linewidth=1)
plt.vlines(x=11.5, ymin=23.5, ymax=24.5+GEG2P, color='black', linewidth=1)
plt.vlines(x=22.5, ymin=23.5, ymax=24.5+GEG2P, color='black', linewidth=1)

models = [ 'LCNN','DNNGP','DeepGS',  'DLGWAS', 'gMLP',"GEG2P(DL)", 'KNN', 'RF', 'XGBoost', 'MLP', 'SVR',
          "GEG2P(ML)",'BayesA', 'BayesB', 'BayesC', 'BL', 'BRR', 'rrBLUP', 'LASSO', 'SPLS', 'RR', 'BRNN',
            "GEG2P(SS)"]
for i in [0,1,3,4,6,7,9,10,12,13,14,15,16,17,18,19,20,21,]:
    plt.text(x=i, y=23.1, s=models[i], fontsize=10, color="black", fontweight="bold", ha='center')
for i in [2,8]:
    plt.text(x=i, y=23.1, s=models[i], fontsize=9.5, color="black", fontweight="bold", ha='center')
for i in [5,11,22]:
    None
plt.text(x=5, y=23.4, s="GEG2P", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=5, y=22.8, s="(DL)", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=11, y=23.4, s="GEG2P", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=11, y=22.8, s="(ML)", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=22, y=23.4, s="GEG2P", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=22, y=22.8, s="(SS)", fontsize=10, color="black", fontweight="bold", ha='center')

traits = ['EL', 'ERN', 'EW', 'KNPE', 'KNPR', 'KWPE', 'LBT', 'CW', 'ED',
          'EH', 'ELL', 'ELW', 'LNAE', 'LNBE', 'PH', 'TBN', 'TL',
          'DTA', 'DTS', 'DTT', 'ATI', 'SAI', 'STI',]
for i in range(23):
    plt.text(x=-1, y=i-0.1, s=traits[i], fontsize=10, color="black", fontweight="bold", ha='center')

plt.text(x=leftx+0.75, y=4, s="Yield", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=leftx+0.75, y=12.5, s="Plant", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=leftx+0.75, y=11.5, s="Architecture", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=leftx+0.75, y=19.5, s="Flowering", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=leftx+0.75, y=18.5, s="Time", fontsize=10, color="black", fontweight="bold", ha='center')

plt.text(x=2.5, y=24-0.1+GEG2P, s="Deep Learning", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=8.5, y=24-0.1+GEG2P, s="Machine Learning", fontsize=10, color="black", fontweight="bold", ha='center')
plt.text(x=17, y=24-0.1+GEG2P, s="Statistic Method", fontsize=10, color="black", fontweight="bold", ha='center')
#plt.text(x=21, y=24-0.1, s="GEG2P", fontsize=10, color="black", fontweight="bold", ha='center')




ax = plt.gca()


rect1 = patches.Rectangle(xy=(-0.5, 23.5+GEG2P), width=len(models), height=1,  color='gray', alpha=0.4)
ax.add_patch(rect1)
rect1 = patches.Rectangle(xy=(-0.5, 22.5), width=len(models), height=1+GEG2P,  color='orange', alpha=0.2)
ax.add_patch(rect1)
rect1 = patches.Rectangle(xy=(leftx, -0.5), width=-leftx-1.5, height=23,  color='gray', alpha=0.4)
ax.add_patch(rect1)
rect1 = patches.Rectangle(xy=(-1.5, -0.5), width=1, height=23,  color='orange', alpha=0.2)
ax.add_patch(rect1)

rect1 = patches.Rectangle(xy=(4.5, -0.5), width=1, height=23,  color=to_rgba("#686D76", alpha=0.4), alpha=0.2)
ax.add_patch(rect1)
rect1 = patches.Rectangle(xy=(10.5, -0.5), width=1, height=23,  color=to_rgba("#686D76", alpha=0.4), alpha=0.2)
ax.add_patch(rect1)
rect1 = patches.Rectangle(xy=(21.5, -0.5), width=1, height=23,  color=to_rgba("#686D76", alpha=0.4), alpha=0.2)
ax.add_patch(rect1)


plt.grid(False)

for spine in plt.gca().spines.values():
    spine.set_visible(False)

plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)

plt.tight_layout()

plt.xlabel("")
plt.ylabel("")
plt.title("")
plt.xticks([])
plt.yticks([])


plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)


for spine in plt.gca().spines.values():
    spine.set_visible(False)

#
plt.savefig("权重气泡图.png", dpi=300, bbox_inches='tight')
plt.show()