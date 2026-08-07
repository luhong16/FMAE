import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data'
results_dict = np.load(f'{data_dir}/FMAE_fig5a_pretrain_improvements.npy', allow_pickle=True).item()

def y_fmt(x, pos):
    if x ==0 or x >= 1:
        return int(x)
    return f'{x:.1f}'

a = [results_dict['FMAE_wo_pretrain']['Capacity'], results_dict['FMAE_wo_pretrain']['Anomaly'], results_dict['FMAE_wo_pretrain']['RUL'], results_dict['FMAE_wo_pretrain']['IR']]   # no_pretrain
b = [results_dict['FMAE']['Capacity'], results_dict['FMAE']['Anomaly'], results_dict['FMAE']['RUL'], results_dict['FMAE']['IR']]   # pretrain

# 
y_configs = [
    {'ticks': [0, 1, 2],     'lim': [0, 2]},      # Capacity
    {'ticks': [0, 50, 100], 'lim': [0, 100]},    # Anomaly 
    {'ticks': [0, 50, 100], 'lim': [0, 100]},    # RUL 
    {'ticks': [0, 0.4, 0.8], 'lim': [0, 0.8]} # IR 
]

width = 0.6
script_dir = os.path.dirname(os.path.abspath(__file__))

x_names = ['Capacity\nestimation', 'Anomaly\ndetection', 'RUL\nprediction', 'IR\nestimation']
y_names = ['RMSE (%)', 'AUROC(%)', 'RMSE (cycle)', 'RMSE (mΩ)']

fig, axes = plt.subplots(1, 4, figsize=(8, 3), sharey=False)
orders = [0, 3, 1, 2]
idx = 0

for i in orders:
    ax = axes[idx]
    y_cfg = y_configs[i]
    
    # pretrain
    ax.bar(0, b[i], width=width, color='#EFD75C', zorder=2)
    
    # hatch = gap
    rect_x = 0 - width / 2
    rect_y = min(a[i], b[i])
    rect_height = max(a[i], b[i]) - min(a[i], b[i])
    rect = plt.Rectangle((rect_x, rect_y), width, rect_height,
                         hatch="///", fill=False, edgecolor='red', linewidth=1.5, zorder=3)
    ax.add_patch(rect)
    
    ax.set_ylim(y_cfg['lim'])
    ax.set_yticks(y_cfg['ticks'])
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([0])
    ax.set_xticklabels([x_names[i]], fontsize=8)
    
    ax.set_ylabel(y_names[i], fontsize=9)
    idx += 1


legend_elements = [
    Patch(facecolor='#EFD75C', label='FMAE'),
    Patch(hatch='///', fill=False, edgecolor='red', label='Performance loss\nw/o pretraining')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=2,
           bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=12)

plt.subplots_adjust(wspace=5)
fig.suptitle('Pretraining Improvements on FMAE', fontsize=14, y=1.22)

save_path = os.path.join(script_dir, 'FMAE_fig5a_pretraining_improvements.png')
plt.savefig(save_path, dpi=300, bbox_inches='tight')
plt.show()
plt.close()