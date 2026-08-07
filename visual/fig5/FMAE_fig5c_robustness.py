import matplotlib.pyplot as plt
import os
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from pathlib import Path
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data'
results_dict = np.load(f'{data_dir}/FMAE_fig5c_robustness.npy', allow_pickle=True).item()

def y_fmt(x, pos):
    if x == 0 or x >= 1:
        return int(x)
    return f'{x:.1f}'

a = [results_dict['FMAE']['all'], results_dict['LSTM']['all']]          # pretrain/LSTM
b_volt = [results_dict['FMAE']['mask_max_min_volt'], results_dict['LSTM']['mask_max_min_volt']]     # mask max/min volt
b_temp = [results_dict['FMAE']['mask_min_volt_max_min_temp'], results_dict['LSTM']['mask_min_volt_max_min_temp']]     # mask min volt + max/min temp

width = 0.3
x_names = ['FMAE', 'LSTM']
colors = ['#EFD75C', "#F5E6C3"]
rect_color = ['red', '#F20ECC']

configs = [
    {'b': b_volt, 'title': 'Mask Max/Min Volt'},
    {'b': b_temp, 'title': 'Mask Min Volt and Max/Min Temp'},
]

fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), sharey=False)

for idx, (ax, cfg) in enumerate(zip(axes, configs)):
    b = cfg['b']
    
    for i in range(2):
        ax.bar(i, a[i], width=width, color=colors[i], zorder=2)
        
        # hatch gap
        rect_x = i - width / 2
        rect_y = min(a[i], b[i])
        rect_height = abs(a[i] - b[i])
        rect = plt.Rectangle((rect_x, rect_y), width, rect_height,
                             hatch="///", fill=False, edgecolor=rect_color[idx],
                             linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        
    
    
    ax.set_ylim([0, 100])
    ax.set_yticks([0,50, 100])
    ax.yaxis.set_major_formatter(FuncFormatter(y_fmt))
    ax.set_ylabel("AUROC (%)", fontsize=9)
    ax.set_xlim(-0.5, 1.5)          
    ax.set_xticks([0, 1])
    ax.set_xticklabels(x_names, fontsize=8)


colors = ['#EFD75C', "#F5E6C3"]
rect_color = ['red', '##F20ECC']
legend_elements = [
    Patch(facecolor='#EFD75C', label='FMAE'),
    Patch(facecolor='#F5E6C3', label ='LSTM'),
    Patch(hatch='///', fill=False, edgecolor='red', label='Performance loss\nw/o max/min voltage'),
    Patch(hatch='///', fill=False, edgecolor='#F20ECC', label='Performance loss\nw/o max voltage and max/min temp.')
]
fig.legend(handles=legend_elements, loc='upper center', ncol=4,
           bbox_to_anchor=(0.5, 1.12), frameon=False, fontsize=10)

plt.tight_layout()
plt.subplots_adjust(wspace=0.7)
fig.suptitle('Robustness to missed system-level statistics on anomaly detection', fontsize=12, y=1.18)


script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, 'FMAE_fig5c_robustness.png'),
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()