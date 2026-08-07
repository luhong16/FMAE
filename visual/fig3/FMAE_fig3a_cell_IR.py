import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data'
results_data = np.load(f'{data_dir}/FMAE_fig3a_cell_IR.npy', allow_pickle=True).item()
print(results_data)
models = results_data['models']
rmse = results_data['rmse']
colors = results_data['colors']

bar_width = 0.3
alpha = 1
figsize = (10,8)
# without numbers
fig, ax = plt.subplots(figsize=figsize)
ax.bar(models, rmse, bar_width, color=colors, alpha=alpha)

ax.tick_params(axis='x', which='major', labelsize=0)
ax.tick_params(axis='y', which='major', labelsize=0)
ax.set_ylim(top=1.0)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'FMAE_fig3a_cell_IR.jpg'))
plt.close()

# with numbers
fig, ax = plt.subplots(figsize=figsize)
ax.bar(models, rmse, bar_width, color=colors, alpha=alpha)

for i, v in enumerate(rmse):
    ax.text(i, v + 0.01, f"{v:.4f}", ha='center', va='bottom', fontsize=20)

ax.set_ylabel('RMSE (mΩ)', fontsize=20)
ax.tick_params(axis='x', which='major', labelsize=15)
ax.tick_params(axis='y', which='major', labelsize=20)
ax.set_ylim(top=1.0)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'FMAE_fig3a_cell_IR_w_numbers.jpg'))
plt.close()