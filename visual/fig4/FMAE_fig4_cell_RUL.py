import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data'
results_dict = np.load(f'{data_dir}/FMAE_fig4_cell_RUL.npy', allow_pickle=True).item()

print(results_dict)
models = list(results_dict.keys())
models.remove('Naive_baseline')

brands = results_dict[models[0]]['brands']

brand_mappings = {
    10: 'MIT1',
    12: 'MTI2',
    13: 'KIT'
}


# drawing
for brand_idx, brand in enumerate(brands):
    naive_baseline_rmses = results_dict['Naive_baseline']['brand_rmses'][brand_idx]
    naive_baseline_mapes = results_dict['Naive_baseline']['brand_mapes'][brand_idx]

    model_rmses = [results_dict[model_name]['brand_rmses'][brand_idx] for model_name in models]
    model_mapes = [results_dict[model_name]['brand_mapes'][brand_idx] for model_name in models]
    model_rmses_std_err = [results_dict[model_name]['brand_rmses_std'][brand_idx] for model_name in models]
    model_mapes_std_err = [results_dict[model_name]['brand_mapes_std'][brand_idx] for model_name in models]

    colors = [results_dict[model_name]['color'] for model_name in models]

    # RUL MAPE
    fig, ax = plt.subplots(figsize = (7, 2))

    bars = plt.bar(models, [round(mape) for mape in model_mapes], yerr=model_mapes_std_err, capsize=5, alpha=1, color=colors)
    plt.axhline(y = round(naive_baseline_mapes), color='grey', linestyle='--')
    ax.text(-0.7, round(naive_baseline_mapes)*0.95, f'{round(naive_baseline_mapes)}', 
        ha='left', va='top',                 
        color='grey', fontsize=10, alpha = 0.9,
        fontweight='bold')

    # draw text
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x(),           
            height,                
            f'{int(height)}',      
            ha='center',           
            va='bottom',           
            fontsize=10,
            fontweight='bold'
        )

    plt.tick_params(axis='x', which='major',length=0, labelsize=0)
    plt.tick_params(axis='y', which='major',length=0, labelsize=0)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, f'FMAE_fig4_RUL_{brand_mappings[brand]}_mapes.jpg'))
    plt.close()
    # RUL RMSE
    fig, ax = plt.subplots(figsize = (7, 2))

    bars = plt.bar(models, [round(rmse) for rmse in model_rmses], yerr=model_rmses_std_err, capsize=5, alpha=1, color=colors)
    plt.axhline(y = round(naive_baseline_rmses), color='grey', linestyle='--')
    ax.text(-0.7, round(naive_baseline_rmses)*0.95, f'{round(naive_baseline_rmses)}', 
        ha='left', va='top',                 
        color='grey', fontsize=10, alpha = 0.9,
        fontweight='bold')

    # draw text
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x(),           
            height,                
            f'{int(height)}',      
            ha='center',           
            va='bottom',           
            fontsize=10,
            fontweight='bold'
        )

    plt.tick_params(axis='x', which='major',length=0, labelsize=0)
    plt.tick_params(axis='y', which='major',length=0, labelsize=0)
    plt.tight_layout()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    plt.savefig(os.path.join(script_dir, f'FMAE_fig4_RUL_{brand_mappings[brand]}_rmse.jpg'))
    plt.close()
    # exit()

naive_baseline_rmses = np.mean(results_dict['Naive_baseline']['brand_rmses'])
naive_baseline_mapes = np.mean(results_dict['Naive_baseline']['brand_mapes'])

model_rmses = [np.mean(results_dict[model_name]['brand_rmses']) for model_name in models]
model_mapes = [np.mean(results_dict[model_name]['brand_mapes']) for model_name in models]

# RUL MAPE
fig, ax = plt.subplots(figsize = (7, 2))

bars = plt.bar(models, [round(mape) for mape in model_mapes], capsize=5, alpha=1, color=colors)
plt.axhline(y = round(naive_baseline_mapes), color='grey', linestyle='--')
ax.text(-0.7, round(naive_baseline_mapes)*0.95, f'{round(naive_baseline_mapes)}', 
    ha='left', va='top',                 
    color='grey', fontsize=10, alpha = 0.9,
    fontweight='bold')

# draw text
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x(),           
        height,                
        f'{int(height)}',      
        ha='center',           
        va='bottom',           
        fontsize=10,
        fontweight='bold'
    )

plt.tick_params(axis='x', which='major',length=0, labelsize=0)
plt.tick_params(axis='y', which='major',length=0, labelsize=0)
plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, f'FMAE_fig4_RUL_Overall_mapes.jpg'))
plt.close()
# RUL RMSE
fig, ax = plt.subplots(figsize = (7, 2))

bars = plt.bar(models, [round(rmse) for rmse in model_rmses], capsize=5, alpha=1, color=colors)
plt.axhline(y = round(naive_baseline_rmses), color='grey', linestyle='--')
ax.text(-0.7, round(naive_baseline_rmses)*0.95, f'{round(naive_baseline_rmses)}', 
    ha='left', va='top',                 
    color='grey', fontsize=10, alpha = 0.9,
    fontweight='bold')

# draw text
for bar in bars:
    height = bar.get_height()
    ax.text(
        bar.get_x(),           
        height,                
        f'{int(height)}',      
        ha='center',           
        va='bottom',           
        fontsize=10,
        fontweight='bold'
    )

plt.tick_params(axis='x', which='major',length=0, labelsize=0)
plt.tick_params(axis='y', which='major',length=0, labelsize=0)
plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
plt.savefig(os.path.join(script_dir, f'FMAE_fig4_RUL_Overall_rmse.jpg'))
plt.close()