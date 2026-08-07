import os 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data'
results_dict = np.load(f'{data_dir}/FMAE_fig3b_system_capacity.npy', allow_pickle=True).item()

print(results_dict)
vit_pretrained_ae_from_dict = np.concatenate(results_dict['System']['FMAE']['AE_list'])
LSTM_ae_from_dict = np.concatenate(results_dict['System']['LSTM']['AE_list'])
RF_ae_from_dict = np.concatenate(results_dict['System']['RF']['AE_list'])
XGBoost_ae_from_dict = np.concatenate(results_dict['System']['XGBoost']['AE_list'])
patchtst_ae_from_dict = np.concatenate(results_dict['System']['PatchTST']['AE_list'])
itransformer_ae_from_dict = np.concatenate(results_dict['System']['iTransformer']['AE_list'])
timemixer_ae_from_dict = np.concatenate(results_dict['System']['TimeMixer++']['AE_list'])

# withour numbers
fig, ax = plt.subplots(figsize = (10,8))

visualize_orders = ['FMAE', 'LSTM', 'XGBoost', 'RF', 'PatchTST', 'iTransformer', 'TimeMixer++']
data = [list(np.concatenate(results_dict['System'][order]['AE_list'])) for order in visualize_orders]
colors = [results_dict['System'][order]['color'] for order in visualize_orders]
positions = [(len(data)-1)/(len(data)+1) *(i+1)  for i in range(0, len(data))]
# box plot
boxplot = plt.boxplot(
    data,
    positions = positions,
    patch_artist=True,  
    showfliers=False,   
    labels=visualize_orders,   
    widths = 0.5,
    medianprops=dict(
        color='black',        
        linewidth=3           
    )
)

ae_means = [np.mean(val) for val in data]
box_upper_bounds = [np.percentile(ae, 75) for ae in data]

# draw mean point
for i in range(len(ae_means)):
    ax.plot(positions[i], ae_means[i], 'black', markersize=8, marker='o',zorder=5)  #
    ax.text(positions[i]+0.25, box_upper_bounds[i]+0.3 , f'{ae_means[i]:.2f}',
            ha='center', va='bottom', fontsize=20, 
            color='black', fontweight='bold')
    ax.plot([positions[i], positions[i]+0.25], [ae_means[i], box_upper_bounds[i]+0.3],color='black', linestyle='-', linewidth=3,zorder=5) # mean to text
    ax.plot([positions[i]+0.07, positions[i]+0.43], [box_upper_bounds[i]+0.3, box_upper_bounds[i]+0.3],color='black', linestyle='-', linewidth=3,zorder=5) # hline
    
# fill color for boxes
for box, color in zip(boxplot["boxes"], colors):
    box.set_facecolor(color)
    box.set_edgecolor("black")
    box.set_linewidth(3)
 
plt.setp(boxplot["whiskers"], color="gray", linestyle="--", linewidth = 3)
plt.tick_params(axis='x', which='major', length = 0, labelsize=0)
plt.tick_params(axis='y', which='major', length = 0, labelsize=0)  
plt.ylim(bottom= 0, top = 10)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'FMAE_fig3b_system_capacity.png'))
plt.close()

# with numbers
fig, ax = plt.subplots(figsize = (10,8))

visualize_orders = ['FMAE', 'LSTM', 'XGBoost', 'RF', 'PatchTST', 'iTransformer', 'TimeMixer++']
data = [list(np.concatenate(results_dict['System'][order]['AE_list'])) for order in visualize_orders]
colors = [results_dict['System'][order]['color'] for order in visualize_orders]
positions = [(len(data)-1)/(len(data)+1) *(i+1)  for i in range(0, len(data))]
# box plot
boxplot = plt.boxplot(
    data,
    positions = positions,
    patch_artist=True,  
    showfliers=False,   
    labels=visualize_orders,   
    widths = 0.5,
    medianprops=dict(
        color='black',        
        linewidth=3           
    )
)

ae_means = [np.mean(val) for val in data]
box_upper_bounds = [np.percentile(ae, 75) for ae in data]

# draw mean point
for i in range(len(ae_means)):
    ax.plot(positions[i], ae_means[i], 'black', markersize=8, marker='o',zorder=5)  # draw mean point
    ax.text(positions[i]+0.25, box_upper_bounds[i]+0.3 , f'{ae_means[i]:.2f}',
            ha='center', va='bottom', fontsize=20, 
            color='black', fontweight='bold') # draw text
    ax.plot([positions[i], positions[i]+0.25], [ae_means[i], box_upper_bounds[i]+0.3],color='black', linestyle='-', linewidth=3,zorder=5) # mean to text
    ax.plot([positions[i]+0.07, positions[i]+0.43], [box_upper_bounds[i]+0.3, box_upper_bounds[i]+0.3],color='black', linestyle='-', linewidth=3,zorder=5) # hline
    
# fill color for boxes
for box, color in zip(boxplot["boxes"], colors):
    box.set_facecolor(color)
    box.set_edgecolor("black")
    box.set_linewidth(3)
 
plt.setp(boxplot["whiskers"], color="gray", linestyle="--", linewidth = 3)
plt.tick_params(axis='x', which='major', labelsize=13)
plt.tick_params(axis='y', which='major', labelsize=20)  
plt.ylim(bottom= 0, top = 10)
ax.set_ylabel('Absolute error (%)', fontsize=20)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'FMAE_fig3b_system_capacity_w_numbers.png'))
plt.close()