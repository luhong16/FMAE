import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from pathlib import Path

def save_per_sample_importance_maps(npy_path, script_dir):

    data_dict = np.load(npy_path, allow_pickle=True).item()
    grad_enabled_input = data_dict['grad_enabled_input']
    input_grads = data_dict['input_grads']
    batch_size = grad_enabled_input[0].shape[0]
    channel_names = data_dict['channel_names']
    num_snippets = len(grad_enabled_input)
    car_label = data_dict['car_label']
    cycles = data_dict['cycles']
    target_range = 0.9

    print('enabled_grad_input:', grad_enabled_input[0][0].shape, grad_enabled_input[1][0].shape)
    print('enabled_grad_input voltage:', grad_enabled_input[0][0])
    print("input grad:", input_grads[0][0].shape, input_grads[1][0].shape)
    
    # create figs
    fig, axes = plt.subplots(1, num_snippets, figsize=(7*num_snippets + 1, 6))
    
    # calculate sample min and max per channel
    sample_min = [min(grad_enabled_input[0][0][i].min(), grad_enabled_input[1][0][i].min()).item() for i in range(len(channel_names))]
    sample_max = [max(grad_enabled_input[0][0][i].max(), grad_enabled_input[1][0][i].max()).item() for i in range(len(channel_names))]
    print(sample_min)
    print(sample_max)
    
    # meassuring contributions
    contribution_1 = torch.abs(input_grads[0][0] * grad_enabled_input[0][0]).cpu().detach().numpy()
    contribution_2 = torch.abs(input_grads[1][0] * grad_enabled_input[1][0]).cpu().detach().numpy()
    non_zero_mask_1 = contribution_1.sum(axis=1) > 0
    non_zero_mask_2 = contribution_2.sum(axis=1) > 0
    filtered_contribution_1 = contribution_1[non_zero_mask_1][:-1, :]  # remove milleage
    filtered_contribution_2 = contribution_2[non_zero_mask_2][:-1, :]  # remove milleage
    contribution_max = max(filtered_contribution_1.max(), filtered_contribution_2.max())
    contribution_min = min(filtered_contribution_1.min(), filtered_contribution_2.min())
    # print(contribution_max, contribution_min)

    heatmaps = []
    # original gradient
    for plt_idx, snippet_idx in enumerate([1, 0]):
        grad = input_grads[snippet_idx][0]
        orig = grad_enabled_input[snippet_idx][0]
        # print("orig: ", orig)
        # exit()
        contribution = torch.abs(grad * orig).cpu().detach().numpy()
        orig_np = orig.cpu().detach().numpy()
        # filter out full 0s
        non_zero_mask = contribution.sum(axis=1) > 0 
        
        filtered_contribution = contribution[non_zero_mask][:-1, :]
        filtered_channels = [channel_names[i] for i in range(len(channel_names)) if non_zero_mask[i]][:-1]
        # print(filtered_channels)
        filtered_original_input = [orig_np[i, :] for i in range(len(channel_names)) if non_zero_mask[i]][:-1]
        filtered_min = [sample_min[i] for i in range(len(channel_names)) if non_zero_mask[i]][:-1]
        filtered_max = [sample_max[i] for i in range(len(channel_names)) if non_zero_mask[i]][:-1]
        filtered_contribution = (filtered_contribution - contribution_min) / (contribution_max -  contribution_min)

        # plot heatmaps
        im = axes[plt_idx].imshow(
            filtered_contribution, 
            cmap='Blues', 
            aspect='auto',
            interpolation='nearest',
            origin='lower'
        )
        heatmaps.append(im)
        
        # draw original inputs
        for channel_idx in range(len(filtered_channels)): 
            channel_data = filtered_original_input[channel_idx]
            # normalize
            data_min, data_max = filtered_min[channel_idx], filtered_max[channel_idx]
            if data_max - data_min > 1e-6:
                normalized = (channel_data - data_min) / (data_max - data_min)
                normalized *= target_range
            else:
                normalized = np.zeros_like(channel_data)
            
            y_positions = normalized + channel_idx - 0.5
            
            # draw curves
            axes[plt_idx].plot(
                range(len(channel_data)), 
                y_positions, 
                color='Black', 
                linewidth=1.2, 
                alpha=0.8
            )
        # title and label
        axes[plt_idx].set_title(
            f'Charging cycle {cycles[snippet_idx].item()}', 
            fontsize=12
        )
        
        if plt_idx == 0:
            axes[plt_idx].set_ylabel('Channels')
            axes[plt_idx].set_yticks(range(len(filtered_channels)))
            axes[plt_idx].set_yticklabels(filtered_channels, rotation=0)
        else:
            axes[plt_idx].set_yticks(range(len(filtered_channels)))
            axes[plt_idx].set_yticklabels(['' for _ in range(len(filtered_channels))], rotation=0)
            
        axes[plt_idx].set_xlabel('Timestamps (s)')
        
    # Filter out None placeholders
    valid_heatmaps = [h for h in heatmaps if h is not None]
    if valid_heatmaps:
        # Get position of the last valid axis
        pos1 = axes[1].get_position()
        # pos2 = axes[0, 1].get_position()
        height = pos1.y1 - pos1.y0
        
        # Create colorbar axes
        cax = fig.add_axes([
            pos1.x1 + 0.02,  # Left: 2% to the right of subplot
            pos1.y0,          # Bottom: align with subplot bottom
            0.02,            # Width: 2% of figure width
            height       # Height: match subplot height
        ])
        
        # Use any valid heatmap to create colorbar
        fig.colorbar(valid_heatmaps[-1], cax=cax, label='Normalized Gradient Contribution')
    plt.subplots_adjust(right=0.88)  # Make room for colorbar
    # plt.title(f'Battery {car[sample_idx]} with RUL_AE_{int(absolute_error[sample_idx])}')
    plt.savefig(
        os.path.join(script_dir, f'{Path(npy_path).stem}.png'), 
        dpi=150, bbox_inches='tight'
    )
    plt.close()

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data/SI_fig4'

for npy_f_name in os.listdir(data_dir):
    save_per_sample_importance_maps(os.path.join(data_dir, npy_f_name), script_dir)
