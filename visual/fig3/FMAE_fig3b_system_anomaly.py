import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = f'{Path(script_dir).parent}/data'
results_dict = np.load(f'{data_dir}/FMAE_fig3b_system_anomaly.npy', allow_pickle=True).item()

print(results_dict)

custom_fpr = results_dict['Dyad']['brand_fprs']
dyad_brand_tprs = results_dict['Dyad']['brand_tprs']
LSTM_brand_tprs = results_dict['LSTM']['brand_tprs']
V_E_brand_tprs = results_dict['Variation_Evaluation']['brand_tprs']
FMAE_brand_tprs = results_dict['FMAE']['brand_tprs']
patchtst_brand_tprs = results_dict['PatchTST']['brand_tprs']
itransformer_brand_tprs = results_dict['iTransformer']['brand_tprs']
timemixer_brand_tprs = results_dict['TimeMixer++']['brand_tprs']

cs = [results_dict[model_name]['color'] for model_name in list(results_dict.keys())]

s_list = [-3, -7, -3, 3, -7, -7, -7]
alphas = [0.3, 0.1, 0.3, 0.3, 0.1, 0.1, 0.05]
linewidth = 5
s = 18

m_width = linewidth-3
fig, ax = plt.subplots(figsize = (10,8))   
# 4.FMAE
FMAE_brand_tprs = np.array(FMAE_brand_tprs)
avg_tpr = np.mean(FMAE_brand_tprs, axis = 0)
std_tpr = np.std(FMAE_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(FMAE_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'FMAE', linewidth=linewidth,color = cs[0], marker='*', markersize=s+s_list[3], markerfacecolor='white', markeredgecolor=cs[0], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[0], color=cs[0]) 

# 7. TimeMixer++
timemixer_brand_tprs = np.array(timemixer_brand_tprs)
avg_tpr = np.mean(timemixer_brand_tprs, axis = 0)
std_tpr = np.std(timemixer_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(timemixer_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'TimeMixer++', linewidth=linewidth,color = cs[6], marker='^', markersize=s+s_list[6], markerfacecolor='white', markeredgecolor=cs[6], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[6], color=cs[6]) 

# 6. iTransformer 
itransformer_brand_tprs = np.array(itransformer_brand_tprs)
avg_tpr = np.mean(itransformer_brand_tprs, axis = 0)
std_tpr = np.std(itransformer_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(itransformer_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'iTransformer', linewidth=linewidth,color = cs[5], marker='o', markersize=s+s_list[5], markerfacecolor='white', markeredgecolor=cs[5], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[5], color=cs[5]) 

# 5. Patchtst 
patchtst_brand_tprs = np.array(patchtst_brand_tprs)
avg_tpr = np.mean(patchtst_brand_tprs, axis = 0)
std_tpr = np.std(patchtst_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(patchtst_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'PatchTST', linewidth=linewidth,color = cs[4], marker='D', markersize=s+s_list[4], markerfacecolor='white', markeredgecolor=cs[4], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[4], color=cs[4]) 

# 2. LSTM
LSTM_brand_tprs = np.array(LSTM_brand_tprs)
avg_tpr = np.mean(LSTM_brand_tprs, axis = 0)
std_tpr = np.std(LSTM_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(LSTM_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'LSTM', linewidth=linewidth,color = cs[1], marker='s', markersize=s+s_list[1], markerfacecolor='white', markeredgecolor=cs[1], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[1], color=cs[1]) 

# 1. DYAD
dyad_brand_tprs = np.array(dyad_brand_tprs)
avg_tpr = np.mean(dyad_brand_tprs, axis = 0)
std_tpr = np.std(dyad_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(dyad_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'DyAD', linewidth=linewidth,color = cs[2], marker='v', markersize=s+s_list[0], markerfacecolor='white', markeredgecolor=cs[2], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[2], color=cs[2]) 

# 3. V-E
V_E_brand_tprs = np.array(V_E_brand_tprs)
avg_tpr = np.mean(V_E_brand_tprs, axis = 0)
std_tpr = np.std(V_E_brand_tprs, axis = 0)
tpr_upper_bound = avg_tpr + std_tpr
tpr_upper_bound[tpr_upper_bound > 1] = 1
tpr_lower_bound = avg_tpr - std_tpr
tpr_lower_bound[tpr_lower_bound < 0] = 0
auc_roc = np.trapz(V_E_brand_tprs, custom_fpr)
plt.plot(custom_fpr, avg_tpr, label = 'V-E', linewidth=linewidth,color = cs[3], marker='p', markersize=s+s_list[2], markerfacecolor='white', markeredgecolor=cs[3], linestyle='-', mew= m_width)
plt.fill_between(custom_fpr, tpr_upper_bound, tpr_lower_bound, alpha=alphas[3], color=cs[3]) 


plt.tick_params(axis='x', which='major', labelsize=0)
plt.tick_params(axis='y', which='major', labelsize=0)
plt.legend(
    loc='lower right',      
    frameon=True,           
    fontsize=28,            
    edgecolor='gray',       
    fancybox=True,          
    shadow=False
)
plt.tight_layout()
plt.savefig(os.path.join(script_dir, f'FMAE_fig3_system_Overall_anomaly_AUROC.jpg'))