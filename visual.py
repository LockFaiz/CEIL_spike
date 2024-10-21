import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

cur_dir = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(cur_dir,'img')
if not os.path.exists(img_path):
    os.mkdir(img_path)

# data_dirs = ['/storage/songjian/Liu/ceil_code/ceil/logs/Hopper-v2_Hopper1-v2_online_lfd_20/window2_context16_seed2_2024-09-15-15-30-55/progress.csv',
#              '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Hopper-v2_Hopper1-v2_online_lfd_20/window2_context16_seed2_2024-09-19-16-51-13/progress.csv',
#              '/storage/songjian/Liu/ceil_code/ceil/logs/Hopper-v2_Hopper1-v2_online_lfo_20/window2_context16_seed2_2024-09-15-15-40-09/progress.csv',
#              '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Hopper-v2_Hopper1-v2_online_lfo_20/window2_context16_seed2_2024-09-23-03-35-27/progress.csv',
#              '/storage/songjian/Liu/ceil_code/ceil/logs/Ant-v2_Ant1-v2_online_lfd_20/window2_context16_seed2_2024-09-15-16-00-21/progress.csv',
#              '/storage/songjian/Liu/ceil_code/ceil/logs/Ant-v2_Ant1-v2_offline-m_lfo_20/window2_context16_seed2_2024-09-15-17-05-18/progress.csv']
data_dirs = ['/storage/songjian/Liu/ceil_code/ceil/logs/Hopper-v2_Hopper1-v2_online_lfd_20/window2_context16_seed2_2024-09-15-15-30-55/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Hopper-v2_Hopper1-v2_online_lfd_20/window2_context16_seed2_2024-09-19-16-51-13/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/Hopper-v2_Hopper1-v2_online_lfo_20/window2_context16_seed2_2024-09-15-15-40-09/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Hopper-v2_Hopper1-v2_online_lfo_20/window2_context16_seed2_2024-09-23-03-35-27/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/HalfCheetah-v2_HalfCheetah1-v2_online_lfd_20/window2_context16_seed2_2024-09-19-03-11-40/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/HalfCheetah-v2_HalfCheetah1-v2_online_lfo_20/window2_context16_seed2_2024-09-19-03-10-52/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/HalfCheetah-v2_HalfCheetah1-v2_online_lfd_20/window2_context16_seed2_2024-09-19-17-55-52/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/HalfCheetah-v2_HalfCheetah1-v2_online_lfo_20/window2_context16_seed2_2024-09-20-16-29-43/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/Walker2d-v2_Walker2d1-v2_online_lfd_20/window2_context16_seed2_2024-09-19-03-10-52/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Walker2d-v2_Walker2d1-v2_online_lfd_20/window2_context16_seed2_2024-09-19-16-57-48/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/Walker2d-v2_Walker2d1-v2_online_lfo_20/window2_context16_seed2_2024-09-19-03-08-24/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Walker2d-v2_Walker2d1-v2_online_lfo_20/window2_context16_seed2_2024-09-23-02-36-04/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/Ant-v2_Ant1-v2_online_lfd_20/window2_context16_seed2_2024-09-15-16-00-21/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs/Ant-v2_Ant1-v2_online_lfo_20/window2_context16_seed2_2024-09-15-15-44-36/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Ant-v2_Ant1-v2_online_lfd_20/window2_context16_seed2_2024-09-23-09-05-31/progress.csv',
             '/storage/songjian/Liu/ceil_code/ceil/logs_debug_spike/Ant-v2_Ant1-v2_online_lfo_20/window2_context16_seed2_2024-09-23-09-19-25/progress.csv'
            ]

envs = ['hopper','walker2d','halfcheetah','ant']
df_dict = {}
for env in envs:
    for idx, path in enumerate(data_dirs):
        df = pd.read_csv(path)
        if env in path.lower():
            if 'lfd' in path.lower():
                if 'spike' in path.lower():
                    df_dict[env+'_lfd_spike'] = df
                else:
                    df_dict[env+'_lfd'] = df
            elif 'lfo' in path.lower(): 
                if 'spike' in path.lower():
                    df_dict[env+'_lfo_spike'] = df
                else:
                    df_dict[env+'_lfo'] = df

# df = pd.read_csv(data_dirs[0])
# df_s = pd.read_csv(data_dirs[1])
labels_to_plot = [
    'evaluate/ep_rew_mean',
    'evaluate/ep_rew_std',
    'evaluate_source/ep_rew_mean',
    'evaluate_source/ep_rew_std',
    'rollout/ep_rew_mean',
    'rollout/ep_rew_std'
]
def draw(df, df_s, mean, std, saved_name):
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df[mean])), df[mean].to_numpy(), label=saved_name,linewidth=1.5)
    plt.fill_between(range(len(df[mean])), df[mean].to_numpy()-df[std].to_numpy(),df[mean].to_numpy()+df[std].to_numpy(),alpha=0.2)
    plt.plot(range(len(df_s[mean])), df_s[mean].to_numpy(), label=saved_name+'_spike',linewidth=1.5)
    plt.fill_between(range(len(df_s[mean])), df_s[mean].to_numpy()-df_s[std].to_numpy(),df_s[mean].to_numpy()+df_s[std].to_numpy(),alpha=0.2)
    plt.xlabel('steps')
    plt.ylabel('reward')
    plt.legend()
    plt.grid()
    saved_name += '.png'
    plt.savefig(os.path.join(img_path,saved_name), bbox_inches='tight')

def draw_3(df, df_s,saved_name):
    draw(df, df_s,'evaluate/ep_rew_mean','evaluate/ep_rew_std', saved_name+'_eval')
    draw(df, df_s,'evaluate_source/ep_rew_mean','evaluate_source/ep_rew_std', saved_name+'_eval_source')
    draw(df, df_s,'rollout/ep_rew_mean','rollout/ep_rew_std', saved_name+'_rollout')

draw_3(df_dict['hopper_lfd'], df_dict['hopper_lfd_spike'], 'hopper_lfd')
draw_3(df_dict['hopper_lfo'], df_dict['hopper_lfo_spike'], 'hopper_lfo')
draw_3(df_dict['walker2d_lfd'], df_dict['walker2d_lfd_spike'], 'walker2d_lfd')
draw_3(df_dict['walker2d_lfo'], df_dict['walker2d_lfo_spike'], 'walker2d_lfo')
draw_3(df_dict['halfcheetah_lfd'], df_dict['halfcheetah_lfd_spike'], 'halfcheetah_lfd')
draw_3(df_dict['halfcheetah_lfo'], df_dict['halfcheetah_lfo_spike'], 'halfcheetah_lfo')
draw_3(df_dict['ant_lfo'], df_dict['ant_lfo_spike'], 'ant_lfo')
draw_3(df_dict['ant_lfd'], df_dict['ant_lfd_spike'], 'ant_lfd')


# plt.figure(figsize=(12, 6))
# plt.plot(range(len(df[labels_to_plot[0]])), df[labels_to_plot[0]].to_numpy(), label=f"hopper_online_lfd",linewidth=1.5)
# plt.fill_between(range(len(df[labels_to_plot[0]])), df[labels_to_plot[0]].to_numpy()-df[labels_to_plot[1]].to_numpy(),df[labels_to_plot[0]].to_numpy()+df[labels_to_plot[1]].to_numpy(),alpha=0.2)
# plt.plot(range(len(df_s[labels_to_plot[0]])), df_s[labels_to_plot[0]].to_numpy(), label=f"Spike_hopper_online_lfd",linewidth=1.5)
# plt.fill_between(range(len(df_s[labels_to_plot[0]])), df_s[labels_to_plot[0]].to_numpy()-df_s[labels_to_plot[1]].to_numpy(),df_s[labels_to_plot[0]].to_numpy()+df_s[labels_to_plot[1]].to_numpy(),alpha=0.2)
# plt.xlabel('steps')
# plt.ylabel('reward')
# plt.legend()
# plt.grid()
# plt.savefig('./evaluate.png', bbox_inches='tight')

# plt.figure(figsize=(12, 6))
# plt.plot(range(len(df[labels_to_plot[2]])), df[labels_to_plot[2]].to_numpy(), label=f"hopper_online_lfd",linewidth=1.5)
# plt.fill_between(range(len(df[labels_to_plot[2]])), df[labels_to_plot[2]].to_numpy()-df[labels_to_plot[3]].to_numpy(),df[labels_to_plot[2]].to_numpy()+df[labels_to_plot[3]].to_numpy(),alpha=0.2)
# plt.plot(range(len(df_s[labels_to_plot[2]])), df_s[labels_to_plot[2]].to_numpy(), label=f"Spike_hopper_online_lfd",linewidth=1.5)
# plt.fill_between(range(len(df_s[labels_to_plot[2]])), df_s[labels_to_plot[2]].to_numpy()-df_s[labels_to_plot[3]].to_numpy(),df_s[labels_to_plot[2]].to_numpy()+df_s[labels_to_plot[3]].to_numpy(),alpha=0.2)
# plt.xlabel('steps')
# plt.ylabel('reward')
# plt.legend()
# plt.grid()
# plt.savefig('./evaluate_source.png', bbox_inches='tight')

# plt.figure(figsize=(12, 6))
# plt.plot(range(len(df[labels_to_plot[4]])), df[labels_to_plot[4]].to_numpy(), label=f"hopper_online_lfd",linewidth=1.5)
# plt.fill_between(range(len(df[labels_to_plot[4]])), df[labels_to_plot[4]].to_numpy()-df[labels_to_plot[5]].to_numpy(),df[labels_to_plot[4]].to_numpy()+df[labels_to_plot[5]].to_numpy(),alpha=0.2)
# plt.plot(range(len(df_s[labels_to_plot[4]])), df_s[labels_to_plot[4]].to_numpy(), label=f"Spike_hopper_online_lfd",linewidth=1.5)
# plt.fill_between(range(len(df_s[labels_to_plot[4]])), df_s[labels_to_plot[4]].to_numpy()-df_s[labels_to_plot[5]].to_numpy(),df_s[labels_to_plot[4]].to_numpy()+df_s[labels_to_plot[5]].to_numpy(),alpha=0.2)
# plt.xlabel('steps')
# plt.ylabel('reward')
# plt.legend()
# plt.grid()
# plt.savefig('./rollout.png', bbox_inches='tight')
