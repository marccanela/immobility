"""
Version: 20/12/2022
@author: mcanela
"""
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as mpatches
import seaborn as sns
import numpy as np
import pandas as pd
import pingouin as pg
from scipy.stats import pearsonr
from scipy.integrate import simps
import statistics
import copy 

# =============================================================================
# TIMESERIES PLOTS
# =============================================================================

def timeseries_SOC_rectangles_plot_bin10(df, protocol='soc', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    df_melted['Time bin'] = [str(int(int(timelapse[9:])*(bin_size/10))) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                 # units="mouse", estimator=None  
                 )
    plt.ylim(0,100)
    handles, labels = ax.get_legend_handles_labels()
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 100) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((23, 0), 6, 100) # 1 min OFF
    off_list.append(off2_coords)
    off3_coords = plt.Rectangle((35, 0), 6, 100) # 1 min OFF
    off_list.append(off3_coords)
    off4_coords = plt.Rectangle((47, 0), 6, 100) # 1 min OFF
    off_list.append(off4_coords)
    off5_coords = plt.Rectangle((59, 0), 6, 100) # 1 min OFF
    off_list.append(off5_coords)
    # From this point, more than 4 trials
    off6_coords = plt.Rectangle((71, 0), 6, 100) # 1 min OFF
    off_list.append(off6_coords)
    off7_coords = plt.Rectangle((83, 0), 6, 100) # 1 min OFF
    off_list.append(off7_coords)
    off8_coords = plt.Rectangle((95, 0), 6, 100) # 1 min OFF
    off_list.append(off8_coords)
    off9_coords = plt.Rectangle((107, 0), 6, 100) # 1 min OFF
    off_list.append(off9_coords)
    #
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    tone_list = []
    tone1_coords = plt.Rectangle((17, 0), 3, 100) # 30 s tone
    tone_list.append(tone1_coords)
    tone2_coords = plt.Rectangle((29, 0), 3, 100) # 30 s tone
    tone_list.append(tone2_coords)
    tone3_coords = plt.Rectangle((41, 0), 3, 100) # 30 s tone
    tone_list.append(tone3_coords)
    tone4_coords = plt.Rectangle((53, 0), 3, 100) # 30 s tone
    tone_list.append(tone4_coords)
    # From this point, more than 4 trials
    tone5_coords = plt.Rectangle((65, 0), 3, 100) # 30 s tone
    tone_list.append(tone5_coords)
    tone6_coords = plt.Rectangle((77, 0), 3, 100) # 30 s tone
    tone_list.append(tone6_coords)
    tone7_coords = plt.Rectangle((89, 0), 3, 100) # 30 s tone
    tone_list.append(tone7_coords)
    tone8_coords = plt.Rectangle((101, 0), 3, 100) # 30 s tone
    tone_list.append(tone8_coords)
    # 
    tones_coll = PatchCollection(tone_list, alpha=0.1, color='blue')
    ax.add_collection(tones_coll)
    tone_coll_border = PatchCollection(tone_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(tone_coll_border)
    tone_patch = mpatches.Patch(color='blue', label='tone', alpha=0.1)
    handles.append(tone_patch)
    
    light_list = []
    light1_coords = plt.Rectangle((20, 0), 3, 100) # 30 s tone
    light_list.append(light1_coords)
    light2_coords = plt.Rectangle((32, 0), 3, 100) # 30 s tone
    light_list.append(light2_coords)
    light3_coords = plt.Rectangle((44, 0), 3, 100) # 30 s tone
    light_list.append(light3_coords)
    light4_coords = plt.Rectangle((56, 0), 3, 100) # 30 s tone
    light_list.append(light4_coords)
    # From this point, more than 4 trials
    light5_coords = plt.Rectangle((68, 0), 3, 100) # 30 s tone
    light_list.append(light5_coords)
    light6_coords = plt.Rectangle((80, 0), 3, 100) # 30 s tone
    light_list.append(light6_coords)
    light7_coords = plt.Rectangle((92, 0), 3, 100) # 30 s tone
    light_list.append(light7_coords)
    light8_coords = plt.Rectangle((104, 0), 3, 100) # 30 s tone
    light_list.append(light8_coords)
    # 
    lights_coll = PatchCollection(light_list, alpha=0.1, color='red', edgecolor='black')
    ax.add_collection(lights_coll)
    light_coll_border = PatchCollection(light_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(light_coll_border)
    light_patch = mpatches.Patch(color='red', label='light', alpha=0.1)
    handles.append(light_patch)
    
    plt.legend(handles=handles)
    
    return ax


def timeseries_simultaneous_SOC_rectangles_plot_bin10(df, protocol='soc', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    df_melted['Time bin'] = [str(int(int(timelapse.split("_")[-1])*(bin_size/1))/60) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                 # units="mouse", estimator=None  
                 )

    plt.ylim(0,100)
    ax.set(xlabel='Time [min]')
    handles, labels = ax.get_legend_handles_labels()
    
    # To plot only every 6th tick on the X axis
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[5::6]))
    for label in temp:
        label.set_visible(False)
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 100) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((20, 0), 6, 100) # 1 min OFF
    off_list.append(off2_coords)
    off3_coords = plt.Rectangle((29, 0), 12, 100) # 2 min OFF
    off_list.append(off3_coords)
    off4_coords = plt.Rectangle((44, 0), 9, 100) # 1.5 min OFF
    off_list.append(off4_coords)
    off5_coords = plt.Rectangle((56, 0), 6, 100) # 1 min OFF
    off_list.append(off5_coords)
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    tone_list = []
    tone1_coords = plt.Rectangle((17, 0), 3, 100) # 30 s
    tone_list.append(tone1_coords)
    tone2_coords = plt.Rectangle((26, 0), 3, 100) # 30 s
    tone_list.append(tone2_coords)
    tone3_coords = plt.Rectangle((41, 0), 3, 100) # 30 s
    tone_list.append(tone3_coords)
    tone4_coords = plt.Rectangle((53, 0), 3, 100) # 30 s
    tone_list.append(tone4_coords)
    tones_coll = PatchCollection(tone_list, alpha=0.1, color='green')
    ax.add_collection(tones_coll)
    tone_coll_border = PatchCollection(tone_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(tone_coll_border)
    tone_patch = mpatches.Patch(color='green', label='tone+light', alpha=0.1)
    handles.append(tone_patch)
    
    
    plt.legend(handles=handles)
    
    return ax


def timeseries_SOC_rectangles_plot_bin30(df, protocol='soc', ax=None, bin_size=30, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    df_melted['Time bin'] = [str(int(int(timelapse[9:])*(bin_size/10))) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                 # units="mouse", estimator=None  
                 )
    plt.ylim(0,100)
    handles, labels = ax.get_legend_handles_labels()
    
    off_list = []
    off1_coords = plt.Rectangle((-0.5, 0), 6, 100) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((7.5, 0), 2, 100) # 1 min OFF
    off_list.append(off2_coords)
    off3_coords = plt.Rectangle((11.5, 0), 2, 100) # 1 min OFF
    off_list.append(off3_coords)
    off4_coords = plt.Rectangle((15.5, 0), 2, 100) # 1 min OFF
    off_list.append(off4_coords)
    off5_coords = plt.Rectangle((19.5, 0), 2, 100) # 1 min OFF
    off_list.append(off5_coords)
    # Up to this point is for 4 trials
    off6_coords = plt.Rectangle((19.5, 0), 2, 100) # 1 min OFF
    off_list.append(off6_coords)
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    tone_list = []
    tone1_coords = plt.Rectangle((5.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone1_coords)
    tone2_coords = plt.Rectangle((9.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone2_coords)
    tone3_coords = plt.Rectangle((13.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone3_coords)
    tone4_coords = plt.Rectangle((17.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone4_coords)
    # Up to this point is for 4 trials
    tone5_coords = plt.Rectangle((21.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone5_coords)
    tone6_coords = plt.Rectangle((25.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone6_coords)
    tone7_coords = plt.Rectangle((29.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone7_coords)
    tone8_coords = plt.Rectangle((33.5, 0), 1, 100) # 30 s tone
    tone_list.append(tone8_coords)
    # Up to this point is for 8 trials
    tones_coll = PatchCollection(tone_list, alpha=0.1, color='blue')
    ax.add_collection(tones_coll)
    tone_coll_border = PatchCollection(tone_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(tone_coll_border)
    tone_patch = mpatches.Patch(color='blue', label='tone', alpha=0.1)
    handles.append(tone_patch)
    
    light_list = []
    light1_coords = plt.Rectangle((6.5, 0), 1, 100) # 30 s tone
    light_list.append(light1_coords)
    light2_coords = plt.Rectangle((10.5, 0), 1, 100) # 30 s tone
    light_list.append(light2_coords)
    light3_coords = plt.Rectangle((14.5, 0), 1, 100) # 30 s tone
    light_list.append(light3_coords)
    light4_coords = plt.Rectangle((18.5, 0), 1, 100) # 30 s tone
    light_list.append(light4_coords)
    # Up to this point is for 4 trials
    light5_coords = plt.Rectangle((22.5, 0), 1, 100) # 30 s tone
    light_list.append(light5_coords)
    lights_coll = PatchCollection(light_list, alpha=0.1, color='red', edgecolor='black')
    ax.add_collection(lights_coll)
    light_coll_border = PatchCollection(light_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(light_coll_border)
    light_patch = mpatches.Patch(color='red', label='light', alpha=0.1)
    handles.append(light_patch)
    
    plt.legend(handles=handles)
    
    return ax


def timeseries_probetest_rectangles_plot_bin10(df, protocol='light1', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    # The numbers of the X axis will be expressed in seconds
    df_melted['Time bin'] = [str(int(int(timelapse[9:])*(bin_size/1))/60) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                  # units="mouse", estimator=None  
                 )
        
    plt.ylim(0,100)
    ax.set(xlabel='Time [min]')
    handles, labels = ax.get_legend_handles_labels()
    
    # To plot only every 6th tick on the X axis
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[5::6]))
    for label in temp:
        label.set_visible(False)
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 100) # 3 min OFF
    off_list.append(off1_coords)
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    if protocol == 'tone1' or protocol == 'tone2' or protocol == 's2':
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 30, 100) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='blue')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='blue', label='tone', alpha=0.1)
        handles.append(tone_patch)
    
    if protocol == 'light1' or protocol == 'light2' or protocol == 's1':
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 30, 100) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='red')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='red', label='light', alpha=0.1)
        handles.append(tone_patch)
        
    plt.legend(handles=handles)
       
    return ax


def new_timeseries_probetest_rectangles_plot_bin10(df, protocol='s2', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')    
    
    # The numbers of the X axis will be expressed in seconds
    df_melted['Time bin'] = [str(int(int(timelapse.split("_")[-1])*(bin_size/1))/60) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                  # units="mouse", estimator=None  
                 )
        
    plt.ylim(0,100)
    ax.set(xlabel='Time [min]')
    handles, labels = ax.get_legend_handles_labels()
    
    # To plot only every 6th tick on the X axis
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[5::6]))
    for label in temp:
        label.set_visible(False)
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 100) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((23, 0), 6, 100) # 1 min OFF
    off_list.append(off2_coords)    
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    if protocol == 'tone1' or protocol == 'tone2' or protocol == 's2':
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 6, 100) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((29, 0), 6, 100) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='blue')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='blue', label='S2', alpha=0.1)
        handles.append(tone_patch)
    
    else:
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 6, 100) # 1 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((29, 0), 6, 100) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='red')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='red', label='S1', alpha=0.1)
        handles.append(tone_patch)
        
    plt.legend(handles=handles)
       
    return ax


def new_timeseries_speed_probetest_rectangles_plot_bin10(df, protocol='s1', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'speed_' in c],
                              value_name='Mean speed in each bin [mm/s]', var_name='Time bin')
    df_melted = df_melted[df_melted['Time bin'] != 'speed_freezing']
    df_melted = df_melted[df_melted['Time bin'] != 'speed_moving']
    # The numbers of the X axis will be expressed in seconds
    df_melted['Time bin'] = [str(int(int(timelapse.split('_')[-1])*(bin_size/1))/60) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Mean speed in each bin [mm/s]', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                  # units="mouse", estimator=None  
                 )
        
    plt.ylim(0,100)
    ax.set(xlabel='Time [min]')
    handles, labels = ax.get_legend_handles_labels()
    
    # To plot only every 6th tick on the X axis
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[5::6]))
    for label in temp:
        label.set_visible(False)
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 130) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((23, 0), 6, 130) # 1 min OFF
    off_list.append(off2_coords)    
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    if protocol == 'tone1' or protocol == 'tone2' or protocol == 's2':
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 6, 130) # 1 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((29, 0), 6, 130) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='blue')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='blue', label='S2', alpha=0.1)
        handles.append(tone_patch)
    
    else:
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 6, 130) # 1 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((29, 0), 6, 130) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='red')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='red', label='S1', alpha=0.1)
        handles.append(tone_patch)
        
    plt.legend(handles=handles)
       
    return ax


def new_timeseries_distance_probetest_rectangles_plot_bin10(df, protocol='s1', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'distance_' in c],
                              value_name='Total distance moved in each bin [mm]', var_name='Time bin')
    # The numbers of the X axis will be expressed in seconds
    df_melted['Time bin'] = [str(int(int(timelapse.split("_")[-1])*(bin_size/1))/60) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Total distance moved in each bin [mm]', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                  # units="mouse", estimator=None  
                 )
        
    plt.ylim(0,1000)
    ax.set(xlabel='Time [min]')
    handles, labels = ax.get_legend_handles_labels()
    
    # To plot only every 6th tick on the X axis
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[5::6]))
    for label in temp:
        label.set_visible(False)
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 1000) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((23, 0), 6, 1000) # 1 min OFF
    off_list.append(off2_coords)    
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    if protocol == 'tone1' or protocol == 'tone2' or protocol == 's2':
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 6, 1000) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((29, 0), 6, 1000) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='blue')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='blue', label='tone', alpha=0.1)
        handles.append(tone_patch)
    
    else:
        probetest_list = []
        probetest_coords = plt.Rectangle((17, 0), 6, 1000) # 1 min probetest
        probetest_list.append(probetest_coords)
        probetest_coords_2 = plt.Rectangle((29, 0), 6, 1000) # 1 min probetest
        probetest_list.append(probetest_coords_2)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='red')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='red', label='light', alpha=0.1)
        handles.append(tone_patch)
        
    plt.legend(handles=handles)
       
    return ax


def timeseries_probetest_rectangles_plot_bin30(df, protocol='tone1', ax=None, bin_size=30, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    df_melted['Time bin'] = [str(int(int(timelapse[9:])*(bin_size/10))) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                  # units="mouse", estimator=None  
                 )
    plt.ylim(0,100)
    handles, labels = ax.get_legend_handles_labels()
    
    off_list = []
    off1_coords = plt.Rectangle((-0.5, 0), 6, 100) # 3 min OFF
    off_list.append(off1_coords)
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    if protocol == 'tone1':
        probetest_list = []
        probetest_coords = plt.Rectangle((5.5, 0), 10, 100) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='blue')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='blue', label='tone', alpha=0.1)
        handles.append(tone_patch)
    
    if protocol == 'light1':
        probetest_list = []
        probetest_coords = plt.Rectangle((17.5, 0), 30, 100) # 5 min probetest
        probetest_list.append(probetest_coords)
        probetest_coll = PatchCollection(probetest_list, alpha=0.1, color='red')
        ax.add_collection(probetest_coll)
        probetest_coll_border = PatchCollection(probetest_list, facecolor='none', edgecolor='black', alpha=0.5)
        ax.add_collection(probetest_coll_border)
        tone_patch = mpatches.Patch(color='red', label='light', alpha=0.1)
        handles.append(tone_patch)
        
    plt.legend(handles=handles)
    
    return ax


def timeseries_FOC_rectangles_plot_bin10(df, protocol='foc2', ax=None, bin_size=10, hue='group'):
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    # The numbers of the X axis will be expressed in seconds
    df_melted['Time bin'] = [str(int(int(timelapse[9:])*(bin_size/1))/60) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                  # units="mouse", estimator=None  
                 )
        
    plt.ylim(0,100)
    ax.set(xlabel='Time [min]')
    handles, labels = ax.get_legend_handles_labels()
    
    # To plot only every 6th tick on the X axis
    temp = ax.xaxis.get_ticklabels()
    temp = list(set(temp) - set(temp[5::6]))
    for label in temp:
        label.set_visible(False)
    
    off_list = []
    off1_coords = plt.Rectangle((0, 0), 17, 100) # 3 min OFF
    off_list.append(off1_coords)
    off2_coords = plt.Rectangle((18, 0), 6, 100) # 1 min OFF
    off_list.append(off2_coords)
    off3_coords = plt.Rectangle((25, 0), 6, 100) # 1 min OFF
    off_list.append(off3_coords)
    off4_coords = plt.Rectangle((32, 0), 6, 100) # 1 min OFF
    off_list.append(off4_coords)
    off5_coords = plt.Rectangle((39, 0), 6, 100) # 1 min OFF
    off_list.append(off5_coords)
    off6_coords = plt.Rectangle((46, 0), 6, 100) # 1 min OFF
    off_list.append(off6_coords)    
    off_coll = PatchCollection(off_list, alpha=0.1, color='yellow')
    ax.add_collection(off_coll)
    off_coll_border = PatchCollection(off_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(off_coll_border)
    off_patch = mpatches.Patch(color='yellow', label='off period', alpha=0.1)
    handles.append(off_patch)
    
    light_list = []
    light1_coords = plt.Rectangle((17, 0), 1, 100) # 10 s light
    light_list.append(light1_coords)
    light2_coords = plt.Rectangle((24, 0), 1, 100) # 10 s light
    light_list.append(light2_coords)
    light3_coords = plt.Rectangle((31, 0), 1, 100) # 10 s light
    light_list.append(light3_coords)
    light4_coords = plt.Rectangle((38, 0), 1, 100) # 10 s light
    light_list.append(light4_coords)
    light5_coords = plt.Rectangle((45, 0), 1, 100) # 10 s light
    light_list.append(light5_coords) 
    lights_coll = PatchCollection(light_list, alpha=0.1, color='red', edgecolor='black')
    ax.add_collection(lights_coll)
    light_coll_border = PatchCollection(light_list, facecolor='none', edgecolor='black', alpha=0.5)
    ax.add_collection(light_coll_border)
    light_patch = mpatches.Patch(color='red', label='S1', alpha=0.1)
    handles.append(light_patch)
    
    plt.legend(handles=handles)
    
    return ax


def timeseries_plot(df, protocol='hab', ax=None, bin_size=10, hue='group'):

    
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    df_proto = df[df['protocol'] == protocol]
    df_melted = df_proto.melt(id_vars=['mouse', hue, 'protocol'],
                              value_vars=[c for c in df.columns if 'freezing_' in c],
                              value_name='Freezing percentage', var_name='Time bin')
    df_melted['Time bin'] = [str(int(int(timelapse[9:])*(bin_size/10))) for timelapse in df_melted['Time bin']]
    
    sns.set_theme(style="whitegrid")
    sns.lineplot(x='Time bin', y='Freezing percentage', hue=hue, data=df_melted, ax=ax,
                 # To plot individual values:
                 # units="mouse", estimator=None  
                 )
    plt.ylim(0,100)
    handles, labels = ax.get_legend_handles_labels()
    
    return ax


# =============================================================================


def freezing_duration_hue_comparison(df, ax=None, perc=True, thres=1, hue='group', tag='first_on', protocol='tone1', subject='mouse'):
    
    cols_to_plot = []
    for i in range(20): # Number of bins = 20
        interval_start = i * 0.25
        interval_end = interval_start + 0.25
        cols_to_plot.append(tag + ' ' + str(interval_start) + '-' + str(interval_end))
    
    list_columns = [hue, subject]
    list_columns.extend(cols_to_plot)
    new_df = df[list_columns][df.protocol == protocol]
    melted_df = pd.melt(new_df, id_vars=[hue, subject], var_name='Attribute', value_name='Value')     
    
    melted_df[['Delete', 'Interval']] = melted_df['Attribute'].str.split('-', expand=True)

    melted_df.drop('Attribute', axis=1, inplace=True)
    melted_df.drop('Delete', axis=1, inplace=True)
    melted_df['Interval'] = melted_df['Interval'].astype(float)
    
    if thres is not None:
        melted_df = melted_df[melted_df.Interval != thres]
    
    if perc is True:
        melted_df['sum_group'] = melted_df.groupby(subject)['Value'].transform('sum')
        melted_df['perc'] = melted_df.Value / melted_df.sum_group * 100
    
    if ax is None:
        fig, ax = plt.subplots()
       
    if perc is False:
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=melted_df, x='Interval', y='Value', hue=hue)
    if perc is True:
        # melted_df['Interval'] = melted_df['Interval'].astype(str)
        # melted_df = melted_df[melted_df[hue] == specific_hue]
        # pie_df = melted_df.groupby('Interval')['perc'].mean()
        # pie_df = pie_df.to_frame().reset_index().rename(columns={'perc': 'mean_value'})
        # pie_df = pie_df[pie_df['mean_value'] != 0]

        # ax.pie(pie_df.mean_value, labels=['']*len(pie_df.Interval), autopct='')
        # legend_data = ['Interval {}: {:.2f}%'.format(row['Interval'], row['mean_value']) for _, row in pie_df.iterrows()]
        # ax.legend(legend_data, loc='upper right', bbox_to_anchor=(1, 1))
        
        sns.set_theme(style="whitegrid")
        sns.lineplot(data=melted_df, x='Interval', y='perc', hue=hue)

    if perc is False:
        plt.ylabel('Number of freezing events')
        plt.xlabel('Duration of the freezing events [s]')
    if perc is True:
        plt.ylabel('Percentatge of freezing events [%]')
        plt.xlabel('Duration of the freezing events [s]')
        plt.ylim(0,70)
    
    return ax


def freezing_duration_comparison_between_two_periods(df, ax=None, perc=False, protocol='light1', specific_hue='G1'):
    
    off = 'last_off'
    on = 'first_on' # Also used for the habituation
    
    thres=1
    hue='group'
    subject='mouse'
    
    length_of_each_bin = 0.1 # Change this number to modify the number of bins
    total_length = 7
    number_of_bins = int(total_length / length_of_each_bin)
    
    # ==========================================================
    
    cols_to_plot_off = []
    cols_to_plot_on = []
    for i in range(number_of_bins): 
        interval_start = i * length_of_each_bin
        interval_end = interval_start + length_of_each_bin
        cols_to_plot_off.append(off + ' ' + str(interval_start) + '-' + str(interval_end))
        cols_to_plot_on.append(on + ' ' + str(interval_start) + '-' + str(interval_end))
    
    list_columns = [hue, subject]
    list_columns.extend(cols_to_plot_off)
    list_columns.extend(cols_to_plot_on)
      
    new_df = df[list_columns][df[hue] == specific_hue][df.protocol == protocol]
    
    # To add the habituation period (same period as the ON period)
    new_df2 = df[list_columns][df[hue] == specific_hue][df.protocol == 'hab']
    new_list_columns = [subject]
    new_list_columns.extend(cols_to_plot_on)   
    new_df2 = new_df2[new_list_columns]
    new_df2.columns = new_df2.columns.str.replace("on", "hab")
    melted_df = pd.merge(new_df, new_df2, on=subject, how='outer')
    
    melted_df = pd.melt(melted_df, id_vars=[hue, subject], var_name='Attribute', value_name='Value')
    
    melted_df[['Delete', 'Interval']] = melted_df['Attribute'].str.split('-', expand=True)    
    melted_df.drop('Delete', axis=1, inplace=True)
    
    melted_df[['Period', 'Delete']] = melted_df['Attribute'].str.split(' ', expand=True)    
    melted_df.drop('Delete', axis=1, inplace=True)
    melted_df.Period = melted_df.Period.replace(on, 'On period')
    melted_df.Period = melted_df.Period.replace(off, 'Off period')
    melted_df.Period = melted_df.Period.replace('first_hab', 'Habituation')
    
    melted_df.drop('Attribute', axis=1, inplace=True)

    melted_df['Interval'] = melted_df['Interval'].astype(float)
    
    # Normalization: how many periods of x seconds fit into a timelapse of 1 min?
    # melted_df['normalization'] = 60 / melted_df.Interval
    # melted_df['norm_value'] = melted_df.Value / melted_df.normalization

    
    if thres is not None:
        melted_df = melted_df[melted_df.Interval != thres]
    
    # -------------------
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    #  To design a palette:
    if protocol[:-1] == 'tone':
        palette = ['grey', 'cornflowerblue', 'dimgrey']
    if protocol[:-1] == 'light':
        palette = ['grey', 'indianred', 'dimgrey']
    if protocol[:-1] != 'tone' and protocol[:-1] != 'light':
        palette = ['grey', 'dimgrey', 'dimgrey']
    
    # To plot the Habituation with a diffrent line style
    melted_df['is_hab'] = melted_df['Period'] == 'Habituation'
    
    if perc is False:
        sns.lineplot(data=melted_df, x='Interval', y='Value', hue='Period', palette=palette, style='is_hab', legend=False)     
        plt.ylabel('Number of freezing events')
        
    if perc is True:     
        melted_df['sum_group'] = melted_df.groupby(subject)['Value'].transform('sum')
        melted_df['perc'] = melted_df.Value / melted_df.sum_group * 100
        sns.lineplot(data=melted_df, x='Interval', y='perc', hue='Period', palette=palette, style='is_hab', legend=False)
        plt.ylabel('Percentatge of freezing events [%]')
        
    plt.xlabel('Duration of the freezing events [s]')
    plt.ylim(0,6)
    
    # ------------------- FOR THE LEGEND:
    
    # define the line colors, styles, and tags
    if protocol[:-1] == 'tone':
        colors = ['cornflowerblue', 'grey', 'dimgrey']
    if protocol[:-1] == 'light':
        colors = ['indianred', 'grey', 'dimgrey']
    if protocol[:-1] != 'tone' and protocol[:-1] != 'light':
        colors = ['dimgrey', 'grey', 'dimgrey']

    styles = ['-', '-', '--']
    tags = ['On period', 'Off period', 'Habituation']
    # plot empty lines with colors and styles
    for i in range(len(colors)):
        ax.plot([], [], color=colors[i], linestyle=styles[i], label=tags[i])
    # create the legend and set its properties
    legend = ax.legend(loc='upper right', title='Period', 
                       bbox_to_anchor=(1.0, 1.0), frameon=True, borderpad=1, borderaxespad=0.5, 
                       handlelength=1.5, handletextpad=0.5, fancybox=True)

    # set the title font size
    legend.get_title().set_fontsize('12')

    # set the border properties
    legend.get_frame().set_edgecolor('lightgrey') # set more dimmer border color
    legend.get_frame().set_facecolor((1.0, 1.0, 1.0, 0.5)) # set slightly transparent background
  
    # ------------------- FOR THE AREA UNDER THE CURVE:

    area_on = simps(melted_df.Value[melted_df.Period == 'On period'], melted_df.Interval[melted_df.Period == 'On period'])
    area_off = simps(melted_df.Value[melted_df.Period == 'Off period'], melted_df.Interval[melted_df.Period == 'Off period'])
    area_hab = simps(melted_df.Value[melted_df.Period == 'Habituation'], melted_df.Interval[melted_df.Period == 'Habituation'])
    
    ax.annotate(f"On period: Area = {area_on:.2f}", xy=(0.5, 0.9), xycoords="axes fraction")
    ax.annotate(f"Off period: Area = {area_off:.2f}", xy=(0.5, 0.8), xycoords="axes fraction")
    ax.annotate(f"Habituation: Area = {area_hab:.2f}", xy=(0.5, 0.7), xycoords="axes fraction")
    
    return ax


# =============================================================================
# SWARM PLOTS
# =============================================================================


def generate_swarm_plot(df, protocol='s1', hue='group', specific_hue='tone-shock'):

    # Starting to reshape the dataframe
    list_columns = ['mouse', hue, 'lengths_last_off', 'lengths_first_on']
    new_df = df[list_columns][df[hue] == specific_hue][df.protocol == protocol]
    new_df = new_df.rename(columns={'lengths_last_off': 'Off period'})
    new_df = new_df.rename(columns={'lengths_first_on': 'On period'})

    # To add the habituation period (same period as the ON period)
    hab_list_columns = ['mouse', 'lengths_first_on']
    hab_new_df = df[hab_list_columns][df[hue] == specific_hue][df.protocol == 'hab']
    hab_new_df = hab_new_df.rename(columns={'lengths_first_on': 'Habituation'})
    
    # Reshape the DataFrame into a long format
    melted_df = pd.merge(new_df, hab_new_df, on='mouse', how='outer')
    melted_df = pd.melt(melted_df, id_vars=[hue, 'mouse'], var_name='Group', value_name='Freezing Length')
    melted_df['Freezing Length'] = melted_df['Freezing Length'].fillna('')
    
    new_melted_df = melted_df.copy(deep = True)
    new_melted_df['Freezing Length'] = new_melted_df['Freezing Length'].apply(lambda x: [float(i.strip('[]').replace(',', '.')) if i.strip('[]') else None for i in x.split(',')])
    
    # Explode the list column into separate rows
    new_melted_df.drop('mouse', axis=1, inplace=True)
    new_melted_df = new_melted_df.explode('Freezing Length').reset_index(drop=True)
    new_melted_df['Freezing Length'] = new_melted_df['Freezing Length'].astype(float)
    
    # -------------------------
    
    # Add a text annotation with the number of dots
    hab_num = new_melted_df['Freezing Length'][new_melted_df.Group == 'Habituation'].to_list()
    hab_num = len([x for x in hab_num if not math.isnan(x)])
    hab_num = 'Habituation\nn = ' + str(hab_num)
    new_melted_df['Group'] = new_melted_df['Group'].replace('Habituation', hab_num)

    off_num = new_melted_df['Freezing Length'][new_melted_df.Group == 'Off period'].to_list()
    off_num = len([x for x in off_num if not math.isnan(x)])
    off_num = 'Off period\nn = ' + str(off_num)
    new_melted_df['Group'] = new_melted_df['Group'].replace('Off period', off_num)
    
    on_num = new_melted_df['Freezing Length'][new_melted_df.Group == 'On period'].to_list()
    on_num = len([x for x in on_num if not math.isnan(x)])
    on_num = 'On period\nn = ' + str(on_num)
    new_melted_df['Group'] = new_melted_df['Group'].replace('On period', on_num)
    
    #  To design a palette:
    if protocol[:-1] == 'tone' or protocol == 'tone' or protocol == 's2':
        palette = ['grey', 'dimgrey', 'cornflowerblue']
    if protocol[:-1] == 'light' or protocol == 'light' or protocol == 's1':
        palette = ['grey', 'dimgrey', 'indianred']
    # else:
    #     palette = ['grey', 'dimgrey', 'dimgrey']

    # Generate the swarm plot
    sns.set_theme(style="whitegrid")
    order = [hab_num, off_num, on_num]
    sns.catplot(data=new_melted_df, x="Group", y="Freezing Length", kind="swarm", order=order, palette=palette)
        
    # Set the axis labels
    plt.xlabel('Number of freezing events')
    plt.ylabel('Length of the freezing events [s]')
    plt.ylim(0,25)
    
    plt.show()


# =============================================================================
# BOX PLOTS
# =============================================================================


def boxplot_general_view_by_hue(df, ax=None, hue='group'):
    if ax is None:
        fig, ax = plt.subplots()
    sorting_order = ['hab', 'foc1', 'foc2', 'foc3', 'foc4', 'soc1', 'soc2', 'tone1', 'light1', 'tone2', 'light2']
    sns.set_theme(style="whitegrid")
    sns.boxplot(x='protocol', y='perc_freezing', data=df, hue=hue, ax=ax, order=sorting_order,
                whis=[0, 100], width=.6, palette='Pastel1')
    sns.stripplot(x='protocol', y='perc_freezing', data=df, hue=hue, ax=ax, order=sorting_order,
                  size=4, color=".3", dodge=True, palette='Set1')
    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


# =============================================================================


def boxplot_OffOn_paired_data(df, ax=None, subject='mouse'):
    
    # Creating a first type of dataframe with three columns
    tone_off = df[['meas_slices1', subject]][df.protocol == 'tone']
    tone_off.rename(columns = {'meas_slices1': 'freezing_on_off'}, inplace = True)
    tone_off['time_point'] = 'tone off'
        
    tone_on = df[['meas_slices2', subject]][df.protocol == 'tone']
    tone_on.rename(columns = {'meas_slices2': 'freezing_on_off'}, inplace = True)
    tone_on['time_point'] = 'tone on'
        
    light_off = df[['meas_slices1', subject]][df.protocol == 'light']
    light_off.rename(columns = {'meas_slices1': 'freezing_on_off'}, inplace = True)
    light_off['time_point'] = 'light off'
        
    light_on = df[['meas_slices2', subject]][df.protocol == 'light']
    light_on.rename(columns = {'meas_slices2': 'freezing_on_off'}, inplace = True)
    light_on['time_point'] = 'light on'
    
    dataframe_type_1 = pd.concat([tone_off, tone_on, light_off, light_on])
    
    # Creating a second type of dataframe with four columns
    tone_off = df[['meas_slices1', subject]][df.protocol == 'tone']
    tone_off.rename(columns = {'meas_slices1': 'tone off'}, inplace = True)
    
    tone_on = df[['meas_slices2', subject]][df.protocol == 'tone']
    tone_on.rename(columns = {'meas_slices2': 'tone on'}, inplace = True)
    
    light_off = df[['meas_slices1', subject]][df.protocol == 'light']
    light_off.rename(columns = {'meas_slices1': 'light off'}, inplace = True)
    
    light_on = df[['meas_slices2', subject]][df.protocol == 'light']
    light_on.rename(columns = {'meas_slices2': 'light on'}, inplace = True)
    
    dataframe_type_2 = tone_off
    dataframe_type_2 = dataframe_type_2.merge(tone_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[['tone off', 'tone on']], positions=[0,1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2[['light off', 'light on']], positions=[2,3], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,['tone off','tone on']], dataframe_type_2.loc[idx,['tone off','tone on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['light off','light on']], dataframe_type_2.loc[idx,['light off','light on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def boxplot_OffOn_paired_data_by_hue(df, ax=None, s1='light1', s2='tone1', subject='mouse', hue='group', specific_hue='G1'):
    
    # Creating a first type of dataframe with three columns:
    s2_off = df[['meas_slices1', subject, hue]][df.protocol == s2]
    s2_off.rename(columns = {'meas_slices1': 'freezing_on_off'}, inplace = True)
    s2_off['time_point'] = s2 + ' off'
    
    s2_on = df[['meas_slices2', subject, hue]][df.protocol == s2]
    s2_on.rename(columns = {'meas_slices2': 'freezing_on_off'}, inplace = True)
    s2_on['time_point'] = s2 + ' on'
    
    s1_off = df[['meas_slices1', subject, hue]][df.protocol == s1]
    s1_off.rename(columns = {'meas_slices1': 'freezing_on_off'}, inplace = True)
    s1_off['time_point'] = s1 + ' off'
    
    s1_on = df[['meas_slices2', subject, hue]][df.protocol == s1]
    s1_on.rename(columns = {'meas_slices2': 'freezing_on_off'}, inplace = True)
    s1_on['time_point'] = s1 + ' on'

    # This type of dataframe is ideal for the statistic analysis,
    # because it considers all levels of the between factor at the same time
    dataframe_type_1 = pd.concat([s2_off, s2_on, s1_off, s1_on])
    
    # stats_df = create_dataframe_type_1(df)
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_slices1', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_slices1': s2 + ' off'}, inplace = True)
    
    s2_on = df[['meas_slices2', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_slices2': s2 + ' on'}, inplace = True)
    
    s1_off = df[['meas_slices1', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_slices1': s1 + ' off'}, inplace = True)
    
    s1_on = df[['meas_slices2', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_slices2': s1 + ' on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[[s2 + ' off', s2 + ' on']], positions=[0,1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2[[s1 + ' off', s1 + ' on']], positions=[2,3], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2 + ' off',s2 + ' on']], dataframe_type_2.loc[idx,[s2 + ' off',s2 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,[s1 + ' off',s1 + ' on']], dataframe_type_2.loc[idx,[s1 + ' off',s1 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,[s2 + ' on',s1 + ' off']], dataframe_type_2.loc[idx,[s2 + ' on',s1 +' off']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def boxplot_OffOn_normalizedOnbyOff_by_hue(df, ax=None, s1='light2', s2='tone2', subject='mouse', hue='group', specific_hue='G2'):
      
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_slices1', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_slices1': s2 + '_off'}, inplace = True)
    
    s2_on = df[['meas_slices2', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_slices2': s2 + '_on'}, inplace = True)
    
    s1_off = df[['meas_slices1', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_slices1': s1 + '_off'}, inplace = True)
    
    s1_on = df[['meas_slices2', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_slices2': s1 + '_on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    dataframe_type_2[s2] = dataframe_type_2[s2 + '_on'] / dataframe_type_2[s2 + '_off']
    dataframe_type_2[s1] = dataframe_type_2[s1 + '_on'] / dataframe_type_2[s1 + '_off']
    dataframe_type_2 = dataframe_type_2.drop([s2 + '_off', s1 + '_off', s2 + '_on', s1 + '_on'], axis=1)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[s2], positions=[0], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2[s1], positions=[1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2,s1]], dataframe_type_2.loc[idx,[s2,s1]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

    plt.ylim(0,5)
    plt.ylabel('Ratio')
    plt.xlabel('')
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    return ax


def boxplot_OffOn_1_min_paired_data_by_hue(df, ax=None, s1='light1', s2='tone1', subject='mouse', hue='group', specific_hue='G2'):
         
    # Creating a first type of dataframe with three columns
    s2_off = df[['meas_off_1min', subject, hue]][df.protocol == s2]
    s2_off.rename(columns = {'meas_off_1min': 'freezing_on_off'}, inplace = True)
    s2_off['time_point'] = s2 + ' off'
        
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on.rename(columns = {'meas_on_1min': 'freezing_on_off'}, inplace = True)
    s2_on['time_point'] = s2 + ' on'
        
    s1_off = df[['meas_off_1min', subject, hue]][df.protocol == s1]
    s1_off.rename(columns = {'meas_off_1min': 'freezing_on_off'}, inplace = True)
    s1_off['time_point'] = s1 + ' off'
        
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on.rename(columns = {'meas_on_1min': 'freezing_on_off'}, inplace = True)
    s1_on['time_point'] = s1 + ' on'
    
    # This type of dataframe is ideal for the statistic analysis,
    # because it considers all levels of the between factor at the same time
    dataframe_type_1 = pd.concat([s2_off, s2_on, s1_off, s1_on])
    
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_off_1min', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_off_1min': s2 + ' off'}, inplace = True)
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_on_1min': s2 + ' on'}, inplace = True)
    
    s1_off = df[['meas_off_1min', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_off_1min': s1 + ' off'}, inplace = True)
    
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_on_1min': s1 + ' on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")

    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[[s2 + ' off', s2 + ' on']], positions=[0,1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2[[s1 + ' off', s1 + ' on']], positions=[2,3], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2 + ' off',s2 + ' on']], dataframe_type_2.loc[idx,[s2 + ' off',s2 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,[s1 + ' off',s1 + ' on']], dataframe_type_2.loc[idx,[s1 + ' off',s1 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,[s2 + ' on',s1 + ' off']], dataframe_type_2.loc[idx,[s2 + ' on',s1 + ' off']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def boxplot_OffOn_1_min_normalizedOnbyOff_by_hue(df, ax=None, s1='light2', s2='tone2', subject='mouse', hue='group', specific_hue='G2'):
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_off_1min', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_off_1min': s2 + '_off'}, inplace = True)
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_on_1min': s2 + '_on'}, inplace = True)
    
    s1_off = df[['meas_off_1min', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_off_1min': s1 + '_off'}, inplace = True)
    
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_on_1min': s1 + '_on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    dataframe_type_2[s2] = dataframe_type_2[s2 + '_on'] / dataframe_type_2[s2 + '_off']
    dataframe_type_2[s1] = dataframe_type_2[s1 + '_on'] / dataframe_type_2[s1 + '_off']
    dataframe_type_2 = dataframe_type_2.drop([s2 + '_off', s1 + '_off', s2 + '_on', s1 + '_on'], axis=1)

    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[s2], positions=[0], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2[s1], positions=[1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2,s1]], dataframe_type_2.loc[idx,[s2,s1]], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

    plt.ylim(0,7)
    plt.ylabel('Ratio')
    plt.xlabel('')
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    return ax


def boxplot_OffOn_2_min_paired_data_by_hue(df, ax=None, subject='mouse', hue='group', specific_hue='1x8'):
    
    # Creating a first type of dataframe with three columns
    tone_off = df[['meas_off_2min', subject, hue]][df.protocol == 'tone']
    tone_off.rename(columns = {'meas_off_2min': 'freezing_on_off'}, inplace = True)
    tone_off['time_point'] = 'tone off'
        
    tone_on = df[['meas_on_2min', subject, hue]][df.protocol == 'tone']
    tone_on.rename(columns = {'meas_on_2min': 'freezing_on_off'}, inplace = True)
    tone_on['time_point'] = 'tone on'
        
    light_off = df[['meas_off_2min', subject, hue]][df.protocol == 'light']
    light_off.rename(columns = {'meas_off_2min': 'freezing_on_off'}, inplace = True)
    light_off['time_point'] = 'light off'
        
    light_on = df[['meas_on_2min', subject, hue]][df.protocol == 'light']
    light_on.rename(columns = {'meas_on_2min': 'freezing_on_off'}, inplace = True)
    light_on['time_point'] = 'light on'
    
    # This type of dataframe is ideal for the statistic analysis,
    # because it considers all levels of the between factor at the same time
    dataframe_type_1 = pd.concat([tone_off, tone_on, light_off, light_on])
    
    
    # Creating a second type of dataframe with four columns
    tone_off = df[['meas_off_2min', subject, hue]][df.protocol == 'tone']
    tone_off = tone_off[tone_off[hue] == specific_hue].drop(hue, axis=1)
    tone_off.rename(columns = {'meas_off_2min': 'tone off'}, inplace = True)
    
    tone_on = df[['meas_on_2min', subject, hue]][df.protocol == 'tone']
    tone_on = tone_on[tone_on[hue] == specific_hue].drop(hue, axis=1)
    tone_on.rename(columns = {'meas_on_2min': 'tone on'}, inplace = True)
    
    light_off = df[['meas_off_2min', subject, hue]][df.protocol == 'light']
    light_off = light_off[light_off[hue] == specific_hue].drop(hue, axis=1)
    light_off.rename(columns = {'meas_off_2min': 'light off'}, inplace = True)
    
    light_on = df[['meas_on_2min', subject, hue]][df.protocol == 'light']
    light_on = light_on[light_on[hue] == specific_hue].drop(hue, axis=1)
    light_on.rename(columns = {'meas_on_2min': 'light on'}, inplace = True)
    
    dataframe_type_2 = tone_off
    dataframe_type_2 = dataframe_type_2.merge(tone_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    # For the boxplots with seaborn
    # sns.boxplot(data=mydataframe, showfliers=False, width=.3, 
    #             boxprops=dict(alpha=.3),        # Transparency of the boxplot
    #             medianprops=dict(alpha=.7),     # Transparency of the median bar
    #             whiskerprops=dict(alpha=.3),    # Transparency of the whiskers
    #             minimumprops=dict(alpha=.7),    # Transparency of the minimum
    #             notch=True,                     # Makes a shape like an sand clock
    #             ax=ax)
    
    # For the violinplots with seaborn
    # my_pal = {"tone off": "blue", "tone on": "blue", "light off": "red", "light on": "red"}
    # sns.violinplot(x = dataframe_type_1.time_point, y = dataframe_type_1.freezing_on_off,
    #                showfliers=False,
    #                linewidth = 0,
    #                 palette = my_pal,
    #                 alpha=.5,
    #                ax=ax)
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[['tone off', 'tone on']], positions=[0,1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2[['light off', 'light on']], positions=[2,3], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,['tone off','tone on']], dataframe_type_2.loc[idx,['tone off','tone on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['light off','light on']], dataframe_type_2.loc[idx,['light off','light on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['tone on','light off']], dataframe_type_2.loc[idx,['tone on','light off']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def boxplot_OffOn_2_min_normalizedOnbyOff_by_hue(df, ax=None, subject='mouse', hue='group', specific_hue='2x4'):
      
    # Creating a second type of dataframe with four columns
    tone_off = df[['meas_off_2min', subject, hue]][df.protocol == 'tone']
    tone_off = tone_off[tone_off[hue] == specific_hue].drop(hue, axis=1)
    tone_off.rename(columns = {'meas_off_2min': 'tone_off'}, inplace = True)
    
    tone_on = df[['meas_on_2min', subject, hue]][df.protocol == 'tone']
    tone_on = tone_on[tone_on[hue] == specific_hue].drop(hue, axis=1)
    tone_on.rename(columns = {'meas_on_2min': 'tone_on'}, inplace = True)
    
    light_off = df[['meas_off_2min', subject, hue]][df.protocol == 'light']
    light_off = light_off[light_off[hue] == specific_hue].drop(hue, axis=1)
    light_off.rename(columns = {'meas_off_2min': 'light_off'}, inplace = True)
    
    light_on = df[['meas_on_2min', subject, hue]][df.protocol == 'light']
    light_on = light_on[light_on[hue] == specific_hue].drop(hue, axis=1)
    light_on.rename(columns = {'meas_on_2min': 'light_on'}, inplace = True)
    
    dataframe_type_2 = tone_off
    dataframe_type_2 = dataframe_type_2.merge(tone_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light_on, how='left', on=subject).drop(subject, axis=1)
    
    dataframe_type_2['tone'] = dataframe_type_2.tone_on / dataframe_type_2.tone_off
    dataframe_type_2['light'] = dataframe_type_2.light_on / dataframe_type_2.light_off
    dataframe_type_2 = dataframe_type_2.drop(['tone_off', 'light_off', 'tone_on', 'light_on'], axis=1)
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    # For the boxplots with seaborn
    # sns.boxplot(data=mydataframe, showfliers=False, width=.3, 
    #             boxprops=dict(alpha=.3),        # Transparency of the boxplot
    #             medianprops=dict(alpha=.7),     # Transparency of the median bar
    #             whiskerprops=dict(alpha=.3),    # Transparency of the whiskers
    #             minimumprops=dict(alpha=.7),    # Transparency of the minimum
    #             notch=True,                     # Makes a shape like an sand clock
    #             ax=ax)
    
    # For the violinplots with seaborn
    # my_pal = {"tone off": "blue", "tone on": "blue", "light off": "red", "light on": "red"}
    # sns.violinplot(x = dataframe_type_1.time_point, y = dataframe_type_1.freezing_on_off,
    #                showfliers=False,
    #                linewidth = 0,
    #                 palette = my_pal,
    #                 alpha=.5,
    #                ax=ax)
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2['tone'], positions=[0], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    for pc in ax.violinplot(dataframe_type_2['light'], positions=[1], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,['tone','light']], dataframe_type_2.loc[idx,['tone','light']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)

    plt.ylim(0,7)
    plt.ylabel('Ratio')
    plt.xlabel('')
    plt.axhline(y = 1, color = 'black', linestyle = '--')
    return ax


# =============================================================================


def boxplot_SOCphase_tone_paired_data_by_hue(df, ax=None, protocol='soc1', subject='mouse', hue='group', specific_hue='G1'):
    
    # Creating a first type of dataframe with three columns
    tone1 = df[['meas_tone1', subject, hue]][df.protocol == protocol]
    tone1.rename(columns = {'meas_tone1': 'freezing_value'}, inplace = True)
    tone1['time_point'] = 'tone1'
        
    tone2 = df[['meas_tone2', subject, hue]][df.protocol == protocol]
    tone2.rename(columns = {'meas_tone2': 'freezing_value'}, inplace = True)
    tone2['time_point'] = 'tone2'
        
    tone3 = df[['meas_tone3', subject, hue]][df.protocol == protocol]
    tone3.rename(columns = {'meas_tone3': 'freezing_value'}, inplace = True)
    tone3['time_point'] = 'tone3'
        
    tone4 = df[['meas_tone4', subject, hue]][df.protocol == protocol]
    tone4.rename(columns = {'meas_tone4': 'freezing_value'}, inplace = True)
    tone4['time_point'] = 'tone4'
    
    # This type of dataframe is ideal for the statistic analysis,
    # because it considers all levels of the between factor at the same time
    dataframe_type_1 = pd.concat([tone1, tone2, tone3, tone4])
    
    
    # Creating a second type of dataframe with four columns
    tone1 = df[['meas_tone1', subject, hue]][df.protocol == protocol]
    tone1 = tone1[tone1[hue] == specific_hue].drop(hue, axis=1)
    tone1.rename(columns = {'meas_tone1': 'tone1'}, inplace = True)
    
    tone2 = df[['meas_tone2', subject, hue]][df.protocol == protocol]
    tone2 = tone2[tone2[hue] == specific_hue].drop(hue, axis=1)
    tone2.rename(columns = {'meas_tone2': 'tone2'}, inplace = True)
    
    tone3 = df[['meas_tone3', subject, hue]][df.protocol == protocol]
    tone3 = tone3[tone3[hue] == specific_hue].drop(hue, axis=1)
    tone3.rename(columns = {'meas_tone3': 'tone3'}, inplace = True)
    
    tone4 = df[['meas_tone4', subject, hue]][df.protocol == protocol]
    tone4 = tone4[tone4[hue] == specific_hue].drop(hue, axis=1)
    tone4.rename(columns = {'meas_tone4': 'tone4'}, inplace = True)
    
    dataframe_type_2 = tone1
    dataframe_type_2 = dataframe_type_2.merge(tone2, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(tone3, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(tone4, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    # For the boxplots with seaborn
    # sns.boxplot(data=mydataframe, showfliers=False, width=.3, 
    #             boxprops=dict(alpha=.3),        # Transparency of the boxplot
    #             medianprops=dict(alpha=.7),     # Transparency of the median bar
    #             whiskerprops=dict(alpha=.3),    # Transparency of the whiskers
    #             minimumprops=dict(alpha=.7),    # Transparency of the minimum
    #             notch=True,                     # Makes a shape like an sand clock
    #             ax=ax)
    
    # For the violinplots with seaborn
    # my_pal = {"tone off": "blue", "tone on": "blue", "light off": "red", "light on": "red"}
    # sns.violinplot(x = dataframe_type_1.time_point, y = dataframe_type_1.freezing_on_off,
    #                showfliers=False,
    #                linewidth = 0,
    #                 palette = my_pal,
    #                 alpha=.5,
    #                ax=ax)
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[['tone1', 'tone2', 'tone3', 'tone4']], positions=[0,1,2,3], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('blue')
        pc.set_edgecolor('blue')
        
    # for pc in ax.violinplot(dataframe_type_2[['light off', 'light on']], positions=[2,3], showmeans=False, showmedians=True)['bodies']:
    #     pc.set_facecolor('red')
    #     pc.set_edgecolor('red')

    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,['tone1','tone2']], dataframe_type_2.loc[idx,['tone1','tone2']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['tone2','tone3']], dataframe_type_2.loc[idx,['tone2','tone3']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['tone3','tone4']], dataframe_type_2.loc[idx,['tone3','tone4']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        # ax.plot(df_x_jitter.loc[idx,['light off','light on']], dataframe_type_2.loc[idx,['light off','light on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def boxplot_SOCphase_light_paired_data_by_hue(df, ax=None, protocol='soc1', subject='mouse', hue='group', specific_hue='G1'):
    
    # Creating a first type of dataframe with three columns
    light1 = df[['meas_light1', subject, hue]][df.protocol == protocol]
    light1.rename(columns = {'meas_light1': 'freezing_value'}, inplace = True)
    light1['time_point'] = 'light1'
        
    light2 = df[['meas_light2', subject, hue]][df.protocol == protocol]
    light2.rename(columns = {'meas_light2': 'freezing_value'}, inplace = True)
    light2['time_point'] = 'light2'
        
    light3 = df[['meas_light3', subject, hue]][df.protocol == protocol]
    light3.rename(columns = {'meas_light3': 'freezing_value'}, inplace = True)
    light3['time_point'] = 'light3'
        
    light4 = df[['meas_light4', subject, hue]][df.protocol == protocol]
    light4.rename(columns = {'meas_light4': 'freezing_value'}, inplace = True)
    light4['time_point'] = 'light4'
    
    # This type of dataframe is ideal for the statistic analysis,
    # because it considers all levels of the between factor at the same time
    dataframe_type_1 = pd.concat([light1, light2, light3, light4])
    
    
    # Creating a second type of dataframe with four columns
    light1 = df[['meas_light1', subject, hue]][df.protocol == protocol]
    light1 = light1[light1[hue] == specific_hue].drop(hue, axis=1)
    light1.rename(columns = {'meas_light1': 'light1'}, inplace = True)
    
    light2 = df[['meas_light2', subject, hue]][df.protocol == protocol]
    light2 = light2[light2[hue] == specific_hue].drop(hue, axis=1)
    light2.rename(columns = {'meas_light2': 'light2'}, inplace = True)
    
    light3 = df[['meas_light3', subject, hue]][df.protocol == protocol]
    light3 = light3[light3[hue] == specific_hue].drop(hue, axis=1)
    light3.rename(columns = {'meas_light3': 'light3'}, inplace = True)
    
    light4 = df[['meas_light4', subject, hue]][df.protocol == protocol]
    light4 = light4[light4[hue] == specific_hue].drop(hue, axis=1)
    light4.rename(columns = {'meas_light4': 'light4'}, inplace = True)
    
    dataframe_type_2 = light1
    dataframe_type_2 = dataframe_type_2.merge(light2, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light3, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(light4, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    # For the boxplots with seaborn
    # sns.boxplot(data=mydataframe, showfliers=False, width=.3, 
    #             boxprops=dict(alpha=.3),        # Transparency of the boxplot
    #             medianprops=dict(alpha=.7),     # Transparency of the median bar
    #             whiskerprops=dict(alpha=.3),    # Transparency of the whiskers
    #             minimumprops=dict(alpha=.7),    # Transparency of the minimum
    #             notch=True,                     # Makes a shape like an sand clock
    #             ax=ax)
    
    # For the violinplots with seaborn
    # my_pal = {"tone off": "blue", "tone on": "blue", "light off": "red", "light on": "red"}
    # sns.violinplot(x = dataframe_type_1.time_point, y = dataframe_type_1.freezing_on_off,
    #                showfliers=False,
    #                linewidth = 0,
    #                 palette = my_pal,
    #                 alpha=.5,
    #                ax=ax)
    
    
    # Creating the violin plots
    for pc in ax.violinplot(dataframe_type_2[['light1', 'light2', 'light3', 'light4']], positions=[0,1,2,3], showmeans=False, showmedians=True)['bodies']:
        pc.set_facecolor('red')
        pc.set_edgecolor('red')
    
    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=1,       # Make the line on top of the dot
                ms=3)           # The size of the dot
    ax.set_xticks(range(len(dataframe_type_2.columns)))
    ax.set_xticklabels(dataframe_type_2.columns)
    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,['light1','light2']], dataframe_type_2.loc[idx,['light1','light2']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['light2','light3']], dataframe_type_2.loc[idx,['light2','light3']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)
        ax.plot(df_x_jitter.loc[idx,['light3','light4']], dataframe_type_2.loc[idx,['light3','light4']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=-1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def SOCphase_bullet_by_hue(df, ax=None, protocol='soc1', subject='mouse', hue='group', specific_hue='G2'):
    
    # Creating a first type of dataframe with three columns
    tone1 = df[['meas_tone1', subject, hue]][df.protocol == protocol]
    tone1.rename(columns = {'meas_tone1': 'freezing_value'}, inplace = True)
    tone1['time_point'] = 'trial 1'
    tone1['stimulus'] = 'tone'
        
    tone2 = df[['meas_tone2', subject, hue]][df.protocol == protocol]
    tone2.rename(columns = {'meas_tone2': 'freezing_value'}, inplace = True)
    tone2['time_point'] = 'trial 2'
    tone2['stimulus'] = 'tone'
        
    tone3 = df[['meas_tone3', subject, hue]][df.protocol == protocol]
    tone3.rename(columns = {'meas_tone3': 'freezing_value'}, inplace = True)
    tone3['time_point'] = 'trial 3'
    tone3['stimulus'] = 'tone'
        
    tone4 = df[['meas_tone4', subject, hue]][df.protocol == protocol]
    tone4.rename(columns = {'meas_tone4': 'freezing_value'}, inplace = True)
    tone4['time_point'] = 'trial 4'
    tone4['stimulus'] = 'tone'
    
    
    light1 = df[['meas_light1', subject, hue]][df.protocol == protocol]
    light1.rename(columns = {'meas_light1': 'freezing_value'}, inplace = True)
    light1['time_point'] = 'trial 1'
    light1['stimulus'] = 'light'
        
    light2 = df[['meas_light2', subject, hue]][df.protocol == protocol]
    light2.rename(columns = {'meas_light2': 'freezing_value'}, inplace = True)
    light2['time_point'] = 'trial 2'
    light2['stimulus'] = 'light'
        
    light3 = df[['meas_light3', subject, hue]][df.protocol == protocol]
    light3.rename(columns = {'meas_light3': 'freezing_value'}, inplace = True)
    light3['time_point'] = 'trial 3'
    light3['stimulus'] = 'light'
        
    light4 = df[['meas_light4', subject, hue]][df.protocol == protocol]
    light4.rename(columns = {'meas_light4': 'freezing_value'}, inplace = True)
    light4['time_point'] = 'trial 4'
    light4['stimulus'] = 'light'
    
    
    # This type of dataframe is ideal for the statistic analysis,
    # because it considers all levels of the between factor at the same time
    dataframe_type_1 = pd.concat([light1, light2, light3, light4, tone1, tone2, tone3, tone4])
    dataframe_type_1 = dataframe_type_1[dataframe_type_1[hue] == specific_hue]
    
    # sns.set_theme(style="whitegrid")
    
    colors = {"tone": "royalblue", "light": "salmon"}
    
    # Set up a grid to plot survival probability against several variables
    g = sns.catplot(data=dataframe_type_1, x='time_point', y="freezing_value", hue="stimulus",
                    kind="point", palette=colors,
                    capsize=.15, errwidth=0.75)
    
    g._legend.remove()
    g.set(xlabel=None)

    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    plt.legend(loc="upper right")
    return g


# =============================================================================
# CORRELATION PLOT
# =============================================================================


def correlation_OffOn_by_hue(df, ax=None, s1='s1', s2='s2', subject='mouse', hue='group', specific_hue='G1'):
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_slices1', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_slices1': s2 + '_off'}, inplace = True)
    
    s2_on = df[['meas_slices2', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_slices2': s2 + '_on'}, inplace = True)
    
    s1_off = df[['meas_slices1', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_slices1': s1 + '_off'}, inplace = True)
    
    s1_on = df[['meas_slices2', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_slices2': s1 + '_on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    dataframe_type_2[s2] = dataframe_type_2[s2 + '_on'] - dataframe_type_2[s2 + '_off']
    dataframe_type_2[s1] = dataframe_type_2[s1 + '_on'] - dataframe_type_2[s1 + '_off']
    dataframe_type_2 = dataframe_type_2.drop([s2 + '_off', s1 + '_off', s2 + '_on', s1 + '_on'], axis=1)
    
    sns.set_theme(style="whitegrid")
    
    g = sns.jointplot(x=s1, y=s2, data=dataframe_type_2,
                  kind="reg", truncate=False,
                  # xlim=(0, 60), ylim=(0, 12),
                  color='black')  
    plt.setp(g.ax_marg_y.patches, color="blue", alpha=.2)
    plt.setp(g.ax_marg_x.patches, color="red", alpha=.2)

    
    # r = str(pg.corr(dataframe_type_2[s1], dataframe_type_2[s2], method='pearson')['r'][0])[0:6]
    # pval = str(pg.corr(dataframe_type_2[s1], dataframe_type_2[s2], method='pearson')['p-val'][0])[0:6]
    
    pearson_coef, p_value = pearsonr(dataframe_type_2[s1], dataframe_type_2[s2])
    g.ax_joint.annotate(f"Pearson = {pearson_coef:.4f} \nPval = {p_value:.4f}", xy=(0.05, 0.9), xycoords="axes fraction")

    g.ax_joint.set_xlabel('Direct learning: ' + s1[:-1])
    g.ax_joint.set_ylabel('Mediated learning: ' + s2[:-1])
    return g


def correlation_OffOn_1_mins_by_hue(df, ax=None, s1='s1', s2='s2', subject='mouse', hue='group', specific_hue='light-shock'):
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_off_1min', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_off_1min': s2 + '_off'}, inplace = True)
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_on_1min': s2 + '_on'}, inplace = True)
    
    s1_off = df[['meas_off_1min', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_off_1min': s1 + '_off'}, inplace = True)
    
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_on_1min': s1 + '_on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    dataframe_type_2[s2] = dataframe_type_2[s2 + '_on'] - dataframe_type_2[s2 + '_off']
    dataframe_type_2[s1] = dataframe_type_2[s1 + '_on'] - dataframe_type_2[s1 + '_off']
    dataframe_type_2 = dataframe_type_2.drop([s2 + '_off', s1 + '_off', s2 + '_on', s1 + '_on'], axis=1)
    
    sns.set_theme(style="whitegrid")
    
    g = sns.jointplot(x=s1, y=s2, data=dataframe_type_2,
                  kind="reg", truncate=False,
                  # xlim=(0, 60), ylim=(0, 12),
                  color='black')  
    plt.setp(g.ax_marg_y.patches, color="blue", alpha=.2)
    plt.setp(g.ax_marg_x.patches, color="red", alpha=.2)

    
    # r = str(pg.corr(dataframe_type_2[s1], dataframe_type_2[s2], method='pearson')['r'][0])[0:6]
    # pval = str(pg.corr(dataframe_type_2[s1], dataframe_type_2[s2], method='pearson')['p-val'][0])[0:6]
    
    pearson_coef, p_value = pearsonr(dataframe_type_2[s1], dataframe_type_2[s2])
    g.ax_joint.annotate(f"Pearson = {pearson_coef:.4f} \nPval = {p_value:.4f}", xy=(0.05, 0.9), xycoords="axes fraction")

    g.ax_joint.set_xlabel('Direct learning: ' + s1[:-1])
    g.ax_joint.set_ylabel('Mediated learning: ' + s2[:-1])
    return g


def correlation_OffOn_lastmin_1min_by_hue(df, ax=None, s1='s1', s2='s2', subject='mouse', hue='group', specific_hue='young_male'):
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_off_last', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_off_last': s2 + '_off'}, inplace = True)
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_on_1min': s2 + '_on'}, inplace = True)
    
    s1_off = df[['meas_off_last', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_off_last': s1 + '_off'}, inplace = True)
    
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_on_1min': s1 + '_on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    dataframe_type_2[s2] = dataframe_type_2[s2 + '_on'] - dataframe_type_2[s2 + '_off']
    dataframe_type_2[s1] = dataframe_type_2[s1 + '_on'] - dataframe_type_2[s1 + '_off']
    dataframe_type_2 = dataframe_type_2.drop([s2 + '_off', s1 + '_off', s2 + '_on', s1 + '_on'], axis=1)
    
    sns.set_theme(style="whitegrid")
    
    g = sns.jointplot(x=s1, y=s2, data=dataframe_type_2,
                  kind="reg", truncate=False,
                  # xlim=(0, 60), ylim=(0, 12),
                  color='black')  
    plt.setp(g.ax_marg_y.patches, color="blue", alpha=.2)
    plt.setp(g.ax_marg_x.patches, color="red", alpha=.2)

    
    # r = str(pg.corr(dataframe_type_2[s1], dataframe_type_2[s2], method='pearson')['r'][0])[0:6]
    # pval = str(pg.corr(dataframe_type_2[s1], dataframe_type_2[s2], method='pearson')['p-val'][0])[0:6]
    
    pearson_coef, p_value = pearsonr(dataframe_type_2[s1], dataframe_type_2[s2])
    g.ax_joint.annotate(f"Pearson = {pearson_coef:.4f} \nPval = {p_value:.4f}", xy=(0.05, 0.9), xycoords="axes fraction")

    # g.ax_joint.set_xlabel('Direct learning: ' + s1)
    # g.ax_joint.set_ylabel('Mediated learning: ' + s2)
    g.ax_joint.set_xlabel('Direct learning: light')
    g.ax_joint.set_ylabel('Mediated learning: tone')

    return g

# =============================================================================
# BAR PLOTS
# =============================================================================


def barplot_OffOn_paired_data_by_hue(df, ax=None, s1='s1', s2='s2', subject='mouse', hue='group', specific_hue='G2'):
    
    # Creating a first type of dataframe with three columns:
    s2_off = df[['meas_slices1', subject, hue]][df.protocol == s2]
    s2_off.rename(columns = {'meas_slices1': 'freezing_on_off'}, inplace = True)
    s2_off['time_point'] = s2 + ' off'
    s2_off['cue'] = s2
    s2_off['period'] = 'off'
    
    s2_on = df[['meas_slices2', subject, hue]][df.protocol == s2]
    s2_on.rename(columns = {'meas_slices2': 'freezing_on_off'}, inplace = True)
    s2_on['time_point'] = s2 + ' on'
    s2_on['cue'] = s2
    s2_on['period'] = 'on'

    s1_off = df[['meas_slices1', subject, hue]][df.protocol == s1]
    s1_off.rename(columns = {'meas_slices1': 'freezing_on_off'}, inplace = True)
    s1_off['time_point'] = s1 + ' off'
    s1_off['cue'] = s1
    s1_off['period'] = 'off'

    s1_on = df[['meas_slices2', subject, hue]][df.protocol == s1]
    s1_on.rename(columns = {'meas_slices2': 'freezing_on_off'}, inplace = True)
    s1_on['time_point'] = s1 + ' on'
    s1_on['cue'] = s1
    s1_on['period'] = 'on'

    dataframe_type_1 = pd.concat([s2_off, s2_on, s1_off, s1_on])
    
    # stats_df = create_dataframe_type_1(df)
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_slices1', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_slices1': s2 + ' off'}, inplace = True)
    
    s2_on = df[['meas_slices2', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_slices2': s2 + ' on'}, inplace = True)
    
    s1_off = df[['meas_slices1', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_slices1': s1 + ' off'}, inplace = True)
    
    s1_on = df[['meas_slices2', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_slices2': s1 + ' on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    
    # Creating the bar plots
            
    s2_data = np.mean(dataframe_type_2[s2 + ' on'])
    s1_data = np.mean(dataframe_type_2[s1 + ' on'])
    off_data = [np.mean(dataframe_type_2[s2 + ' off']), np.mean(dataframe_type_2[s1 + ' off'])]
    
    s2_error = np.std(dataframe_type_2[s2 + ' on'], ddof=1)
    s1_error = np.std(dataframe_type_2[s1 + ' on'], ddof=1)
    off_error = [np.std(dataframe_type_2[s2 + ' off'], ddof=1), np.std(dataframe_type_2[s1 + ' off'], ddof=1)]
    
    s2_positions = 1
    s1_positions = 3
    off_positions = [0, 2]
    
    bar_width = 0.6
    
    ax.bar(off_positions, off_data, color='moccasin', edgecolor='black', width=bar_width, label='off period')
    ax.bar(s2_positions, s2_data, color='cornflowerblue', edgecolor='black', width=bar_width, label=s2[:-1])
    ax.bar(s1_positions, s1_data, color='salmon', edgecolor='black', width=bar_width, label=s1[:-1])
    
    ax.errorbar(off_positions, off_data, yerr=off_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(s2_positions, s2_data, yerr=s2_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(s1_positions, s1_data, yerr=s1_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(['Mediated learning: ' + s2[:-1], 'Direct learning: ' + s1[:-1]])
    
    ax.legend()
    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=2,       # Make the line on top of the dot
                ms=3)           # The size of the dot

    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2 + ' off',s2 + ' on']], dataframe_type_2.loc[idx,[s2 + ' off',s2 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
        ax.plot(df_x_jitter.loc[idx,[s1 + ' off',s1 + ' on']], dataframe_type_2.loc[idx,[s1 + ' off',s1 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
        ax.plot(df_x_jitter.loc[idx,[s2 + ' on',s1 + ' off']], dataframe_type_2.loc[idx,[s2 + ' on',s1 +' off']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)


    plt.ylim(0,60)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def barplot_OffOn_1_mins_paired_data_by_hue(df, ax=None, s1='s1', s2='s2', subject='mouse', hue='group', specific_hue='light-shock'):
         
    
    # Creating a first type of dataframe with three columns:
    s2_off = df[['meas_off_1min', subject, hue]][df.protocol == s2]
    s2_off.rename(columns = {'meas_off_1min': 'freezing_on_off'}, inplace = True)
    s2_off['time_point'] = s2 + ' off'
    s2_off['cue'] = s2
    s2_off['period'] = 'off'
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on.rename(columns = {'meas_on_1min': 'freezing_on_off'}, inplace = True)
    s2_on['time_point'] = s2 + ' on'
    s2_on['cue'] = s2
    s2_on['period'] = 'on'

    s1_off = df[['meas_off_1min', subject, hue]][df.protocol == s1]
    s1_off.rename(columns = {'meas_off_1min': 'freezing_on_off'}, inplace = True)
    s1_off['time_point'] = s1 + ' off'
    s1_off['cue'] = s1
    s1_off['period'] = 'off'

    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on.rename(columns = {'meas_on_1min': 'freezing_on_off'}, inplace = True)
    s1_on['time_point'] = s1 + ' on'
    s1_on['cue'] = s1
    s1_on['period'] = 'on'

    dataframe_type_1 = pd.concat([s2_off, s2_on, s1_off, s1_on])
    
    # Create a dataframe for GraphPad Prism analysis
    directory = 'C:/Users/mcanela/Desktop/'
    dataframe_type_1.to_csv(directory + 'Freezing.csv')

        
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_off_1min', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_off_1min': s2 + ' off'}, inplace = True)
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_on_1min': s2 + ' on'}, inplace = True)
    
    s1_off = df[['meas_off_1min', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_off_1min': s1 + ' off'}, inplace = True)
    
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_on_1min': s1 + ' on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    
    # Creating the bar plots
            
    s2_data = np.mean(dataframe_type_2[s2 + ' on'])
    s1_data = np.mean(dataframe_type_2[s1 + ' on'])
    off_data = [np.mean(dataframe_type_2[s2 + ' off']), np.mean(dataframe_type_2[s1 + ' off'])]
    
    s2_error = np.std(dataframe_type_2[s2 + ' on'], ddof=1)
    s1_error = np.std(dataframe_type_2[s1 + ' on'], ddof=1)
    off_error = [np.std(dataframe_type_2[s2 + ' off'], ddof=1), np.std(dataframe_type_2[s1 + ' off'], ddof=1)]
    
    s2_positions = 1
    s1_positions = 3
    off_positions = [0, 2]
    
    bar_width = 0.6
    
    ax.bar(off_positions, off_data, color='moccasin', edgecolor='black', width=bar_width, label='off period')
    ax.bar(s2_positions, s2_data, color='cornflowerblue', edgecolor='black', width=bar_width, label=s2[:-1])
    ax.bar(s1_positions, s1_data, color='salmon', edgecolor='black', width=bar_width, label=s1[:-1])
    
    ax.errorbar(off_positions, off_data, yerr=off_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(s2_positions, s2_data, yerr=s2_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(s1_positions, s1_data, yerr=s1_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5, 2.5])
    ax.set_xticklabels(['Mediated learning: ' + s2[:-1], 'Direct learning: ' + s1[:-1]])
    
    ax.legend()
    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=2,       # Make the line on top of the dot
                ms=3)           # The size of the dot

    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2 + ' off',s2 + ' on']], dataframe_type_2.loc[idx,[s2 + ' off',s2 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
        ax.plot(df_x_jitter.loc[idx,[s1 + ' off',s1 + ' on']], dataframe_type_2.loc[idx,[s1 + ' off',s1 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
        ax.plot(df_x_jitter.loc[idx,[s2 + ' on',s1 + ' off']], dataframe_type_2.loc[idx,[s2 + ' on',s1 +' off']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


def barplot_OffOn_lastmin_1min_paired_data_by_hue(df, ax=None, s1='s1', s2='s2', subject='mouse', hue='group', specific_hue='tone-shock'):
         
    
    # Creating a first type of dataframe with three columns:
    s2_off = df[['meas_off_last', subject, hue]][df.protocol == s2]
    s2_off.rename(columns = {'meas_off_last': 'freezing_on_off'}, inplace = True)
    s2_off['time_point'] = s2 + ' off'
    s2_off['cue'] = s2
    s2_off['period'] = 'off'
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on.rename(columns = {'meas_on_1min': 'freezing_on_off'}, inplace = True)
    s2_on['time_point'] = s2 + ' on'
    s2_on['cue'] = s2
    s2_on['period'] = 'on'

    s1_off = df[['meas_off_last', subject, hue]][df.protocol == s1]
    s1_off.rename(columns = {'meas_off_last': 'freezing_on_off'}, inplace = True)
    s1_off['time_point'] = s1 + ' off'
    s1_off['cue'] = s1
    s1_off['period'] = 'off'

    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on.rename(columns = {'meas_on_1min': 'freezing_on_off'}, inplace = True)
    s1_on['time_point'] = s1 + ' on'
    s1_on['cue'] = s1
    s1_on['period'] = 'on'

    dataframe_type_1 = pd.concat([s2_off, s2_on, s1_off, s1_on])
        
    
    # Creating a second type of dataframe with four columns
    s2_off = df[['meas_off_last', subject, hue]][df.protocol == s2]
    s2_off = s2_off[s2_off[hue] == specific_hue].drop(hue, axis=1)
    s2_off.rename(columns = {'meas_off_last': s2 + ' off'}, inplace = True)
    
    s2_on = df[['meas_on_1min', subject, hue]][df.protocol == s2]
    s2_on = s2_on[s2_on[hue] == specific_hue].drop(hue, axis=1)
    s2_on.rename(columns = {'meas_on_1min': s2 + ' on'}, inplace = True)
    
    s1_off = df[['meas_off_last', subject, hue]][df.protocol == s1]
    s1_off = s1_off[s1_off[hue] == specific_hue].drop(hue, axis=1)
    s1_off.rename(columns = {'meas_off_last': s1 + ' off'}, inplace = True)
    
    s1_on = df[['meas_on_1min', subject, hue]][df.protocol == s1]
    s1_on = s1_on[s1_on[hue] == specific_hue].drop(hue, axis=1)
    s1_on.rename(columns = {'meas_on_1min': s1 + ' on'}, inplace = True)
    
    dataframe_type_2 = s2_off
    dataframe_type_2 = dataframe_type_2.merge(s2_on, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_off, how='left', on=subject)
    dataframe_type_2 = dataframe_type_2.merge(s1_on, how='left', on=subject).drop(subject, axis=1)
    
    
    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    
    # Creating the bar plots
            
    s2_data = np.mean(dataframe_type_2[s2 + ' on'])
    s1_data = np.mean(dataframe_type_2[s1 + ' on'])
    off_data = [np.mean(dataframe_type_2[s2 + ' off']), np.mean(dataframe_type_2[s1 + ' off'])]
    
    s2_error = np.std(dataframe_type_2[s2 + ' on'], ddof=1)
    s1_error = np.std(dataframe_type_2[s1 + ' on'], ddof=1)
    off_error = [np.std(dataframe_type_2[s2 + ' off'], ddof=1), np.std(dataframe_type_2[s1 + ' off'], ddof=1)]
    
    s2_positions = 1
    s1_positions = 3
    off_positions = [0, 2]
    
    bar_width = 0.6
    
    ax.bar(off_positions, off_data, color='moccasin', edgecolor='black', width=bar_width, label='off period')
    ax.bar(s2_positions, s2_data, color='cornflowerblue', edgecolor='black', width=bar_width, label=s2)
    ax.bar(s1_positions, s1_data, color='salmon', edgecolor='black', width=bar_width, label=s1)
    
    ax.errorbar(off_positions, off_data, yerr=off_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(s2_positions, s2_data, yerr=s2_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(s1_positions, s1_data, yerr=s1_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5, 2.5])
    # ax.set_xticklabels(['Mediated learning: ' + s2, 'Direct learning: ' + s1])
    ax.set_xticklabels(['Mediated learning: light', 'Direct learning: tone'])

    ax.legend()
    
    # For the stripplot
    jitter = 0.15           # Dots dispersion
    df_x_jitter = pd.DataFrame(np.random.normal(loc=0, scale=jitter, size=dataframe_type_2.values.shape), columns=dataframe_type_2.columns)
    df_x_jitter += np.arange(len(dataframe_type_2.columns))

    for col in dataframe_type_2:
        ax.plot(df_x_jitter[col], dataframe_type_2[col],
                'o',            # Dot shape
                alpha=.7,       # Dot color transparency
                color='black',  # Set color for the dots grey
                zorder=2,       # Make the line on top of the dot
                ms=3)           # The size of the dot

    ax.set_xlim(-0.5,len(dataframe_type_2.columns)-0.5)

    # To draw the lines between the paired values
    for idx in dataframe_type_2.index:
        ax.plot(df_x_jitter.loc[idx,[s2 + ' off',s2 + ' on']], dataframe_type_2.loc[idx,[s2 + ' off',s2 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
        ax.plot(df_x_jitter.loc[idx,[s1 + ' off',s1 + ' on']], dataframe_type_2.loc[idx,[s1 + ' off',s1 + ' on']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)
        ax.plot(df_x_jitter.loc[idx,[s2 + ' on',s1 + ' off']], dataframe_type_2.loc[idx,[s2 + ' on',s1 +' off']], color = 'grey', linewidth = 0.5, linestyle = '--', zorder=1)


    plt.ylim(0,100)
    plt.ylabel('Freezing percentage')
    plt.xlabel('')
    return ax


# =============================================================================
# GRAPHPAD PLOTS
# =============================================================================


def on_off_filter(df, protocol, column, hue, specific_hue):
    
    '''
    Last minute OFF period (2-3 min): meas_off_last
    First minute ON period (3-4 min): meas_on_1min
    
    Last 10s OFF (170-180 min): freezing_18
    First 10s ON (180-190 min): freezing_19
    '''
    
    copy_supervised_annotation = copy.deepcopy(df)
    filtered = copy_supervised_annotation[copy_supervised_annotation['protocol'] == protocol]
    
    if isinstance(hue, str) and len(hue) > 0:
        filtered = filtered[filtered[hue] == specific_hue]
    
    mean_values = filtered[column].tolist()

    return mean_values


def barplot_OffOn(df, ax=None, protocol='s1', off_column='freezing_18', on_column='freezing_19', hue='', specific_hue=''):

    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    off_position = 0
    on_position = 1
    # bar_width = 0.6
    
    off_list = on_off_filter(df, protocol, off_column, hue, specific_hue)
    on_list = on_off_filter(df, protocol, on_column, hue, specific_hue)
                  
    off_data = np.mean(off_list)
    on_data = np.mean(on_list)

    off_error = np.std(off_list, ddof=1)
    on_error = np.std(on_list, ddof=1)
    
    if protocol == 's1':
        on_color = 'salmon'
        on_edge = 'darkred'
    elif protocol == 's2':
        on_color = 'cornflowerblue'
        on_edge = 'darkblue'
    else:
        on_color = 'moccasin'
        on_edge = 'darkorange'
    
    # ax.bar(off_position, off_data, color='moccasin', edgecolor='black', width=bar_width)
    # ax.bar(on_position, on_data, color=on_color, edgecolor='black', width=bar_width)
    
    ax.hlines(off_data, xmin=-0.25, xmax=0.25, color='black', linewidth=1)
    ax.hlines(on_data, xmin=0.75, xmax=1.25, color='black', linewidth=1)
    
    ax.errorbar(off_position, off_data, yerr=off_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(on_position, on_data, yerr=on_error, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5])
    ax.set_xticklabels([])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_off = np.random.normal(loc=off_position, scale=jitter, size=len(off_list)).tolist()
    ax.plot(dispersion_values_off, off_list,
            'o',                            
            markerfacecolor='moccasin',    
            markeredgecolor='darkorange',
            markeredgewidth=1,
            markersize=5, 
            label='Absent stimulus')      
    
    dispersion_values_on = np.random.normal(loc=on_position, scale=jitter, size=len(on_list)).tolist()
    ax.plot(dispersion_values_on, on_list,
            'o',                          
            markerfacecolor=on_color,    
            markeredgecolor=on_edge,
            markeredgewidth=1,
            markersize=5, 
            label=protocol)               
    
    if len(off_list) == len(on_list):
        for x in range(len(off_list)):
            ax.plot([dispersion_values_off[x], dispersion_values_on[x]], [off_list[x], on_list[x]], color = 'black', linestyle='-', linewidth=0.5)
        
    ax.set_ylim(0,100)
    ax.set_xlabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05))
    
    pvalue = pg.ttest(off_list, on_list, paired=True)['p-val'][0]
    
    def convert_pvalue_to_asterisks(pvalue):
        ns = "ns (p=" + str(pvalue)[1:4] + ")"
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ns

    y, h, col = max(max(off_list), max(on_list)) + 5, 2, 'k'
    
    ax.plot([off_position, off_position, on_position, on_position], [y, y+h, y+h, y], lw=1.5, c=col)
    
    if pvalue > 0.05:
        ax.text((off_position+on_position)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    elif pvalue <= 0.05:    
        ax.text((off_position+on_position)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    return ax


def multiplot_OffOn(df):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5, 4))  # 1 row, 2 columns
    
    # Plot data on ax1
    barplot_OffOn(df, ax=ax1, protocol='s1', off_column='meas_off_last', on_column='meas_on_1min', hue='', specific_hue='')
    
    # Plot data on ax2
    barplot_OffOn(df, ax=ax2, protocol='s2', off_column='meas_off_last', on_column='meas_on_1min', hue='', specific_hue='')

    # Adjust the spacing between subplots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.suptitle('Immobility (%): 2-3 vs 3-4 min')
    # plt.suptitle('Immobility (%): 170-180 vs 180-190 s')
    
    # Show the plots
    plt.show()


def discrimination_index(df, ax=None, index='di', protocol_1='s1', protocol_2='s2', off_column='meas_off_last', on_column='meas_on_1min', hue='group', specific_hue='PU'):

    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    position_1 = 0
    position_2 = 1
    # bar_width = 0.6
    
    off_list_1 = on_off_filter(df, protocol_1, off_column, hue, specific_hue)
    on_list_1 = on_off_filter(df, protocol_1, on_column, hue, specific_hue)
    
    off_list_2 = on_off_filter(df, protocol_2, off_column, hue, specific_hue)
    on_list_2 = on_off_filter(df, protocol_2, on_column, hue, specific_hue)
    
    # DISCRIMINATION INDEX
    if index == 'di':
        ax.set_ylim(-1,1)
        ax.axhline(y=0, color='black', linestyle='--')
        if len(off_list_1) == len(on_list_1):
            subtraction_1 = [x - y for x, y in zip(on_list_1, off_list_1)]
            addition_1 = [x + y for x, y in zip(on_list_1, off_list_1)]
            discrimination_1 = [x / y for x, y in zip(subtraction_1, addition_1)]
        if len(off_list_2) == len(on_list_2):
            subtraction_2 = [x - y for x, y in zip(on_list_2, off_list_2)]
            addition_2 = [x + y for x, y in zip(on_list_2, off_list_2)]
            discrimination_2 = [x / y for x, y in zip(subtraction_2, addition_2)]
        
    # GENERALIZATION INDEX
    elif index == 'gi':
        ax.set_ylim(0,1)
        if len(off_list_1) == len(on_list_1):
            addition_1 = [x + y for x, y in zip(on_list_1, off_list_1)]
            discrimination_1 = [x / y for x, y in zip(off_list_1, addition_1)]
        if len(off_list_2) == len(on_list_2):
            addition_2 = [x + y for x, y in zip(on_list_2, off_list_2)]
            discrimination_2 = [x / y for x, y in zip(off_list_2, addition_2)]    
                  
    discrimination_data_1 = np.mean(discrimination_1)
    discrimination_data_2 = np.mean(discrimination_2)

    discrimination_error_1 = np.std(discrimination_1, ddof=1)
    discrimination_error_2 = np.std(discrimination_2, ddof=1)
    
    if protocol_1 == 's1':
        on_color_1 = 'salmon'
        on_edge_1 = 'darkred'
    elif protocol_1 == 's2':
        on_color_1 = 'cornflowerblue'
        on_edge_1 = 'darkblue'
    else:
        on_color_1 = 'moccasin'
        on_edge_1 = 'darkorange'
        
    if protocol_2 == 's1':
        on_color_2 = 'salmon'
        on_edge_2 = 'darkred'
    elif protocol_2 == 's2':
        on_color_2 = 'cornflowerblue'
        on_edge_2 = 'darkblue'
    else:
        on_color_2 = 'moccasin'
        on_edge_2 = 'darkorange'
    
    ax.hlines(discrimination_data_1, xmin=-0.25, xmax=0.25, color='black', linewidth=1)
    ax.hlines(discrimination_data_2, xmin=0.75, xmax=1.25, color='black', linewidth=1)
    
    ax.errorbar(position_1, discrimination_data_1, yerr=discrimination_error_1, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(position_2, discrimination_data_2, yerr=discrimination_error_2, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5])
    ax.set_xticklabels([])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_off = np.random.normal(loc=position_1, scale=jitter, size=len(discrimination_1)).tolist()
    ax.plot(dispersion_values_off, discrimination_1,
            'o',                            
            markerfacecolor=on_color_1,    
            markeredgecolor=on_edge_1,
            markeredgewidth=1,
            markersize=5, 
            label=protocol_1)      
    
    dispersion_values_on = np.random.normal(loc=position_2, scale=jitter, size=len(discrimination_2)).tolist()
    ax.plot(dispersion_values_on, discrimination_2,
            'o',                          
            markerfacecolor=on_color_2,    
            markeredgecolor=on_edge_2,
            markeredgewidth=1,
            markersize=5, 
            label=protocol_2)               
    
    # if len(discrimination_1) == len(discrimination_2):
    #     for x in range(len(discrimination_1)):
    #         ax.plot([dispersion_values_off[x], dispersion_values_on[x]], [discrimination_1[x], discrimination_2[x]], color = 'black', linestyle='-', linewidth=0.5)
            
    pvalue = pg.ttest(discrimination_1, discrimination_2, paired=True)['p-val'][0]
    
    def convert_pvalue_to_asterisks(pvalue):
        ns = "ns (p=" + str(pvalue)[1:4] + ")"
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ns

    y, h, col = max(max(discrimination_1), max(discrimination_2)) + 0.05, 0.02, 'k'
    
    ax.plot([position_1, position_1, position_2, position_2], [y, y+h, y+h, y], lw=1.5, c=col)
    
    if pvalue > 0.05:
        ax.text((position_1+position_2)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    elif pvalue <= 0.05:    
        ax.text((position_1+position_2)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    ax.set_xlabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    return ax


def rmsd(df, ax=None, protocol_1='s1', protocol_2='s2', hue='', specific_hue=''):

    if ax is None:
        fig, ax = plt.subplots()
    
    sns.set_theme(style="whitegrid")
    
    position_1 = 0
    position_2 = 1
    # bar_width = 0.6
    
    protocols = [protocol_1, protocol_2]
    protocols_dict = {}
    
    for protocol in protocols:
        data_1 = on_off_filter(df, protocol, 'meas_off_last', hue, specific_hue)
        data_2 = on_off_filter(df, protocol, 'meas_on_1min', hue, specific_hue)
        data_3 = on_off_filter(df, protocol, 'meas_off2', hue, specific_hue)
        data_4 = on_off_filter(df, protocol, 'meas_on_last', hue, specific_hue)
        protocols_dict[protocol] = [data_1, data_2, data_3, data_4]
    
    # RMSD
    data_dict = {}
    for key, value in protocols_dict.items():
        rmsd_list = []
        for i in range(12):
            squared_deviations = [(value[0][i] - sum(protocols_dict[protocol_1][0])/12)**2,
                                  (value[1][i] - sum(protocols_dict[protocol_1][1])/12)**2,
                                  (value[2][i] - sum(protocols_dict[protocol_1][2])/12)**2,
                                  (value[3][i] - sum(protocols_dict[protocol_1][3])/12)**2]
            # squared_deviations = [(value[0][i] - 20)**2,
            #                       (value[1][i] - 70)**2,
            #                       (value[2][i] - 20)**2,
            #                       (value[3][i] - 60)**2]

            rmsd = np.sqrt(sum(squared_deviations) / 4)
            rmsd_list.append(rmsd)
        data_dict[key] = rmsd_list
    
    discrimination_1 = data_dict[protocol_1]
    discrimination_2 = data_dict[protocol_2]
                  
    discrimination_data_1 = np.mean(discrimination_1)
    discrimination_data_2 = np.mean(discrimination_2)

    discrimination_error_1 = np.std(discrimination_1, ddof=1)
    discrimination_error_2 = np.std(discrimination_2, ddof=1)
    
    if protocol_1 == 's1':
        on_color_1 = 'salmon'
        on_edge_1 = 'darkred'
    elif protocol_1 == 's2':
        on_color_1 = 'cornflowerblue'
        on_edge_1 = 'darkblue'
    else:
        on_color_1 = 'moccasin'
        on_edge_1 = 'darkorange'
        
    if protocol_2 == 's1':
        on_color_2 = 'salmon'
        on_edge_2 = 'darkred'
    elif protocol_2 == 's2':
        on_color_2 = 'cornflowerblue'
        on_edge_2 = 'darkblue'
    else:
        on_color_2 = 'moccasin'
        on_edge_2 = 'darkorange'
    
    ax.hlines(discrimination_data_1, xmin=-0.25, xmax=0.25, color='black', linewidth=1)
    ax.hlines(discrimination_data_2, xmin=0.75, xmax=1.25, color='black', linewidth=1)
    
    ax.errorbar(position_1, discrimination_data_1, yerr=discrimination_error_1, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    ax.errorbar(position_2, discrimination_data_2, yerr=discrimination_error_2, lolims=False, capsize = 3, ls='None', color='k', zorder=-1)
    
    ax.set_xticks([0.5])
    ax.set_xticklabels([])
    
    jitter = 0.15 # Dots dispersion
    
    dispersion_values_off = np.random.normal(loc=position_1, scale=jitter, size=len(discrimination_1)).tolist()
    ax.plot(dispersion_values_off, discrimination_1,
            'o',                            
            markerfacecolor=on_color_1,    
            markeredgecolor=on_edge_1,
            markeredgewidth=1,
            markersize=5, 
            label=protocol_1)      
    
    dispersion_values_on = np.random.normal(loc=position_2, scale=jitter, size=len(discrimination_2)).tolist()
    ax.plot(dispersion_values_on, discrimination_2,
            'o',                          
            markerfacecolor=on_color_2,    
            markeredgecolor=on_edge_2,
            markeredgewidth=1,
            markersize=5, 
            label=protocol_2)               
    
    # if len(discrimination_1) == len(discrimination_2):
    #     for x in range(len(discrimination_1)):
    #         ax.plot([dispersion_values_off[x], dispersion_values_on[x]], [discrimination_1[x], discrimination_2[x]], color = 'black', linestyle='-', linewidth=0.5)
            
    pvalue = pg.ttest(discrimination_1, discrimination_2, paired=True)['p-val'][0]
    
    def convert_pvalue_to_asterisks(pvalue):
        ns = "ns (p=" + str(pvalue)[1:4] + ")"
        if pvalue <= 0.0001:
            return "****"
        elif pvalue <= 0.001:
            return "***"
        elif pvalue <= 0.01:
            return "**"
        elif pvalue <= 0.05:
            return "*"
        return ns

    y, h, col = max(max(discrimination_1), max(discrimination_2)) + 0.05, 0.02, 'k'
    
    ax.plot([position_1, position_1, position_2, position_2], [y, y+h, y+h, y], lw=1.5, c=col)
    
    if pvalue > 0.05:
        ax.text((position_1+position_2)*.5, y+2*h, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=11)
    elif pvalue <= 0.05:    
        ax.text((position_1+position_2)*.5, y, convert_pvalue_to_asterisks(pvalue), ha='center', va='bottom', color=col, size=18)
    
    ax.set_xlabel('')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)

    return ax

  
def multiplot_index(df):
    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6, 4))  # 1 row, 2 columns
    
    # Plot data on ax1
    discrimination_index(df, ax=ax1, index='di', protocol_1='s1', protocol_2='s2', off_column='meas_off_last', on_column='meas_on_1min', hue='', specific_hue='')
    ax1.set_title('Discrimination Index (D.I.)')
    
    # Plot data on ax2
    discrimination_index(df, ax=ax2, index='gi', protocol_1='s1', protocol_2='s2', off_column='meas_off_last', on_column='meas_on_1min', hue='', specific_hue='')
    ax2.set_title('Generalization Index (G.I.)')

    # Adjust the spacing between subplots
    plt.tight_layout()  # Increase the pad value to increase the space between plots
    plt.subplots_adjust(top=0.85, bottom=0.15)
    
    plt.suptitle('2-3 vs 3-4 min')
    # plt.suptitle('170-180 vs 180-190 s')
    
    # Show the plots
    plt.show()


def easy_index(df, ax=None, protocol_1='s1', protocol_2='s2', hue='group', specific_hue='PP'):

    if ax is None:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
    protocols = [protocol_1, protocol_2]
    protocols_dict = {}
    
    for protocol in protocols:
        data_1 = on_off_filter(df, protocol, 'meas_off_last', hue, specific_hue)
        data_2 = on_off_filter(df, protocol, 'meas_on_1min', hue, specific_hue)
        data_3 = on_off_filter(df, protocol, 'meas_off2', hue, specific_hue)
        data_4 = on_off_filter(df, protocol, 'meas_on_last', hue, specific_hue)
        protocols_dict[protocol] = [data_1, data_2, data_3, data_4]
    
    # EASY INDEX
    data_dict = {}
    for key, value in protocols_dict.items():
        rmsd_list = []
        for i in range(12):
            squared_deviations = [value[0][i] - value[1][i],
                                  value[1][i] - value[2][i],
                                  value[2][i] - value[3][i]]
                                   
            # if squared_deviations[0] < -3:
            if squared_deviations[0] < 0:
                squared_deviations[0] = 1
            else:
                squared_deviations[0] = 0
                
                
            # if squared_deviations[1] > 3:
            if squared_deviations[1] > 0:
                squared_deviations[1] = 1
            else:
                squared_deviations[1] = 0
                
                
            # if squared_deviations[2] < -3:
            if squared_deviations[2] < 0:
                squared_deviations[2] = 1
            else:
                squared_deviations[2] = 0
                
                
            rmsd = sum(squared_deviations)
            rmsd_list.append(rmsd)
        data_dict[key] = rmsd_list
        
    tags = ['full learners', 'mid learners', 'bad learners', 'non learners']
    colors = ['#208e00', '#a4b800', '#e36500', '#ff0000']
    
    discrimination_1 = data_dict[protocol_1]
    values_1 = [discrimination_1.count(3), discrimination_1.count(2), discrimination_1.count(1), discrimination_1.count(0)]
    colors_dict_1 = dict(zip(tags, colors))
    values_1_dict = dict(zip(tags, values_1))
    for key, value in list(values_1_dict.items()):
        if value == 0:
            del colors_dict_1[key]
            del values_1_dict[key]

    
    discrimination_2 = data_dict[protocol_2]
    values_2 = [discrimination_2.count(3), discrimination_2.count(2), discrimination_2.count(1), discrimination_2.count(0)]
    colors_dict_2 = dict(zip(tags, colors))
    values_2_dict = dict(zip(tags, values_2))
    for key, value in list(values_2_dict.items()):
        if value == 0:
            del colors_dict_2[key]
            del values_2_dict[key]
         
            
    ax1.pie(values_1_dict.values(), autopct='%1.1f%%', labels=values_1_dict.keys(), colors=colors_dict_1.values())
    ax1.set_title(protocol_1.capitalize())
    
    ax2.pie(values_2_dict.values(), autopct='%1.1f%%', labels=values_2_dict.keys(), colors=colors_dict_2.values())
    ax2.set_title(protocol_2.capitalize())

    plt.show()






























