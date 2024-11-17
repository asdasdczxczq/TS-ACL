import numpy as np
import matplotlib.pyplot as plt
import statistics
#F27970
#BB9727
#54B345
#32B897
#05B9E2
#8983BF
#C76DA2

#A1A9D0
#F0988C
#B883D4
#9E9E9E
#CFEAF1
#C4A5DE
#F6CAE5
#96CCCB

agents = ['Offline', '0.01', '0.1', '1', '10', '100']

colors = {

    'Offline': 'black',
    '0.01': '#A9A9A9',     # 灰色，柔和的中性色
    '0.1': '#B0D5DF',      # 浅蓝绿色，清新但不过于饱和
    '1': '#AB91C8',        # 中性紫色，柔和且协调
    '10': '#E5A1C0',       # 柔和的粉色，显眼但不刺目
    '100': '#7DB6A6',      # 中性青绿色，稳重且不失层次
}

linestyles = {
    # 'Naive': '-',
    'Offline': '-',
    '0.01': '-',
    '0.1': '-',
    '1': '-',
    '10': '-',
    '100': '-',
}

markers = {
    # 'Naive': '^',
    'Offline': 'o',
    '0.01': 'o',
    '0.1': 'o',
    '1': 'o',
    '10': 'o',
    '100': 'o',
}

markersize = {
    # 'Naive': 8,
    'Offline': 0,
    '0.01': 0,
    '0.1': 0,
    '1': 0,
    '10': 0,
    '100': 0,
}

# Font sizes
tick_fontsize = 20
label_fontsize = 20
title_fontsize = 24  # Increased title font size
legend_fontsize = 24

def get_dataset_results(data):
    if data == 'UCI-HAR':
        bn_acc = {
            # 'Naive': [98.77, 51.34, 32.00],
            'Offline': 94.94,
            '0.01': [99.42, 92.94, 85.83],
            '0.1': [99.34, 92.9, 83.53],
            '1': [99.45, 93.86, 87.15],
            '10': [99.41, 93.76, 88.2],
            '100': [99.39, 93.4, 88.41],
        }

        ln_acc = {
            # 'Naive': [99.49, 53.95, 36.44],
            'Offline': 92.31,
            '0.01': [99.65, 92.8, 83.38],
            '0.1': [99.05, 92.53, 84.43],
            '1': [99.15, 93.35, 87.47],
            '10': [99.01, 93.45, 87.75],
            '100': [99.18, 93.37, 87.38],
        }

        num_tasks = 3

    elif data == 'UWave':
        bn_acc = {
            # 'Naive': [98.99, 49.47, 34.04, 25.36],
            'Offline': 96.61,
            '0.01': [99.28, 95.59, 93.51, 91.67],
            '0.1': [99.26, 95.96, 93.74, 91.89],
            '1': [99.24, 95.73, 93.24, 91.68],
            '10': [99.24, 95.7, 93.1, 91.35],
            '100': [99.3, 93.84, 90.01, 88.21],
        }

        ln_acc = {
            # 'Naive': [98.43, 49.47, 32.36, 24.85],
            'Offline': 96.39,
            '0.01': [98.32, 96.02, 93.38, 92.05],
            '0.1': [98.43, 95.82, 93.12, 91.8],
            '1': [98.27, 95.95, 93.68, 92.12],
            '10': [98.7, 95.00, 92.68, 91.00],
            '100': [98.72, 93.97, 90.52, 88.12],
        }

        num_tasks = 4

    elif data == 'DSA':
        bn_acc = {
            # 'Naive': [1.000e+02, 5.000e+01, 3.353e+01, 2.502e+01, 2.187e+01, 1.704e+01],
            'Offline': 99.65,
            '0.01': [100.00, 99.46, 97.94, 96.9, 94.97, 93.67],
            '0.1': [100.00, 99.67, 99.03, 98.25, 97.1, 96.5],
            '1': [100.00, 99.58, 99.39, 98.83, 98.53, 98.26],
            '10': [100.00, 99.5, 99.03, 98.46, 98.28, 97.89],
            '100': [100.00, 99.04, 98.14, 97.23, 96.98, 96.54],
        }

        ln_acc = {
            # 'Naive': [100.00, 58.96, 37.92, 25.48, 23.02, 19.81],
            'Offline': 99.53,
            '0.01': [100.00, 98.67, 98.42, 97.44, 97.13, 96.49],
            '0.1': [100.00, 98.96, 99.03, 98.65, 98.45, 98.12],
            '1': [100.00, 98.96, 98.5, 98.29, 98.18, 98.03],
            '10': [100.00, 98.5, 97.58, 97.0, 96.83, 96.44],
            '100': [100.00, 97.29, 95.89, 94.54, 94.43, 93.89],
        }

        num_tasks = 6

    elif data == 'GRABMyo':
        bn_acc = {
            # 'Naive': [94.82, 48.24, 31.61, 23.7, 19.44],
            'Offline': 93.63,
            '0.01': [93.84, 56.23, 52.86, 51.78, 50.11],
            '0.1': [95.88, 70.17, 57.95, 51.64, 48.43],
            '1': [93.42, 75.72, 64.42, 58.57, 54.91],
            '10': [93.28, 78.51, 67.79, 61.4, 57.06],
            '100': [92.57, 75.36, 63.71, 57.23, 53.14],
        }

        ln_acc = {
            # 'Naive': [93.81, 47.78, 31.62, 23.15, 19.46],
            'Offline': 93.83,
            '0.01': [90.55, 48.6, 52.46, 51.78, 50.82],
            '0.1': [92.59, 61.46, 53.13, 50.75, 49.27],
            '1': [91.3, 71.53, 60.17, 55.19, 52.52],
            '10': [91.8, 78.13, 67.27, 60.59, 56.44],
            '100': [91.64, 76.94, 65.93, 59.14, 55.17],
        }

        num_tasks = 5

    elif data == 'WISDM':
        bn_acc = {
            # 'Naive': [89.53, 44.76, 30.54, 23.16, 19.12, 14.89],
            'Offline': 85.31,
            '0.01': [97.99, 93.77, 89.59, 85.0, 82.39, 79.57],
            '0.1': [98.14 ,95.51 ,92.89, 89.71, 87.55, 84.86],
            '1': [98.11 ,95.41, 92.12 ,89.21 ,87.82 ,85.35],
            '10': [97.44 ,94.04 ,89.38, 85.42 ,83.72 ,80.89],
            '100': [95.8  ,89.35 ,81.78 ,74.33 ,72.61 ,68.92],

            }

        ln_acc = {
            # 'Naive': [85.9,  47.41, 31.29, 23.07, 18.74, 14.60],
            'Offline': 88.6,
            '0.01': [95.3 , 90.23, 83.88 ,78.81, 75.73 ,73.75],
            '0.1': [96.72 ,94.05, 90.56 ,86.57 ,84.56, 82.46],
            '1': [97.05, 94.75 ,91.19 ,87.80 , 85.98, 83.53],
            '10': [96.75 ,92.89, 87.84 ,83.15, 81.45 ,78.45],
            '100': [94.41 ,88.31 ,81.65 ,75.45 ,73.11 ,69.06],
            }

        num_tasks = 6

    return bn_acc, ln_acc, num_tasks


def plot_avg_acc_curves(data_list, norm, separate=False):

    if not separate:
        num_dataset = len(data_list)
        fig, axes = plt.subplots(nrows=1, ncols=num_dataset, figsize=(30, 5), dpi=128)
        axes = axes.flatten()
        lines = []  # To keep track of line objects for the legend

        for i in range(num_dataset):
            bn_acc, ln_acc, num_tasks = get_dataset_results(data_list[i])
            acc_collections = bn_acc if norm == 'BN' else ln_acc
            for agent, acc in acc_collections.items():
                if agent == 'Offline':
                    line, = axes[i].plot(num_tasks, acc, marker='*', label='Offline', color='black', markersize=16)
                    # Add the line object to the list for legend handling
                    if i == 0:
                        lines.append(line)
                else:
                    line, = axes[i].plot(
                    list(range(1, 1+num_tasks)), 
                    acc, 
                    linewidth='3', 
                    label=r"$\gamma={}$".format(agent) if agent.replace('.', '', 1).isdigit() else agent, 
                    color=colors[agent],
                    linestyle=linestyles[agent], 
                    marker=markers[agent], 
                    markersize=markersize[agent]
                    )

                    if i == 0:
                        lines.append(line)

                axes[i].set_title('{}'.format(data_list[i]), fontsize=title_fontsize, pad=10)
                axes[i].set_xlabel('Task Number', fontsize=label_fontsize)
                axes[i].set_xticks(list(range(1, 1+num_tasks)))
                axes[i].set_xticklabels(labels=list(range(1, 1+num_tasks)), fontsize=tick_fontsize)
                if i == 0:  # Only for the first subplot
                    axes[i].set_ylabel('Avg Acc', fontsize=label_fontsize)
                axes[i].tick_params(axis='y', labelsize=tick_fontsize)
                if data_list[i] == 'UCI-HAR':
                    axes[i].set_ylim(10, 105)
                elif data_list[i] == 'UWave':
                    axes[i].set_ylim(10, 105)
                elif data_list[i] == 'DSA':
                    axes[i].set_ylim(10, 105)
                elif data_list[i] == 'GRABMyo':
                    axes[i].set_ylim(10, 105)
                elif data_list[i] == 'WISDM':
                    axes[i].set_ylim(10, 105)
                axes[i].grid(axis='y', color='gainsboro')

        # Place a legend above the subplots
        # Optional: Adjust the layout
        plt.tight_layout(rect=[0, 0, 1, 0.85])
        # fig.legend(agents, loc='upper center', ncol=len(agents), fontsize=legend_fontsize, frameon=False)

        # Create a custom legend
        # Extract all the other line objects except the one for 'Offline' for the legend
        legend_lines = [line for line in lines if line.get_label() != 'Offline']
        # Add a separate legend entry for 'Offline' with a single star
        legend_lines.append(
            plt.Line2D([0], [0], linestyle='none', marker='*', color='black', markersize=16, label='Offline'))
        # Create the legend with the custom entries. Place a legend above the subplots
        fig.legend(handles=legend_lines, fontsize=label_fontsize, loc='upper center', ncol=len(agents), frameon=False)

        plt.savefig("/data/yt/TSCIL/result/plosts/acc_evol_all_{}.png".format(norm), dpi=128, bbox_inches='tight')
        plt.show()

    else:
        num_dataset = len(data_list)
        for i in range(num_dataset):
            figure = plt.figure(figsize=(6, 6), dpi=128)
            bn_acc, ln_acc, num_tasks = get_dataset_results(data_list[i])
            acc_collections = bn_acc if norm == 'BN' else ln_acc
            for agent, acc in acc_collections.items():
                if agent == 'Offline':
                    plt.plot(num_tasks, acc, marker='*', label='Offline', color='black', markersize=10)
                else:
                    plt.plot(list(range(1, 1+num_tasks)), acc, linewidth='1.8', label=agent, color=colors[agent],
                                 linestyle=linestyles[agent], marker=markers[agent], markersize=markersize[agent])

            # plt.title("{}: BatchNorm".format(data), fontsize=18)
            plt.xlabel('Task Number', fontsize=24)
            plt.ylabel('Avg Acc', fontsize=22)
            plt.tick_params(axis='both', which='major', labelsize=12)  # Increase tick font sizes
            plt.grid(color='gainsboro')
            plt.xticks(list(range(1, 1+num_tasks)))
            if i == 0:
                plt.legend(fontsize=14)
            plt.savefig("../result/plots/acc_evol_{}_{}.png".format(data_list[i], norm), dpi=128, bbox_inches='tight')
            plt.show()


if __name__ == '__main__':
    # ['UCI-HAR', 'UWave', 'DSA', 'GRABMyo', 'WISDM']
    plot_avg_acc_curves(['UCI-HAR', 'UWave', 'DSA', 'GRABMyo', 'WISDM'], norm='BN', separate=False)
    plot_avg_acc_curves(['UCI-HAR', 'UWave', 'DSA', 'GRABMyo', 'WISDM'], norm='LN', separate=False)