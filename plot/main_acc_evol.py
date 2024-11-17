import numpy as np
import matplotlib.pyplot as plt

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


agents = ['Naive', 'Offline', 'LwF', 'MAS', 'DT2W', 'GR','ER', 'DER', 'Herding', 'ASER', 'CLOPS', 'FastICARL','TS-ACL']

colors = {'Naive': '#F27970',
          'Offline': 'black',
          'LwF': '#54B345',
          'MAS': '#32B897',
          'DT2W': '#05B9E2',
          'GR': '#8983BF',
          'ER': '#BB9727',
          'DER': '#C76DA2',
          'Herding': '#A1A9D0',
          'ASER': '#F0988C',
          'CLOPS': '#96CCCB',
          'FastICARL': '#F6CAE5',
          'TS-ACL': '#FFC107'
          }

linestyles = {'Naive': '-',
              'Offline': '-',
              'LwF': '-',
              'MAS': '-',
              'DT2W': '-',
              'GR': '-',
              'ER': '-',
              'DER': '-',
              'Herding': '-',
              'ASER': '-',
              'CLOPS': '-',
              'FastICARL': '-',
              'TS-ACL': '-'
              }

markers = {'Naive': 'o',
           'Offline': 'o',
           'LwF': 'o',
           'MAS': 'o',
           'DT2W': 'o',
           'GR': '^',
           'ER': '^',
           'DER': '^',
           'Herding': '^',
           'ASER': '^',
           'CLOPS': '^',
           'FastICARL': '^',
           'TS-ACL': 's'
           }

markersize = {'Naive': 8,
              'Offline': 8,
              'LwF': 8,
              'MAS': 8,
              'DT2W': 8,
                'GR': 8,
              'ER': 10,
              'DER': 10,
              'Herding': 10,
              'ASER': 10,
              'CLOPS': 10,
              'FastICARL':10,
              'TS-ACL':10}


# Font sizes
tick_fontsize = 20
label_fontsize = 20
title_fontsize = 24  # Increased title font size
legend_fontsize = 24


def get_dataset_results(data):
    if data == 'UCI-HAR':
            bn_acc = {
                'Naive': [98.77, 51.34, 32.00],
                'Offline': 94.94,
                'LwF': [99.06, 56.18, 35.96],
                'MAS': [98.77, 52.17, 52.34],
                'DT2W': [98.61, 71.86, 53.23],
                'GR': [99.18, 80.67 ,66.66],
                'ER': [99.38, 66.8, 65.46],
                'DER': [99.52, 92.18, 74.41],
                'Herding': [99.38, 75.91, 69.58],
                'ASER': [99.67, 98.99, 92.36],
                'CLOPS': [98.89, 74.61, 72.87],
                'FastICARL': [99.91, 94.89, 79.69],
                'TS-ACL':[99.39, 93.4, 88.41]

            }

            ln_acc = {
                'Naive': [99.49, 53.95, 36.44],
                'Offline': 92.31,
                'LwF': [99.21, 61.44, 47.4],
                'MAS': [99.26, 73.76, 59.53],
                'DT2W': [99.39, 90.4,  80.15],
                'GR': [98.66 ,93.1 , 80.04],
                'ER': [99.59, 97.58, 89.53],
                'DER': [99.74, 97.92, 90.75],
                'Herding': [99.72, 98.29, 89.95],
                'ASER': [99.59, 98.66, 89.82],
                'CLOPS': [98.32, 96.49, 89.64],
                'FastICARL': [99.23, 96.26, 85.43],
                'TS-ACL':[99.01, 93.45, 87.75]

            }

            num_tasks = 3

    elif data == 'UWave':
            bn_acc = {
                'Naive': [98.99, 49.47, 34.04, 25.36],
                'Offline': 96.61,
                'LwF': [98.76, 75.01, 55.47, 44.67],
                'MAS': [88.71, 76.77, 64.89, 53.8],
                'DT2W': [99.01, 85.48, 75.92, 64.44],
                'GR': [99.35 ,88.62 ,81.65, 76.2],
                'ER': [98.56, 88.65, 84.33, 70.28],
                'DER': [99.05, 89.9,  82.51, 70.88],
                'Herding': [99.01, 91.00, 85.7,  78.47],
                'ASER': [99.35, 96.03, 90.02, 82.74],
                'CLOPS': [98.85, 87.74, 79.31, 71.04],
                'FastICARL': [99.1, 91.77, 86.91, 67.77],
                'TS-ACL':[99.26, 95.96, 93.74, 91.89]

            }

            ln_acc = {
                'Naive': [98.43, 49.47, 32.36, 24.85],
                'Offline': 96.39,
                'LwF': [98.65, 59.47, 34.87, 29.09],
                'MAS': [98.43, 74.23, 50.11, 40.74],
                'DT2W': [98.43, 83.12, 69.5, 55.09],
                'GR': [98.54, 90.5  ,87.42, 85.77],
                'ER': [98.95, 95.07, 87.43, 78.89],
                'DER': [98.36, 88.76, 83.21, 77.74],
                'Herding': [98.77, 95.00,   89.31, 85.42],
                'ASER': [98.9, 91.89, 83.32, 77.89],
                'CLOPS': [98.5,  89.18, 80.35, 73.79],
                'FastICARL': [98.34, 92.23, 87.07, 79.01],
                'TS-ACL':[98.27, 95.95, 93.68, 92.12]

            }

            num_tasks = 4

    elif data == 'DSA':
            bn_acc = {
                'Naive': [1.000e+02, 5.000e+01, 3.353e+01, 2.502e+01, 2.187e+01, 1.704e+01],
                'Offline': 99.65,
                'LwF': [100.00, 51.62,  39.72,  25.62,  23.22, 24.82],
                'MAS': [100.00,  75.12,  46.11,  38.40,  34.97,  31.82],
                'DT2W': [100.00, 60.46, 46.94, 31.10,  29.10, 19.56],
                'GR': [100.00, 65.63, 51.61,43.67,35.8 ,31.51],
                'ER': [100.00, 78.38,  68.53,  67.15,  80.37,  79.75],
                'DER': [100.00,  77.38,  65.69, 52.52,  66.12,  59.19],
                'Herding': [100.00, 78.79,  65.78,  68.6,   81.8,  82.43],
                'ASER': [100.00,  99.63,  99.25,  97.81,  98.18,  97.26],
                'CLOPS': [100.00,  82.33,  69.33,  64.96,  73.38,  74.1],
                'FastICARL': [100.00,   92.67,  91.22,  88.73,  91.03, 67.28],
                'TS-ACL':[100.00, 99.58,  99.39,  98.83,  98.53,  98.26]
            }

            ln_acc = {
                'Naive': [100.00, 58.96, 37.92, 25.48,  23.02, 19.81],
                'Offline': 99.53,
                'LwF': [100.00,  55.25,  35.61,  24.42,  18.72, 17.01],
                'MAS': [100.00, 72.50,  54.06, 42.69, 45.07, 35.75],
                'DT2W': [100.00, 60.46, 34.31, 27.71, 22.95, 19.06],
                'GR': [99.92 ,87.92, 85.53, 74.46, 77.23, 69.5],
                'ER': [100.00, 99.87,  99.72,  97.44,  98.52,  97.24],
                'DER': [100.00, 99.83,  99.75,  98.73,  98.90,  98.01],
                'Herding': [100.00,  99.75,  99.64,  97.54, 98.5,  97.75],
                'ASER': [100.00, 99.92,  99.06,  98.29,  98.73,  95.97],
                'CLOPS': [100.00,  97.96,  96.17,  92.67,  91.35,  89.65],
                'FastICARL': [100.00,  98.75,  97.28, 93.31,  93.1, 91.39],
                'TS-ACL':[100.00,  98.96,  99.03,  98.65,  98.45, 98.12]

            }

            num_tasks = 6

    elif data == 'GRABMyo':
            bn_acc = {
                'Naive': [94.82, 48.24, 31.61, 23.7,  19.44],
                'Offline': 93.63,
                'LwF': [89.73, 45.07, 29.53, 21.57, 19.22],
                'MAS': [90.67, 43.17, 28.24, 21.36, 19.04],
                'DT2W': [94.94, 48.37, 35.68, 27.96, 21.34],
                'GR': [94.86, 50.57 ,31.04, 23.77, 20.59],
                'ER': [95.99, 61.58, 60.03, 52.01, 47.03],
                'DER': [96.55, 74.05, 57.13, 41.66, 31.38],
                'Herding': [95.98, 64.65, 58.32, 51.7, 47.14],
                'ASER': [95.62, 83.78, 71.41, 62.38, 56.5],
                'CLOPS': [95.39, 56.3, 54.53, 46.92, 43.75],
                'FastICARL': [95.65, 75.71, 65.3,  57.6,  40.55],
                'TS-ACL':[93.28, 78.51, 67.79, 61.4, 57.06]
            }

            ln_acc = {
                'Naive': [93.81, 47.78, 31.62, 23.15, 19.46],
                'Offline': 93.83,
                'LwF': [90.89, 47.68, 31.91, 23.33, 19.42],
                'MAS': [94.15, 49.27, 31.7, 22.29, 18.15],
                'DT2W': [90.29, 54.17, 36.68, 26.17, 20.09],
                'GR': [95.35 ,50.19 ,33.73 ,25.08 ,20.56],
                'ER': [94.45, 88.29, 76.41, 67.29, 61.16],
                'DER': [94.78, 88.18, 78.93, 70.25, 63.78],
                'Herding': [94.66, 88.39, 77.16, 66.89, 60.07],
                'ASER': [95.08, 87.02, 74.17, 63.82, 57.9],
                'CLOPS': [94.73, 79.81, 65.51, 57.49, 52.05],
                'FastICARL': [95.03, 81.98, 68.79, 61.02, 52.84],
                'TS-ACL':[91.8, 78.13, 67.27, 60.59, 56.44]

            }

            num_tasks = 5

    elif data == 'WISDM':
            bn_acc = {
                'Naive': [89.53, 44.76, 30.54, 23.16, 19.12, 14.89],
                'Offline': 85.31,
                'LwF': [90.21, 48.36, 29.54, 21.34, 17.46, 10.74],
                'MAS': [96.63, 49.38, 30.22, 21.76, 17.58, 11.21],
                'DT2W': [93.92, 50.6,  32.54, 27.01, 21.52, 17.29],
                'GR': [94.72 ,45.15 ,36.36 ,25.24 ,20.29 ,15.44],
                'ER': [96.33, 56.86, 56.45, 47.89, 43.7,  41.69],
                'DER': [96.15, 64.18, 52.26, 40.06, 33.7,  28.99],
                'Herding': [97.54, 60.5,  53.78, 48.19, 43.64, 42.42],
                'ASER': [97.45, 87.03, 72.3,  60.08, 54.04, 48.36],
                'CLOPS': [94.61, 52.54, 48.54, 38.25, 37.00,  32.95],
                'FastICARL': [95.27, 65.71, 54.2,  44.53, 39.78, 32.72],
                'TS-ACL':[98.11, 95.41, 92.12, 89.21, 87.82, 85.35]
            }

            ln_acc = {
                'Naive': [85.9,  47.41, 31.29, 23.07, 18.74, 14.6],
                'Offline': 88.6,
                'LwF': [89.00, 45.16, 30.76, 23.12, 19.65, 13.83],
                'MAS': [90.05, 51.2,  34.27, 27.12, 23.62, 19.25],
                'DT2W': [85.44, 49.51, 31.33, 23.26, 19.19, 15.59],
                'GR': [95.95 ,64.49 ,45.38 ,35.87 ,33.55 ,24.33],
                'ER': [97.36, 94.8,  87.05, 78.31, 72.51, 66.88],
                'DER': [97.36, 93.49, 86.37, 75.9,  71.37, 67.14],
                'Herding': [97.36, 92.59, 87.32, 78.17, 73.35, 69.67],
                'ASER': [97.36, 86.27, 75.03, 62.49, 58.1,  51.48],
                'CLOPS': [94.55, 76.68, 63.16, 53.97, 48.69, 44.00],
                'FastICARL': [94.36, 76.54, 63.94, 51.65, 49.07, 44.87],
                'TS-ACL':[97.05, 94.75, 91.19, 87.8,  85.98, 83.53]
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
                    line, = axes[i].plot(list(range(1, 1+num_tasks)), acc, linewidth='1.8', label=agent, color=colors[agent],
                                         linestyle=linestyles[agent], marker=markers[agent], markersize=markersize[agent])
                    if i == 0:
                        lines.append(line)

                axes[i].set_title('{}'.format(data_list[i]), fontsize=title_fontsize, pad=10)
                axes[i].set_xlabel('Task Number', fontsize=label_fontsize)
                axes[i].set_xticks(list(range(1, 1+num_tasks)))
                axes[i].set_xticklabels(labels=list(range(1, 1+num_tasks)), fontsize=tick_fontsize)
                if i == 0:  # Only for the first subplot
                    axes[i].set_ylabel('Avg Acc', fontsize=label_fontsize)
                axes[i].tick_params(axis='y', labelsize=tick_fontsize)
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