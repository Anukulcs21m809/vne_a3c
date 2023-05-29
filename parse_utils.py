import numpy as np
import os
import matplotlib.pyplot as plt

directory = 'result_text_files'
n_algos = 4
utils = [[] for x in range(4)]

k = 0
for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if f.split('/')[1].split('_')[0] == 'util':
        fl = open(f, 'r')
        i = 1
        for line in fl:
            if i == 7:
                i = 1
                continue
            line = line.strip()
            line_split = line.split(':')
            if i == 6:
                util_ = eval(line_split[1])  
                utils[k].append(util_)
            i += 1
        k += 1
        fl.close()

titles = ['arrival_rate_change', 'departure_time_change', 'vnr_resource_requirement_change']
x_fields = ['arrival_rate', 'max_departure_time', 'max_vnr_resource_requirements']
x_arr = [0.04, 0.08, 0.12, 0.16, 0.2]
x_dep_t = [500, 600, 700, 800, 900]
x_vnr_size = [4, 6, 8, 10, 12]
algos = ['a3c', 'dqn', 'greedy', 'heuristic']
colors = ['b', 'r', 'g', 'c']
markers = ['+', 'D', 'h', 'x']

values = [x_arr, x_dep_t, x_vnr_size]

g = 0
for x in range(len(x_fields)):
    for y in range(len(x_arr)):
        plt.title(x_fields[x] + '= {}'.format(values[x][y]))
        plt.xlabel('steps')
        plt.ylabel('average_node_util')
        x_labels = [i+1 for i in range(len(utils[0][0]))]
        for z in range(len(utils)):
            l = 1
            avg = 0
            avgs = []
            for k in utils[z][g]:
                avg += k
                if l == 10:
                    avgs.append(avg / l)
                    l = 1
                    avg = 0
                l += 1
            # plt.plot(x_labels[:len(utils[z][g])], utils[z][g], label=algos[z], color=colors[z])
            plt.plot(x_labels[:len(avgs)], avgs, label=algos[z], color=colors[z])
        name = str(x_fields[x]) + '_' + str(values[x][y])
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('results/utilizations/' + name + '_.png')
        plt.clf()
        g += 1

             