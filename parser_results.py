# from itertools import count
# import numpy as np

# f2 = open('results1.txt', 'r')

# all_values = {0:[np.zeros((3, 5)) for x in range(3)], 1:[np.zeros((3, 5)) for x in range(3)], 2:[np.zeros((3, 5)) for x in range(3)]}
# stress_index = {'arr_rate':0, 'dept_time':1, 'vnr_size':2}
# algo_index = {'dqn':0, 'greedy':1, 'a3c':2}
# metric_index = {'accept_ratio':0, 'rev_cost':1, 'cum_rev':2}
# arr_value_index = [str(0.04 * i) for i in range(1, 6)]
# dept_time = [str(400 + (i * 100)) for i in range(1,6)]
# vnr_size = [20 + (i * 10) for i in range(5)]

# i = 1
# j = 0

# for line in f2:
#     line = line.strip()
#     line_split = line.split(':')
#     if i == 1:
#         test = stress_index[line_split[1]]
#     if i == 2:
#         algo = algo_index[line_split[1]]
#     if i == 6 or i == 7 or i == 8:
#         metric = metric_index[line_split[0]]
#         if i == 6:
#             all_values[metric]

#     if i == 3:
#         break

#     i += 1

import numpy as np
import matplotlib.pyplot as plt


f2 = open('results1.txt', 'r')
all_values = {0:[np.zeros((3, 5)) for x in range(3)], 1:[np.zeros((3, 5)) for x in range(3)], 2:[np.zeros((3, 5)) for x in range(3)]}
stress_index = {'arr_rate':0, 'dept_time':1, 'vnr_size':2}
metric_index = {'accept_ratio':0, 'rev_cost':1, 'cum_rev':2}
algo_index = {'dqn':0, 'greedy':1, 'a3c':2}

i = 1
k = 0
count_mteric = 0
for line in f2:
    if k == 5:
        k = 0
    if i == 9:
        count_mteric += 1
        i = 1
        if count_mteric == 3:
            k += 1
            count_mteric = 0
        continue
    line = line.strip()
    line_split = line.split(':')
    if i == 1:
        test = stress_index[line_split[1]]
    if i == 2:
        algo = algo_index[line_split[1]]
    if i == 6 or i == 7 or i == 8:
        metric = metric_index[line_split[0]]
        all_values[int(test)][int(metric)][int(algo)][k] = float(line_split[1])
        # print(int(test), int(metric), int(algo), k)
    
    i += 1

x_arr = [0.04, 0.08, 0.12, 0.16, 0.2]
x_dep_t = [500, 600, 700, 800, 900]
x_vnr_size = [20, 30, 40, 50, 60]

x_vals = [x_arr, x_dep_t, x_vnr_size]
x_fields = ['arrival_rate', 'max_departure_time', 'max_vnr_resource_requirements']
y_fields = ['acceptance_rate', 'revenue_to_cost_ratio', 'cumulative_revenue']
titles = ['arrival_rate_change', 'departure_time_change', 'vnr_resource_requirement_change']
algos = ['dqn', 'greedy', 'a3c']
colors = ['b', 'r', 'g']
markers = ['+', 'D', 'h']

i = 0
for evaluations in all_values.values():
    j = 0
    for metric in evaluations:
        # plot the results
        plt.title(titles[i])
        plt.xlabel(x_fields[i])
        plt.ylabel(y_fields[j])
        for algo_index in range(len(metric)):
            plt.plot(x_vals[i], metric[algo_index], label=algos[algo_index], marker=markers[algo_index], color=colors[algo_index])
        plt.legend()
        name = str(x_fields[i]) + '_' + str(y_fields[j])
        plt.savefig('results/' + name + '_.png')
        plt.clf()
        j += 1
    i += 1