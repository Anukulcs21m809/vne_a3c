import numpy as np
import matplotlib.pyplot as plt
import os

n_algos = 4
n_tests = 3

all_values = {0:[np.zeros((n_algos, 5)) for x in range(n_tests)], 1:[np.zeros((n_algos, 5)) for x in range(n_tests)], 2:[np.zeros((n_algos, 5)) for x in range(n_tests)]}
stress_index = {'arr_rate':0, 'dept_time':1, 'vnr_size':2}
metric_index = {'accept_ratio':0, 'rev_cost':1, 'cum_rev':2}
algo_index = {'a3c':0, 'dqn':1, 'greedy':2, 'heuristic':3}

directory = 'result_text_files'

for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    if f.split('/')[1].split('_')[0] == 'results':
        fl = open(f, 'r')
        i = 1
        k = 0
        count_metric = 0
        for line in fl:
            if k == 5:
                k = 0
            if i == 9:
                i = 1
                k += 1
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
        fl.close()

# print(all_values)

x_arr = [0.04, 0.08, 0.12, 0.16, 0.2]
x_dep_t = [500, 600, 700, 800, 900]
x_vnr_size = [4, 6, 8, 10, 12]

x_vals = [x_arr, x_dep_t, x_vnr_size]
x_fields = ['arrival_rate', 'max_departure_time', 'max_vnr_resource_requirements']
y_fields = ['acceptance_rate', 'revenue_to_cost_ratio', 'cumulative_revenue']
titles = ['arrival_rate_change', 'departure_time_change', 'vnr_resource_requirement_change']
algos = ['a3c', 'dqn', 'greedy', 'heuristic']
colors = ['b', 'r', 'g', 'm']
markers = ['+', 'D', 'h', 'x']

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
        plt.savefig('results/acc_rate_rev_cost_rew/' + name + '_.png')
        plt.clf()
        j += 1
    i += 1