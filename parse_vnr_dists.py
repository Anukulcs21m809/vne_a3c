import matplotlib.pyplot as plt

f = open('result_text_files/vnr_dist_a3c.txt', 'r')

x_fields = ['arrival_rate', 'max_departure_time', 'max_vnr_resource_requirements']
x_arr = [0.04, 0.08, 0.12, 0.16, 0.2]
x_dep_t = [500, 600, 700, 800, 900]
x_vnr_size = [4, 6, 8, 10, 12]
colors = ['b', 'r', 'g']
values = [x_arr, x_dep_t, x_vnr_size]
sub_names = ['substrate_{}'.format(i+1) for i in range(3)]


i = 1
dists = []
for line in f:
    if i == 6:
        i = 1
        continue
    if i == 5:
        line_ = line.split(':')
        dist = eval(line_[1])
        dists.append(dist)
    i +=1

g = 0
for x in range(len(x_fields)):
    for y in range(len(x_arr)):
        plt.title(x_fields[x] + '= {}'.format(values[x][y]))
        plt.xlabel('vnr distribution over substrate graphs')
        plt.ylabel('number of vnrs embedded')
        plt.bar(sub_names, dists[g], color=colors[x], label='a3c')
        plt.legend()
        name = str(x_fields[x]) + '_' + str(values[x][y]) + '_vnr_dist'
        plt.savefig('results/vnr_dists/' + name + '_.png')
        plt.clf()
        g += 1
