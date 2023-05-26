import numpy as np
import random

class Time_Sim:
    def __init__(self, mu=0.3, dep_t=600) -> None:
        self.MU = mu # mean arrival rate
        # self.TAU = tau
        self.arrival_time = 0
        self.dep_time = dep_t
    
    def ret_arr_time(self):
        u = np.random.uniform()
        self.arrival_time = self.arrival_time + (-np.log(1-u) / self.MU)
        arr_t = self.arrival_time#int(self.arrival_time * 10)
        return arr_t

    def ret_dep_time(self, arr_t):
        # u = np.random.uniform()
        # dep_time = self.arrival_time + (-np.log(1-u) / self.TAU)
        # dep_t = int(dep_time * 10)
        # return dep_t
        return arr_t + random.randint(400, self.dep_time)
    
    def reset(self):
        self.arrival_time = 0

# time_sim = Time_Sim(mu=0.15)
# for x in range(100):
#     print(time_sim.ret_arr_time())
#     print(time_sim.ret_dep_time())
#     print('\n')