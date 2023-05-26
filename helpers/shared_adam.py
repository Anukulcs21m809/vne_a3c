import torch

# class SharedAdam(torch.optim.Adam):
#     def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8,
#             weight_decay=0):
#         super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps,
#                 weight_decay=weight_decay)

#         for group in self.param_groups:
#             for p in group['params']:
#                 state = self.state[p]
#                 state['step'] = 0
#                 state['exp_avg'] = torch.zeros_like(p.data)
#                 state['exp_avg_sq'] = torch.zeros_like(p.data)

#                 state['exp_avg'].share_memory_()
#                 state['exp_avg_sq'].share_memory_()

import torch
import torch.optim as optim

class SharedAdam(optim.Adam):
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-08,
                 weight_decay=0, amsgrad=False):
        # params = params
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        
        # self.model = model

    def step(self, closure=None):
        # loss = None
        # if closure is not None:
        #     loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                # if p.grad is None:
                #     continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    state['exp_avg'].share_memory_()
                    state['exp_avg_sq'].share_memory_()
                
        #         exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
        #         beta1, beta2 = group['betas']
                
        #         state['step'] += 1
                
        #         # Decay the first and second moment running average coefficient
        #         exp_avg.mul_(beta1).add_(1 - beta1, grad)
        #         exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                
        #         # Bias correction
        #         bias_correction1 = 1 - beta1 ** state['step']
        #         bias_correction2 = 1 - beta2 ** state['step']
                
        #         # Update parameters
        #         step_size = group['lr'] * (bias_correction2.sqrt() / bias_correction1)
        #         p.data.addcdiv_(-step_size, exp_avg, exp_avg_sq.sqrt().add_(group['eps']))

        # return loss
