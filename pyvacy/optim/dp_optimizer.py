import torch
from torch.optim import Optimizer
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.distributions.normal import Normal
from torch.optim import SGD, Adam, Adagrad, RMSprop

def make_optimizer_class(cls):
    class DPOptimizerClass(cls):
        def __init__(self, l2_norm_clip, noise_multiplier, minibatch_size, microbatch_size, *args, **kwargs):
            super(DPOptimizerClass, self).__init__(*args, **kwargs)

            self.l2_norm_clip = l2_norm_clip
            self.noise_multiplier = noise_multiplier
            self.microbatch_size = microbatch_size
            self.minibatch_size = minibatch_size

            self.ex = torch.zeros_like(self.param_groups[0]['params'][0].data) 
            for group in self.param_groups:
                group['accum_grads'] = [torch.zeros_like(param.data) if param.requires_grad else None for param in group['params']]

        def zero_microbatch_grad(self):
            super(DPOptimizerClass, self).zero_grad()

        def microbatch_step(self):
            total_norm = 0.
            i = 0
            for group in self.param_groups:
                for param in filter(lambda p: p.grad is not None, group['params']):
                    if param.requires_grad:
                        total_norm += param.grad.data.norm(2).item() ** 2.
            total_norm = total_norm ** .5      
            clip_coef = min(self.l2_norm_clip / (total_norm + 1e-6), 1.)
            # clip_coef = 1.
            # print(total_norm, clip_coef)
            for group in self.param_groups:
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad and param.grad is not None and accum_grad is not None:
                        # print(group['accum_grads'])
                        accum_grad.add_(param.grad.data.mul(clip_coef))

        def zero_grad(self):
            for group in self.param_groups:
                for accum_grad in group['accum_grads']:
                    if accum_grad is not None:
                        accum_grad.zero_()

        def step(self, *args, **kwargs):
            i = 0
            j = 0
            k = 0
            for group in self.param_groups:
                # print(len(group))
                for param, accum_grad in zip(group['params'], group['accum_grads']):
                    if param.requires_grad and param.grad is not None and accum_grad is not None:
                        # if(i == 0):
                        #     # print(accum_grad.clone())
                        #     print(accum_grad.clone().add(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data)).mul(self.microbatch_size / self.minibatch_size).div(accum_grad.clone()))    
                        #     print()
                        # i+=param.numel()                                  
                        param.grad.data = accum_grad.clone()
                        param.grad.data.add_(self.l2_norm_clip * self.noise_multiplier * torch.randn_like(param.grad.data))
                        param.grad.data.mul_(self.microbatch_size / self.minibatch_size)
            # print(i, j, k)
            
            super(DPOptimizerClass, self).step(*args, **kwargs)


    return DPOptimizerClass

DPAdam = make_optimizer_class(Adam)
DPAdagrad = make_optimizer_class(Adagrad)
DPSGD = make_optimizer_class(SGD)
DPRMSprop = make_optimizer_class(RMSprop)

