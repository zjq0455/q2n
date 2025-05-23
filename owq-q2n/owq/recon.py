import math
import time

import torch
import torch.nn as nn
import transformers

from .quant import *

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

def fast_find_energy_drop_index( S, threshold=0.05):
    S = S.flatten()[1:]
    S = torch.clamp(S, min=0)

    prefix_sum = torch.cumsum(S, dim=0)         
    total_sum = prefix_sum[-1]
    suffix_sum = total_sum - prefix_sum         

    valid = prefix_sum[:-1] > 1e-12
    ratio = suffix_sum[1:] / prefix_sum[:-1]    # ratio[i] = suffix[i+1] / prefix[i]
    mask = (ratio <= threshold) & valid

    drop_index = None
    if torch.any(mask):
        i = torch.nonzero(mask)[0].item() + 1  
        drop_index = i + 1

    return drop_index

class GPTQ_OWQ:
    def __init__(self, layer, n_out):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = layer.weight.data.clone()

        if isinstance(self.layer, nn.Conv2d): 
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        
        self.rows = W.shape[0]
        self.columns = W.shape[1] 
        self.H = torch.zeros((self.columns, self.columns), device=self.dev) 
        self.nsamples = 0 

        self.n_out = n_out
        self.n_nonout = W.shape[1] - n_out
        self.owq = n_out > 0
        self.out_quantizer = None
        self.ids = None
        
        self.average_x = torch.zeros(2048,self.columns, device=self.dev)
        self.p = torch.zeros((self.columns, self.columns), device=self.dev)
    
    def add_batch(self, inp, out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0] 
        if isinstance(self.layer, nn.Linear) or isinstance(self.layer, transformers.Conv1D):
            if len(inp.shape) == 3:
                inp = inp.reshape((-1, inp.shape[-1]))
            inp = inp.t() 
        if isinstance(self.layer, nn.Conv2d):
            unfold = nn.Unfold(
                self.layer.kernel_size,
                dilation=self.layer.dilation,
                padding=self.layer.padding,
                stride=self.layer.stride
            )
            inp = unfold(inp)
            inp = inp.permute([1, 0, 2])
            inp = inp.flatten(1)
        
        self.H *= self.nsamples / (self.nsamples + tmp)

        self.average_x *= self.nsamples / (self.nsamples + tmp)

        self.nsamples += tmp
        # inp = inp.float()

        self.average_x += (inp.t().float() / self.nsamples)    

        inp = math.sqrt(2 / self.nsamples) * inp.float()
        # self.H += 2 / self.nsamples * inp.matmul(inp.t())
        self.H += inp.matmul(inp.t())     

        
        
    def hessian_sorting(self, actorder=False, frob_norm=None):
        H = self.H

        if not self.owq:
            if actorder:
                self.ids = torch.argsort(torch.diag(H), descending=True)
            return torch.tensor([])
        
        temp_mask = torch.full([self.columns], True, device=self.dev)
        
        H_diag = torch.diag(H)
        if frob_norm is not None:
            H_diag *= frob_norm
        descending_ids = torch.argsort(H_diag, descending=True)
        
        temp_mask[descending_ids[:self.n_out]] = False
        if actorder:
            ids = torch.cat([descending_ids[self.n_out:],descending_ids[:self.n_out]])
        else:
            ids = torch.cat([torch.arange(self.columns, device=self.dev)[temp_mask], descending_ids[:self.n_out]])
        
        self.ids = ids
        return torch.sort(descending_ids[:self.n_out])[0].to(torch.int32)

    
    def fasterquant(
        self, threshold, lambda_t, index, name, blocksize=128, percdamp=.01, groupsize=-1, actorder=False
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()
        
        tick = time.time()
        
        if actorder or self.owq:
            W = W[:, self.ids]
            self.H = self.H[self.ids][:,self.ids]
        
        self.quantizer.find_params(W[:,:self.n_nonout], weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0 

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        
        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp 
        
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H
        
        for i1 in range(0, self.n_nonout, blocksize):
            i2 = min(i1 + blocksize, self.n_nonout)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                w = W1[:, i]
                d = Hinv1[i, i]

                if groupsize != -1:
                    if (i1 + i) % groupsize == 0:
                        self.quantizer.find_params(W[:, (i1 + i):min((i1 + i + groupsize),(self.columns-self.n_out))], weight=True, num=40)

                q = self.quantizer.quantize(w.unsqueeze(1)).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d       
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        torch.cuda.synchronize()
        print('time %.2f' % (time.time() - tick))
        print('error', torch.sum(Losses).item())

        if name != 'mlp.down_proj':
            X = torch.mm(self.average_x.t(), self.average_x) #/ 2048

            tick = time.time()
            eigvals, eigvecs = torch.linalg.eigh(X)
            sorted_indices = torch.argsort(eigvals, descending=True)
            s = eigvals[sorted_indices]
            u = eigvecs[:, sorted_indices]
            print(time.time()-tick)
            index = fast_find_energy_drop_index(s, threshold=threshold) # 0.05
            # ************

            u_zero = u[:,index:] # rank_X  idx 
            p = u_zero @ u_zero.T

            with torch.enable_grad():
                lambda_reg = lambda_t   #0.3
                self.p = p.detach()

                I = torch.eye(Q.shape[1], device=self.dev)
                B = W @ (I - self.p) + Q @ self.p
                numerator = torch.sum(B * Q, dim=1) + lambda_reg            # (1024,)
                denominator = torch.sum(Q ** 2, dim=1) + lambda_reg           # (1024,)


                m = numerator / (denominator + 1e-8)
                Q = m.unsqueeze(1) * Q  
               
        if actorder or self.owq:
            Q[:,self.n_nonout:] = W[:,self.n_nonout:]
            invids = torch.argsort(self.ids)
            Q = Q[:, invids]
        
        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)


    def free(self):
        self.H = None
        self.Losses = None
        self.ids = None
        torch.cuda.empty_cache()