import time
import torch

import torch
import torch.nn as nn
import transformers

#from gptq import GPTQ
from method import QuantMethod
from vector_balance import quantize_weight_vecbal 

def fast_find_energy_drop_index(S, threshold=0.05):
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

        if 1 <= drop_index < len(S) - 1:
            before_ratio = S[drop_index].item() / (S[drop_index - 1].item() + 1e-12)
            after_ratio = S[drop_index + 1].item() / (S[drop_index].item()  + 1e-12)


class Balance(QuantMethod):

    def configure(self, qmethod, nbits, npasses, unbiased):
        self.qmethod = qmethod
        self.nbits = nbits
        self.npasses = npasses
        self.unbiased = unbiased

    def fasterquant(self, threshold, lambda_t, name, lazy_batch=False):
        w = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            raise NotImplementedError()
        if isinstance(self.layer, transformers.Conv1D):
            raise NotImplementedError()
        tick = time.time()
        if not self.quantizer.ready():
            self.quantizer.find_params(w, weight=True)
        H = self.H.data.clone()

        quant_w = quantize_weight_vecbal(
            w=w, H=H,
            nbits=self.nbits,
            npasses=self.npasses,
            scale=self.quantizer.scale,
            zero=self.quantizer.zero,
            maxq=self.quantizer.maxq,
            unbiased=self.unbiased,
            qfn=self.quantizer.qfn,
            qmethod=self.qmethod,
            lazy_batch=lazy_batch
        )

        if name != 'mlp.down_proj':
            X = torch.mm(self.average_x.t(), self.average_x) #/ 2048

            tick = time.time()
            eigvals, eigvecs = torch.linalg.eigh(X)
            sorted_indices = torch.argsort(eigvals, descending=True)
            s = eigvals[sorted_indices]
            u = eigvecs[:, sorted_indices]
            print(time.time()-tick)
            index = fast_find_energy_drop_index(s, threshold=threshold) # 0.05   0.1
            # ************

            u_zero = u[:,index:] # rank_X  idx 
            p = u_zero @ u_zero.T
 
            with torch.enable_grad():
                lambda_reg = lambda_t
                self.p = p.detach().half()
                I = torch.eye(quant_w.shape[1], device=self.dev)
                B = w @ (I.half() - self.p) + quant_w @ self.p
                numerator = torch.sum(B * quant_w, dim=1) + lambda_reg            # (1024,)
                denominator = torch.sum(quant_w ** 2, dim=1) + lambda_reg           # (1024,)

                m = numerator / (denominator + 1e-8)
                quant_w = m.unsqueeze(1) * quant_w 

        self.layer.weight.data = quant_w
        self.postproc()
        # print('time %.2f' % (time.time() - tick))
        self.time = time.time() - tick
        self.error_compute(w, quant_w)