import os
import math
import time

import numpy as np
from sklearn.cluster import KMeans
from multiprocessing import Pool
from tqdm import tqdm

import torch
import torch.nn as nn
import transformers

from quant import *


DEBUG = False 

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

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

    return drop_index

def kmeans_fit(row_data):
    weights_np, sample_weight, n_cluster, random_seed = row_data
    kmeans = KMeans(
        n_clusters=n_cluster,
        init=np.linspace(weights_np.min(), weights_np.max(), num=n_cluster)[:, None] if n_cluster <= 8 else 'k-means++',
        n_init='auto',
        random_state=random_seed,
        max_iter=100,
        tol=1e-6,
    ).fit(weights_np, sample_weight=sample_weight)
    return kmeans.cluster_centers_.reshape(-1)

pool = Pool(len(os.sched_getaffinity(0)))

class LeanQuant:

    def __init__(self, layer):
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
        self.lut = None
        self.average_x = torch.zeros(2048,self.columns, device=self.dev)
        self.p = torch.zeros((self.columns, self.columns), device=self.dev)

    def add_batch(self, inp, out):
        if DEBUG:
            self.inp1 = inp
            self.out1 = out
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

    def fasterquant(
        self, threshold, lambda_t, index, name, blocksize=128, percdamp=.01, groupsize=-1, actorder=False, static_groups=False, args=None,
    ):
        W = self.layer.weight.data.clone()
        if isinstance(self.layer, nn.Conv2d):
            W = W.flatten(1)
        if isinstance(self.layer, transformers.Conv1D):
            W = W.t()
        W = W.float()

        tick = time.time()

        if not self.quantizer.ready():
            self.quantizer.find_params(W, weight=True)

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if static_groups:
            import copy
            groups = []
            for i in range(0, self.columns, groupsize):
                quantizer = copy.deepcopy(self.quantizer)
                quantizer.find_params(W[:, i:(i + groupsize)], weight=True)
                groups.append(quantizer)

        if H.shape[0] >= args.offload_threshold:
            secondary_device = torch.device('cuda:1')
            H = H.to(secondary_device)

        if actorder:
            perm_H = torch.argsort(torch.diag(H), descending=True)
            perm = perm_H.to(W.device)
            W = W[:, perm]
            H = H[perm_H][:, perm_H]
            invperm = torch.argsort(perm)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=H.device)
        H[diag, diag] += damp

        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)

        if H.shape[0] >= args.offload_threshold:
            H = H.to(self.dev)

        Losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)
        Q_codes = Q.to(torch.uint8).cpu()
        Hinv = H
        torch.cuda.empty_cache()

        if isinstance(args.exponent, float):
            kmeans_tasks = []
            W_np = W.cpu().numpy()
            Hinv_diagonal_np = (torch.diagonal(Hinv) ** (-args.exponent)).cpu().numpy()
            for j in range(W_np.shape[0]):
                kmeans_tasks.append((W_np[j, :, None], Hinv_diagonal_np, 2 ** args.wbits, args.kmeans_seed))
            kmeans_results = list(tqdm(pool.imap(kmeans_fit, kmeans_tasks), total=len(kmeans_tasks)))
            centroids = torch.from_numpy(np.stack(kmeans_results)).reshape(W.shape[0], 2 ** args.wbits).to(W.device)
        else:
            centroids = None

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
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
                    if not static_groups:
                        if (i1 + i) % groupsize == 0:
                            self.quantizer.find_params(W[:, (i1 + i):(i1 + i + groupsize)], weight=True)
                    else:
                        idx = i1 + i
                        if actorder:
                            idx = perm[idx]
                        self.quantizer = groups[idx // groupsize]

                if isinstance(centroids, torch.Tensor):
                    codes = torch.argmin((centroids - w[:, None]).abs(), dim=1, keepdim=True)
                    Q_codes[:, i1+i] = codes.flatten().to(torch.uint8).cpu()
                    q = torch.gather(centroids, 1, codes).flatten()
                else:
                    q = quantize(
                        w.unsqueeze(1), self.quantizer.scale, self.quantizer.zero, self.quantizer.maxq
                    ).flatten()
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d ** 2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            Losses[:, i1:i2] = Losses1 / 2

            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

            if DEBUG:
                self.layer.weight.data[:, :i2] = Q[:, :i2]
                self.layer.weight.data[:, i2:] = W[:, i2:]
                print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))
                print(torch.sum(Losses))

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
                lambda_reg = lambda_t
                self.p = p.detach()

                I = torch.eye(Q.shape[1], device=self.dev)
                B = W @ (I - self.p) + Q @ self.p
                numerator = torch.sum(B * Q, dim=1) + lambda_reg            # (1024,)
                denominator = torch.sum(Q ** 2, dim=1) + lambda_reg           # (1024,)

                m = numerator / (denominator + 1e-8)
                Q = m.unsqueeze(1) * Q

        if actorder:
            Q = Q[:, invperm]
            Q_codes = Q_codes[:, invperm.cpu()]

        if isinstance(args.save_path, str) and isinstance(centroids, torch.Tensor):
            nrows, ncols = Q_codes.shape
            idx = torch.arange(0, ncols, 2)[None, :].repeat(nrows, 1).to(Q_codes.device)
            self.quantized_codes = torch.bitwise_or(torch.bitwise_left_shift(Q_codes.gather(1, idx), 4), Q_codes.gather(1, idx+1))
            self.quant_grid = centroids.cpu()

        if isinstance(self.layer, transformers.Conv1D):
            Q = Q.t()
        print('norm of difference', torch.norm(self.layer.weight.data - Q).item())
        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        if DEBUG:
            print(torch.sum((self.layer(self.inp1) - self.out1) ** 2))

    def free(self):
        if DEBUG:
            self.inp1 = None
            self.out1 = None
        self.H = None
        self.Losses = None
        self.Trace = None
        torch.cuda.empty_cache()
