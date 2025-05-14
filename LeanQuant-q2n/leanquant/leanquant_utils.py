import math
from typing import Optional, Union, Dict

import torch
from torch import nn
import cupy as cp

from transformers.modeling_utils import no_init_weights
from transformers import AutoConfig, AutoModelForCausalLM
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from safetensors import safe_open

kernel_code = '''typedef unsigned char uint8_t;
typedef unsigned short uint16_t;

extern "C"
__global__ void gather_sub4bit(
    const uint16_t* __restrict__ src,       // nrows x 2^bits
    const uint8_t* __restrict__ codes,      // nrows x (ncols / 2)
    uint16_t* __restrict__ dst,             // nrows x ncols
    int bits, int ncols
) {
    extern __shared__ volatile uint16_t cache[]; // 2^bits

    const int row_id = blockIdx.x;
    const int thread_id = threadIdx.x;
    const int n_threads = blockDim.x;

    const int n_floats = 1 << bits;

#pragma unroll
    for (int i = thread_id; i < n_floats; i += n_threads) {
        cache[i] = src[row_id * n_floats + i];
    }
    __syncthreads();

    for (int i = thread_id; i < ncols / 2; i += n_threads) {
        uint8_t code = codes[row_id * ncols / 2 + i];
        dst[row_id * ncols + i * 2] = cache[code >> 4];
        dst[row_id * ncols + i * 2 + 1] = cache[code & 0xf];
    }
}'''
_gather_sub4bit = cp.RawKernel(
    kernel_code,
    'gather_sub4bit'
)

class Sub4BitLinear(nn.Module):
    def __init__(self, orig_weight, bits=4, quant_grid=None, weight_codes=None, dtype=torch.float16):
        super().__init__()

        if isinstance(orig_weight, torch.Tensor):
            rows, cols = orig_weight.shape
            self.quant_grid = nn.Parameter(torch.empty(rows, 2 ** bits, dtype=orig_weight.dtype))
            weight_codes = torch.empty(rows, cols // 2, dtype=torch.uint8)
            self.register_buffer('weight_codes', weight_codes)
        elif isinstance(quant_grid, torch.Tensor) and isinstance(weight_codes, torch.Tensor):
            assert dtype == torch.float16 or dtype == torch.bfloat16
            assert weight_codes.dtype == torch.uint8

            quant_grid = quant_grid.squeeze()
            weight_codes = weight_codes.squeeze()

            self.quant_grid = nn.Parameter(quant_grid.to(dtype))
            self.register_buffer('weight_codes', weight_codes)

            assert self.quant_grid.shape[0] == self.weight_codes.shape[0]
            bits = int(math.log2(self.quant_grid.shape[1]))
            assert bits <= 4
            assert (2 ** bits) == self.quant_grid.shape[1]
        else:
            assert False, "This function can be initialized using either an `orig_weight` tensor or a pair of `quant_grid` and `weight_codes` tensors."

    def forward(self, x):
        assert x.device.index == self.quant_grid.device.index

        rows, cols_div2 = self.weight_codes.shape
        cols = cols_div2 * 2
        bits = int(math.log2(self.quant_grid.shape[1]))

        W = torch.empty(rows, cols, dtype=self.quant_grid.dtype, device=x.device)
        blocks_per_grid = (rows, )
        threads_per_block = (512, )

        with cp.cuda.Device(x.device.index):
            _gather_sub4bit(grid=blocks_per_grid, block=threads_per_block, shared_mem=2 ** bits * 2, args=[
                self.quant_grid.data_ptr(), self.weight_codes.data_ptr(), W.data_ptr(), bits, cols,
            ])

        return torch.matmul(x, W.t())

def replace_with_quantizers(module, quantizers, name=''):
    if isinstance(module, Sub4BitLinear):
        return

    for attr in dir(module):
        tmp = getattr(module, attr)
        name1 = name + '.' + attr if name != '' else attr
        if name1 in quantizers.keys():
            delattr(module, attr)
            setattr(module, attr, Sub4BitLinear(
                None, None,
                quantizers[name1][0],
                quantizers[name1][1],
                next(tmp.parameters()).dtype
            ))
            print(f'{name1}: weights replaced')
            del tmp

    for name1, child in module.named_children():
        replace_with_quantizers(child, quantizers, name + '.' + name1 if name != '' else name1)

def replace_layers(module, bits=4, dtype=torch.float16, name=''):
    """Recursively replace all nn.Linear layers with Sub4BitLinear in the model."""
    for attr_name in dir(module):
        sub_module = getattr(module, attr_name)
        
        if isinstance(sub_module, nn.Linear):
            delattr(module, attr_name)
            setattr(module, attr_name, Sub4BitLinear(
                sub_module.weight.data, bits=bits, dtype=dtype,
            ))
            del sub_module
        
    for child_name, child in module.named_children():
        replace_layers(child, bits, dtype, f"{name}.{child_name}" if name else child_name)


class LeanQuantModelForCausalLM:
    @classmethod
    def from_pretrained(
        cls, base_model_name_or_path: str, quantized_model_path: str, bits=4, torch_dtype=torch.float16,
        device: Optional[str] = None, device_map: str = 'auto',
        max_memory: Optional[Dict[Union[int, str], Union[int, str]]] = None,
    ):
        """Load a pre-trained model and apply quantized layers."""
        with no_init_weights():
            config = AutoConfig.from_pretrained(base_model_name_or_path)
            model = AutoModelForCausalLM.from_config(config, torch_dtype=torch_dtype)
        
        replace_layers(model.model, bits=bits, dtype=torch_dtype)

        state_dict = {}
        with safe_open(quantized_model_path, framework='pt', device='cpu') as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        
        model.load_state_dict(state_dict)
        
        if isinstance(device, str):
            model = model.to(device)
        else:
            assert device_map == 'auto', "device_map should be 'auto' if no specific device is provided."
            no_split_classes = [type(model.model.layers[0]).__name__]
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split_classes, dtype=torch_dtype)
            device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_classes)
            model = dispatch_model(model, device_map)
        
        # Check if any parts of the model are on CPU and warn the user
        if any(param.device.type == 'cpu' for param in model.parameters()):
            print("Warning: Some model layers are on CPU. For inference, ensure the model is fully loaded onto CUDA-compatible GPUs.")
        
        return model