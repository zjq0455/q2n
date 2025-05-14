from argparse import ArgumentParser

import torch
from transformers import AutoTokenizer
import lm_eval

from leanquant import LeanQuantModelForCausalLM

if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument("--base_model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--leanquant_path", type=str, default="models/llama-2-7b_b3_e3_d0.1.safetensors")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--tasks", nargs='+', type=str, default=["mmlu"])
    parser.add_argument("--eval_batch_size", type=int, default=4)
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_name_or_path)
    model = LeanQuantModelForCausalLM.from_pretrained(
        args.base_model_name_or_path, args.leanquant_path,
        args.bits, torch_dtype=torch.bfloat16 if args.use_bf16 else torch.float16,
        device_map="auto",
    )

    model_eval = lm_eval.models.huggingface.HFLM(model, tokenizer=tokenizer, device=model.device, batch_size=args.eval_batch_size, trust_remote_code=True)
    results = lm_eval.simple_evaluate(model=model_eval, tasks=args.tasks)
    print(results['results'])
