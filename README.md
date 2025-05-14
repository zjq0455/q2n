# Boost Post-Training Quantization via Null Space Optimization for Large Language Models
**This paper introduces the null space optimization strategy into LLMs quantization to further enhance the performance of existing baselines and provide a novel perspective for future research.**

Existing post-training quantization methods for large language models (LLMs) offer remarkable success. However, the increasingly marginal performance gains suggest that existing quantization strategies are insufficient to support the development of more advanced algorithms. To inspire new directions for future research, in this paper we introduce the concept of null space into LLMs quantization. We argue that the quantization error can be effectively alleviated by constraining the post-quantization weight perturbation to lie within the null space of input activations. To prove this idea, we propose a plug-and-play null space projection module for existing milestone PTQ baselines named Q2N. Specifically, we first design an efficient and accurate null-space projection approximation method tailored to the characteristics of LLMs. Subsequently, we theoretically derive a closed-form solution for an equivalent vector of the obtained null space projection matrix, which satisfies practical inference condition while avoiding additional memory overhead. Extensive experiments are conducted on various state-of-the-art LLMs and baselines, demonstrating the effectiveness of both our Q2N and the perspective of null space optimization for LLMs quantization. We view this paper the first step to further alleviate the quantization error based on the insights of null space, hoping it inspiring future researchers to design more advanced quantization methods.

## Install
1. Install all dependencies of Q2N, run:

```
conda create -n q2n python=3.10
cd q2n
pip install -r requirements.txt
```
2. If run Qwen3, please update `transformers >= 4.51.0`

## Usage
### Run & evaluate the perplexity & save fake-quantized model
We use LLaMA3.3-70B as an example.
1. GPTQ
```
python llama.py meta-llama/Llama-3.3-70B-Instruct c4 --wbits 2 --groupsize 128 --true-sequential --act-order \
 --threshold 0.1 --lambda 0.2 --save_pth <PATH_NAME>
```

2. QuIP
```
python llama.py meta-llama/Llama-3.3-70B-Instruct c4 --wbits 2 --groupsize 128 --quant ldlq --pre_gptqH --pre_rescale \
 --pre_proj --pre_proj_extra 1 --qfn b --eval --threshold 0.1 --lambda 0.2 --save_pth <PATH_NAME>
```

3. PB-LLM
```
python run.py meta-llama/Llama-3.3-70B-Instruct c4 xnor --low_frac 0.9 --high_bit 8 --salient_metric hessian \
 --threshold 0.1 --lambda 0.2 --save_pth <PATH_NAME>
```

4. LeanQuant
```
python llama.py meta-llama/Llama-3.3-70B-Instruct c4 --wbits 2 --groupsize 128 --nsamples 128 --true-sequential --act-order \
 --percdamp 0.1 --exponent 4 --threshold 0.1 --lambda 0.2 --save_pth <PATH_NAME>
```

5. OWQ
```
python main.py meta-llama/Llama-3.3-70B-Instruct c4 --wbits 3 --target_bit 3.01 \
--threshold 0.1 --lambda 0.2 --save_pth <PATH_NAME>
```

6. QuaRot
```
python main.py --model meta-llama/Llama-3.3-70B-Instruct --cal_dataset c4 --rotate\
--a_bits 4 --w_bits 4 --w_clip --threshold 0.1 --lambda 0.2 --save_pth <PATH_NAME>
```

### Evaluate the accuracies on downstream reasoning tasks
```
CUDA_VISIBLE_DEVICES=1,6  lm_eval --model hf --model_args pretrained="<PATH_NAME>",parallelize=True \
--tasks hellaswag,winogrande,piqa,arc_easy,arc_challenge,mmlu,race --batch_size auto
```


## Related Project
[GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://github.com/IST-DASLab/gptq)

[QuIP: 2-Bit Quantization of Large Language Models with Guarantees](https://github.com/Cornell-RelaxML/QuIP)

[PB-LLM: Partially Binarized Large Language Models](https://github.com/hahnyuan/PB-LLM)

[LeanQuant: Accurate and Scalable Large Language Model Quantization with Loss-error-aware Grid](https://github.com/LeanModels/LeanQuant)

[OWQ: Outlier-Aware Weight Quantization for Efficient Fine-Tuning and Inference of Large Language Models](https://github.com/xvyaward/owq)

[QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs](https://github.com/spcl/QuaRot)

[Language Model Evaluation Harness (lm-eval-harness)](https://github.com/EleutherAI/lm-evaluation-harness)