'''
code from https://github.com/mit-han-lab/smoothquant/
'''
from datasets import load_dataset
import functools
from collections import defaultdict

from functools import partial
import numpy as np
from tqdm import tqdm
import torch
import os

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)
import argparse

def get_act_scales(model, tokenizer, dataset_path, num_samples=512, seq_len=512):
    model.eval()
    device = next(model.parameters()).device
    
    # TODO: 使用 hook 函数统计激活值的最大绝对值
    # note: Hook 是一种在程序运行过程中“插入自定义逻辑”的机制，用于在特定事件（如前向、反向传播）发生时执行额外代码。
    
    act_scales = {} # torch.Size([2048])

    def stat_input_hook(m, x, y, name):
        # hook 函数
        raise NotImplementedError()
    
    # hint: 向所有线性层注册 hook
    #       使用 model.named_modules() 函数遍历模型的 名称和模块
    #       输出形如 {String: torch.Size([2048])} 的层名和激活值，
    #       保存在 act_scales 字典中
    #       注册前向传播时候的 hook，可使用模块的
    #       register_forward_hook() 函数

    dataset = load_dataset("parquet", data_files=dataset_path,split="train")
    dataset = dataset.shuffle(seed=42)

    for i in tqdm(range(num_samples)):
        # text = "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n".format(
		# 	instruction=dataset["instruction"][i], input=dataset["output"][i]
		# )
        text = "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n"
        "Continue the following text naturally, maintaining the same style and coherence.\n\n"
        "### Input:\n"
        f"{dataset['text']}\n\n"
        "### Response:\n"
        input_ids = tokenizer(
            text, return_tensors="pt", max_length=seq_len, truncation=True
        ).input_ids.to(device)
        model(input_ids)

    # TODO: 删除 hooks, 你需要提前记录所有 hook 的句柄
    # hint: register_forward_hook() 函数的返回值即为 hook 的句柄

    return NotImplementedError() # 返回值为 act_scales 字典



def build_model_and_tokenizer(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    kwargs = {"torch_dtype": torch.float16, "device_map": "sequential"}
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    return model, tokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-name", type=str, default="../TinyLlama-chat-1.1b-hf", help="model name"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="./act/TinyLlama-chat-v1.0-act.pt",
        help="where to save the act scales",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="../dataset/wikitext-103-v1/train-00000-of-00002.parquet",
        help="location of the calibration dataset, we use the validation set of the Pile dataset",
    )
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--seq-len", type=int, default=512)
    args = parser.parse_args()
    return args


@torch.no_grad()
def main():
    args = parse_args()
    model, tokenizer = build_model_and_tokenizer(args.model_name)

    act_scales = get_act_scales(
        model, tokenizer, args.dataset_path, args.num_samples, args.seq_len
    )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    torch.save(act_scales, args.output_path)


if __name__ == "__main__":
    main()
