import openai
import argparse
import traceback
import os
import json
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import jsonlines
import re
import ast


model_path = "/model_path/UltraComposer"
max_seq_len = 1024*8

tokenizer = AutoTokenizer.from_pretrained(model_path)
llm = LLM(
    model=model_path,
    trust_remote_code=True,
    tensor_parallel_size=torch.cuda.device_count(),
    max_model_len = max_seq_len,
)


def generate_sample_batch(question_list, batch_size=1000):
    sampling_params = SamplingParams(max_tokens=max_seq_len,
                                    temperature=0.7,
                                    n=1,
                                    stop=["<|eot_id|>"]
                                )
    outputs = llm.generate(question_list, sampling_params, use_tqdm=True)   
    completions = [output.outputs[0].text.strip() for output in outputs]
    return completions


def make_conv(question):
    question = '[initial query]: ' + question
    msg =  [{"role": "user", "content": question}]
    out = tokenizer.apply_chat_template(msg, tokenize=False, 
                                        add_generation_prompt=True, 
                                        chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}")
    return out


def make_conv_multiturn(question, history):
    msg =  [{"role": "user", "content": f"[history]: {history} \n [initial query]: {question}"},]
    out = tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True,
                                        chat_template="{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}")
    return out

def parse_augmented(tool_str):
    try:
        pattern = re.compile(r"```json(.*?)```|```(.*?)```", re.DOTALL)
        matches = re.findall(pattern, tool_str)
        tool_str = matches[0][0]
    except Exception as e:
        pass
    try:
        tool_str = tool_str[1:-1]
        question = tool_str.split("\"question\":")[1].strip()[2:-2].replace("\"", "\'")
        query = tool_str.split("\"question\":")[0].split("\"augmented query\":")[1].strip()[1:-2].replace("\"", "\'")
        tool_str = json.dumps({
            "augmented query": query,
            "question": [question]
        })
    except:
        pass
    
    try:
        return json.loads(tool_str), tool_str
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(tool_str), tool_str
        except (ValueError, SyntaxError):
            return None, None


def load_augmented_query(data_path, save_path):
    cnt = 0
    valid = 0
    data = list(jsonlines.open(data_path, "r"))

    for d in tqdm(data):
        response, new_response = parse_augmented(d["response"])
        try:
            original = d["query"]
            aug_query = response["augmented query"]
            question = response["question"]
            if "human evaluator" in aug_query or "provide the response in JSON format" in aug_query:
                valid += 1
                continue
        except Exception as e:
            cnt += 1
            continue


if __name__ == "__main__":
    cnt = 0
    valid = 0
    # add up to three constraints
    for i in range(0,3):
        data = list(jsonlines.open(f"./Sharegpt_augmented_query_sft_constraint_{i}.jsonl", "r"))
        dataset = list(map(lambda d: make_conv_multiturn(d["query"], d["history"]), data))
        completions = generate_sample_batch(dataset)
        print(len(completions))
        with jsonlines.open(f"./Sharegpt_augmented_query_sft_constraint_{i+1}.jsonl", "w") as f: 
            for d, response in zip(data, completions):
                response, new_response = parse_augmented(response)
                if i == 0:
                    original = d["query"]
                    eval_question = []
                else:
                    original = d["initial query"]
                    eval_question = d["eval question"]
                try:
                    aug_query = response["augmented query"]
                    question = response["question"]
                    if "human evaluator" in aug_query or "provide the response in JSON format" in aug_query or aug_query == d["query"]:
                        raise KeyError
                except:
                    aug_query = d["query"]
                    question = []

                eval_question.extend(question)
                
                f.write({
                    "query": aug_query,
                    "eval question": eval_question,
                    "initial query": original,
                    "response": response,
                    "history": d.get("history", [])
                })