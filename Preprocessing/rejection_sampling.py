import jsonlines
import json
import random
import re
import os
import copy
# import nltk
import numpy as np
import time
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    RetryError
)
import logging
import signal
from tqdm import tqdm
import requests
from concurrent.futures import ThreadPoolExecutor, TimeoutError
import yaml
import asyncio
import ast
import argparse



evaluate_prompt = """You are an expert that is good at judging whether the response to a given query meets the specified evaluator questions.
Your task is to carefully examine the response to determine if it adheres to each requirement outlined in the evaluator questions.

[Query] {query}
[Response] {response}
[Evaluator Question] {question}

For each question, please provide a justification for your evaluation, explaining how the response does or does not satisfy the criteria and a score ('YES' or 'NO') indicating whether the answer satisfies each constraint.
You should only respond in the following JSON format:
```json
{{
    "Question 1": {{
        "explanation": "",
        "score": "YES" or "NO"
    }},
    "Question 2": {{
        "explanation": "",
        "score": "YES" or "NO"
    }},
}}
```
""".strip()


def parse_tool(tool_str):
    try:
        pattern = re.compile(r"(\{.*\})", re.DOTALL)
        matches = re.findall(pattern, tool_str)
        tool_str = matches[0]
    except Exception as e:
        pass
    if tool_str[0] == "[" and tool_str[-1] == "]":
        tool_str = "{" + tool_str[1:-1] + "}"
    try:
        return json.loads(tool_str)
    except:
        try:
            return ast.literal_eval(tool_str)
        except Exception as e:
            # print(tool_str, e)
            return None

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
            with jsonlines.open(save_path, "a") as f:
                f.write({
                    "query": aug_query,
                    "eval question": question,
                    "initial query": original,
                })
        except Exception as e:
            cnt += 1
            continue


def run_resampling_data(data_path, save_path, index=None):
    sft_data = list(jsonlines.open(data_path, "r"))
    cnt = 0
    request_list = []
    if index == None:
        st = 0
        ed = len(sft_data)
    else:
        st = 10000 * index
        ed = min(len(sft_data), 10000 * (index + 1))

    for data in tqdm(sft_data[st:ed]):
        k = 5
        query = data["query"]
        question = data["eval question"]
        history = data.get("history", [])
        
        system_prompt = "You are an expert tasked with answering the given query. Please provide a clear and concise response directly, without introductory phrases such as 'What a great question,' 'Here is the answer,' or similar expressions. Focus solely on addressing the query."
        prompt = f"Now please answer the given query while stritly following its inside constraints.\n[Query] {query}"
        if len(history) == 0:
            messages = [[{"role": "system", "content": system_prompt} ,{"role": "user", "content": prompt}]] * k
        else:
            if len(history) % 2 != 0:
                continue
            message = [{"role": "system", "content": system_prompt}]
            for idx in range(0, len(history), 2):
                message.extend([{"role": "user", "content": history[idx]["value"]}, {"role": "assistant", "content": history[idx+1]["value"]}])
            message.append({"role": "user", "content": prompt})

            messages = [message] * k

        for message in messages:
            cnt += 1
            request_list.append({
                "custom_id": f"request-{cnt}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": message,
                    "max_tokens": 4096,
                    "temperature": 1.0,
                    "top_p": 0.95,
                    "stop": ["<|eot_id|>", "<|end_of_text|>"],
                },
                
            })
    with jsonlines.open(save_path, "w") as f:
        for r in request_list:
            f.write(r)

def run_reevaluation_data(data_path, result_path, save_path, index=None):
    sft_data = list(jsonlines.open(data_path, "r"))
    resample_results = list(jsonlines.open(result_path))
    print(len(sft_data))
    print(len(resample_results))
    map = {}
    for res in resample_results:
        map[res["custom_id"]] = res
    cnt = 0
    k = 5
    request_list = []
    if index == None:
        st = 0
        ed = len(sft_data)
    else:
        st = 10000 * index
        ed = min(len(sft_data), 10000 * (index + 1))
    
    for data in tqdm(sft_data[st:ed]):
        query = data["query"]
        question = data["eval question"]
        history = data.get("history", [])

        if len(history) % 2 != 0:
            continue
        
        temp = ""
        for i in range(1, len(question)+1):
            temp += "{}. {}\n".format(i, question[i-1])
        for i in range(k):
            cnt += 1
            custom_id = "request-" + str(cnt)
            response = map[custom_id]["response"]['body']["choices"][0]["message"]["content"]
            prompt = evaluate_prompt.format(query=query, response=response, question=temp)
            message = [{"role": "user", "content": prompt}]
            request_list.append({
                "custom_id": f"request-{cnt}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": "meta-llama/Llama-3.1-8B-Instruct",
                    "messages": message,
                    "max_tokens": 4096,
                    "temperature": 0,
                    "top_p": 1.0,
                }
            })
    print(len(request_list))
    with jsonlines.open(save_path, "w") as f:
        for r in request_list:
            f.write(r)


def merge_query_with_response(data_path, response_path, evaluate_path, save_path, index=None):
    sft_data = list(jsonlines.open(data_path, "r"))
    resample_results = list(jsonlines.open(response_path))
    evaluate_results = list(jsonlines.open(evaluate_path))
    print(len(sft_data), len(resample_results), len(evaluate_results))
    response_map = {}
    evaluate_map = {}
    for res in resample_results:
        response_map[res["custom_id"]] = res
    for res in evaluate_results:
        evaluate_map[res["custom_id"]] = res
    cnt = 0
    error = 0
    inputs = []
    not_satisfied = []
    if index == None:
        st = 0
        ed = len(sft_data)
    else:
        st = 10000 * index
        ed = min(len(sft_data), 10000 * (index + 1))
    
    k = 5
    print(k)
    for data in tqdm(sft_data[st:ed]):
        history = data.get("history", [])
        if len(history) % 2 != 0:
            continue

        satisfied = False
        for i in range(k):
            cnt += 1
            custom_id = "request-" + str(cnt)
            response = response_map[custom_id]["response"]['body']["choices"][0]["message"]["content"]
            try:
                eval = evaluate_map[custom_id]["response"]['body']["choices"][0]["message"]["content"]
                eval = parse_tool(eval)
            except:
                eval = None
            flag = True
            
            try:
                for key, value in eval.items():
                    if "Question" in key and ("score" not in value or value["score"].lower() == "no"):
                        flag = False
                        break
            except:
                flag = False
                error += 1

            if flag:
                item = {
                    "instruction": data["query"],
                    "output": response,
                    "history": history
                }
                inputs.append(item)
                satisfied = True
                cnt += (k-i-1)
                break
        if not satisfied:
            item = {
                "instruction": data["query"],
                "output": response,
                "history": history
            }
            inputs.append(item)
            not_satisfied.append(data)
    

    with jsonlines.open(save_path, "w") as f:
        for each in inputs:
            f.write(each)


def merge_query_with_response_dpo(data_path, response_path, evaluate_path, save_path, index=None):
    sft_data = list(jsonlines.open(data_path, "r"))
    resample_results = list(jsonlines.open(response_path))
    evaluate_results = list(jsonlines.open(evaluate_path))
    response_map = {}
    evaluate_map = {}
    for res in resample_results:
        response_map[res["custom_id"]] = res
    for res in evaluate_results:
        evaluate_map[res["custom_id"]] = res
    cnt = 0
    error = 0
    inputs = []
    if index == None:
        st = 0
        ed = len(sft_data)
    else:
        st = 10000 * index
        ed = min(len(sft_data), 10000 * (index + 1))

    k = len(resample_results) // (ed-st)
    print(k)
    for data in tqdm(sft_data[st:ed]):
        pos, neg = None, None
        history = data.get("history", [])
        for i in range(k):
            cnt += 1
            custom_id = "request-" + str(cnt)
            response = response_map[custom_id]["response"]['body']["choices"][0]["message"]["content"]
            try:
                eval = evaluate_map[custom_id]["response"]['body']["choices"][0]["message"]["content"]
                # print(eval)
                eval = parse_tool(eval)
            except:
                eval = None
            flag = True
            
            if isinstance(eval, dict):
                try:
                    for key, value in eval.items():
                        if "Question" in key and value["score"].lower() == "no":
                            flag = False
                            break
                except:
                    flag = False
            else:
                error += 1
                flag = False
                continue
            if flag:
                pos = response
            else:
                neg = response
        if pos is not None and neg is not None:
            inputs.append(
                {
                    "prompt": data["query"],
                    "chosen": pos,
                    "rejected": neg,
                    "history": history
                }
            )
    

    with jsonlines.open(save_path, "w") as f:
        for each in inputs:
            f.write(each)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default=0, type=int)
    parser.add_argument("--split", default=1, type=int, required=False)
    args = parser.parse_args()

    if args.split is None:
        split = None
        args.split = ""
    else:
        split = args.split
        args.split = "_" + str(args.split)

    # run the following code line by line
    run_resampling_data("./Sharegpt_turn1_augmented_query_sft_1.jsonl", f"./Sharegpt_turn1_augmented_query_sft_1_query{args.split}.jsonl", index=split)
    # run_reevaluation_data("./Sharegpt_turn1_augmented_query_sft_1.jsonl", f"./Sharegpt_turn1_augmented_query_sft_1_query{args.split}_output.jsonl", f"./result_call_1219/Sharegpt_turn1_augmented_query_sft_1_query{args.split}_evaluate.jsonl", index=split)
    # merge_query_with_response(data_path="./Sharegpt_turn1_augmented_query_sft_1.jsonl", response_path=f"./Sharegpt_turn1_augmented_query_sft_1_query{args.split}_output.jsonl", evaluate_path=f"./Sharegpt_turn1_augmented_query_sft_1_query{args.split}_evaluate_output.jsonl", save_path=f"./Sharegpt_turn1_augmented_query_sft_1{args.split}_response.jsonl", index=split)
    # merge_query_with_response_dpo(..)