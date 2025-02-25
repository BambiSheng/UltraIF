import os
import json
import jsonlines
from tqdm import tqdm
import ast
import re

def parse_tool(tool_str):
    try:
        pattern = re.compile(r"```json(.*?)```|```(.*?)```", re.DOTALL)
        matches = re.findall(pattern, tool_str)
        tool_str = matches[0][0]
    except Exception as e:
        pass
    try:
        return json.loads(tool_str)
    except (json.JSONDecodeError, TypeError):
        try:
            return ast.literal_eval(tool_str)
        except (ValueError, SyntaxError):
            return None
        
results = list(jsonlines.open("query_tool.jsonl", "r"))

for data in tqdm(results):
    query = data["query"]
    simplified = data["simplified query"]
    if query == simplified:
        continue
    tool = parse_tool(data["tool"])
    if tool is None or tool["question"] == "":
        continue

    with jsonlines.open("./ultracomposer_sft.jsonl", "a") as f:
        f.write({
                "simplified query": simplified,
                "query": query,
                "tool": json.dumps({"question": [tool["question"]]}),
            })