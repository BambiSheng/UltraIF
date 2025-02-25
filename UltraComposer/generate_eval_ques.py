import jsonlines
import os
from tqdm import tqdm
import json
import ast
import re

# second stage: generate question
generate_prompt_template = """You are an expert in crafting questions to evaluate whether a response to a query adheres to specific constraints.

For the given constraint, please design a question that human evaluators can use to assess if the response meets the specified constraint. The question should focus solely on the given constraint and not other constraints present in the original query.

Specifically, if the given constraint is meaningless or is a part of the content itself, such as those used in descriptions, scenarios, or examples, you can respond with an empty string.

Here are some examples:
Query: In Shakespeare's tone, recommend me ten Chinese books. Use bulletpoint in your answer.
Constraint: Use bulletpoint in your answer.
Output: 
```json
{{
    "question":  "Does the response use bullet points?"
}}
```
Explanation: This question checks whether the response adheres to the bullet point format, without considering the tone or the number of books.

Query: Where did the nationalists meet in 1786 to discuss the issues regarding the united states government?
Constraint: In 1786.
Output: 
```json
{{
    "question": "Does the response correctly identify the location where nationalists met in 1786?"
}}
```
Explanation: Checking merely for the presence of "1786" doesn't ensure that the response is correct or complete. The output needs to verify that the response specifies the correct location of the meeting in 1786. It should confirm that the meeting place.

Query: I am planning to give you a voice, and communicate through the speech medium. I need a speech recognizer, a wake call detector, and a speech synthesizer for your voice. Suggest a python script utilizing existing libraries to achieves the goal.
Constraint: Suggest a python script.
Output: 
```json
{{
    "question": "Is the response a Python script that utilizes existing libraries to achieve the goal of creating a speech recognizer, a wake call detector, and a speech synthesizer?"
}}
```
Explanation: This query requires a Python script that provides specific functionalities, including a speech recognizer, etc.

Query: In Java, I want to replace string like \"This is a new {{object}} at {{place}}\" with a Map, {{object: \"student\", \"point 3, 4\"}}, and get a result \"This is a new student at point 3, 4\". How can I do?
Constraint: Answer the question using Java language.
Output: 
```json
{{
    "question": "Is this reply a Java program to complete this task?"
}}
```
Explanation: This question checks if the response is a Java program that completes the string replacement task.

Query: In Java, I want to replace string like \"This is a new {{object}} at {{place}}\" with a Map, {{object: \"student\", \"point 3, 4\"}}, and get a result \"This is a new student at point 3, 4\". How can I do?
Constraint: get a result "This is a new student at point 3, 4".
Output: 
```json
{{
    "question": ""
}}
```
Explanation: Using Java language is the real constraint, description like "get result" and so on is just describing the scenario of the query, so this is not a constraint.


Query: This is not less code this is java.
Constraint: In java.
Output: 
```json
{{
    "question": ""
}}
```
Explanation: The constraint "In Java" appears to be part of the narrative or context rather than a specific, actionable constraint. The query itself does not provide a clear directive related to Java code or functionality. Therefore, the constraint does not qualify as something that can be directly assessed or evaluated in isolation.


Please design a question for the specified constraint for the given query, and respond in the JSON format without explanation.
Query: {query}
Constraint: {constraint}
```json
{{
    "question": "string",
}}
```
""".strip()


def packing(data_path, batch_call_results):
    data = list(jsonlines.open(data_path, "r"))
    extract_constraint = list(jsonlines.open(batch_call_results, "r"))

    for idx, d in enumerate(tqdm(data)):
        d["idx"] = str(idx)
        # extract constraint
        try:
            constraint = extract_constraint[idx]['output']
            d['output'] = constraint
        except Exception as e:
            print(e)
            continue
        
        # generate questions
        try:
            constraints = d['output']
            try:
                constraints = json.loads(constraints.strip())
            except:
                try:
                    constraints = ast.literal_eval(constraints.strip())
                except Exception as e:
                    continue

            if not constraints['Complex']:
                continue
            
            basic = constraints["Basic Query"]
            constraints.pop('Complex')
            constraints.pop('Basic Query')
            cnt = 0
            for key, value in constraints.items():
                for dict in value:
                    cnt += 1
                    constraint = dict['constraint']
                    simplified = dict['simplified query']

                    query = d['query']

                    messages = [{'role': 'user', 'content': generate_prompt_template.format(query=query, constraint=constraint)}]
                    with jsonlines.open("./generate_questions_prompts.jsonl", "a") as f:
                        f.write({
                            "custom_id": f"request-{idx}-{cnt}",
                            "method": "POST",
                            "url": "/v1/chat/completions",
                            "body": {
                                "model": "meta-llama/Meta-Llama-3.1-70B-Instruct",
                                "messages": messages,
                                "max_tokens": 4096,
                                "temperature": 0,
                                "top_p": 1.0
                            }
                        })
                    
        except Exception as e:          
            print(e)
            continue            


if __name__ == "__main__":
    packing("your_data_path", "batch_call_results")
    # after packing, you need to run batch and extract the results to 