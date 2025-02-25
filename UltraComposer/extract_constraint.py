import jsonlines
import os
from tqdm import tqdm
import json
import re


# first stage extract constraints
prompt_template = """You are an expert in extracting instruction constraints from a given query. 
Definition of Constraint: The smallest unit of restriction or requirement that the instruction imposes on the task.

Query: {QUERY}

- If the query is not a question, or is simple or straightforward without any constraints, please only respond with the following JSON, indicating that no constraints are present.
```json
{{
    "Complex": False
}}
```

- If constraints are present, follow these steps:
    1. Identify the Basic Query: Clearly understand the primary goal of the query, stripping away any constraints. The Basic Query should be the essential task without any added conditions or restrictions.
    2. Extract and Categorize Constraints: Identify and classify constraints based on the following types:
        - Content Constraints:
            - Specific Terms or Symbols: Mandatory use of certain terms or symbols with their exact placement (e.g., must include the word 'beautiful').
            - Required Elements or Concepts: Mandates for including specific elements or concepts in responses, reflecting a scenario or object (e.g., highlights the Great Wall).
            - Thematic Directives: Instructions related to thematic content, perspective, or tone, emphasizing response significance (e.g., write a poem about London).
        - Numerical Constraints:
            - Constraints on quantities related to the content, such as the number of points, sentences, paragraphs, response length, or examples (e.g., within a single paragraph with three sentences).
        - Stylistic Constraints:
            - Desired tone and style for the response (e.g., formal, informal, conversational).
            - Specific language or terminology to be used or avoided (e.g., encyclopedic style).
        - Format Constraints:
            - Required structure or format for the response (e.g., list, JSON, bullet points, Java language).
            - Presentation styles or formatting requirements (e.g., electronic medical record format).
        - Linguistic Constraint：
            - Language use in specific contexts, such as discourse, dialects, sociolects, and language policies (e.g., in English).
            - Sentence structure, including phrases, constituents, and the use of imperatives (e.g., with nouns and verbs).
            - Internal structure of words, including roots, affixes, and morphological changes (e.g., lowercase, single-rhyme).


    Response Format:
    - Do not consider details that are part of the content itself, such as those used in descriptions, scenarios, or examples, unless they directly impose a restriction of the response.
    - The Basic Query should represent the query’s core goal, free from any constraints. 
    - Ensure that extracted constraints do not overlap with the Basic Query.
    - Present each constraint as a dictionary within a list, where each dictionary contains:
        'constraint': The specific restriction or requirement.
        'simplified query': The query after removing this constraint, polished for coherence and correctness.
    - Exclude any constraint types not present in the query.
    ```json
    {{
        "Complex": True,
        "Basic Query": ...,
        "Content Constraints": [
            {{
                "constraint": "...",
                "simplified query": "..."
            }},
            {{
                "constraint": "...",
                "simplified query": "..."
            }},
        ],
        ...
    }}
    ```
    
Here are some examples with their explanations:
Query: "Tell me about the negatives of farming meat"
Response: 
```json
{{
    "Complex": False,
}}
```
Explanation: This query is straightforward, asking about the negatives of farming meat without any additional constraints.

Query: "How do I check if a user pressed the cancel button on a prompt in JS. Answer with exactly three sentences."
Response:
```json
{{
    "Complex": True,
    "Basic Query": "How do I check if a user pressed the cancel button on a prompt in JS.",
    "Numerical Constraints": [
        {{
            "constraint": "Answer with exactly three sentences.",
            "simplified query": "How do I check if a user pressed the cancel button on a prompt in JS."
        }}
    ]
}}
```
Explanation: This query includes a numerical constraint requiring the answer to be exactly three sentences. This additional requirement complicates the query, as it affects how the answer must be structured. 

Query: "Explain quantum computing in simple terms."
Response:
```json
{{
    "Complex": True,
    "Basic Query": "How do I check if a user pressed the cancel button on a prompt in JS.",
    "Content Constraints": [
        {{
            "constraint": "explain in simple terms",
            "simplified query": "Explain quantum computing."
        }}
    ]
}}
```
Explanation: This query includes a content constraint specifying that the explanation should be in simple terms. The primary goal of the query, "quantum computing," is not itself a constraint..

Query: "In Java, I want to replace string like "This is a new {{object}} at {{place}}" with a Map, {{object: "student", "point 3, 4"}}, and get a result "This is a new student at point 3, 4". How can I do?"
Response:
```json
{{
    "Complex": True,
    "Basic Query": "I want to replace string like "This is a new {{object}} at {{place}}" with a Map, {{object: "student", "point 3, 4"}}, and get a result "This is a new student at point 3, 4". How can I do?",
    "Format Constraints": [
        {{
            "constraint": "Answer the question using Java language.",
            "simplified query": "I want to replace string like "This is a new {{object}} at {{place}}" with a Map, {{object: "student", "point 3, 4"}}, and get a result "This is a new student at point 3, 4". How can I do?"
        }}
    ]
}}
```
Explanation: The query's main goal is to ask how to perform a specific task in Java. The provided string and Map examples are part of the description and are not constraints. The requirement to use Java is a format constraint.

Query: "Write a python program which accept a command line param as question and send it to server via HTTP get method."
Response:
```json
{{
    "Complex": True,
    "Basic Query": "Write a program which accept a command line param as question and send it to server via HTTP get method.",
    "Format Constraints": [
        {{
            "constraint": "python program",
            "simplified query": "Write a program which accept a command line param as question and send it to server via HTTP get method."
        }}
    ]
}}
```
Explanation: This query limits the solution into Python program, and 'a command line ...' is the initial goal of the query.

Query: "In Shakespeare's tone, recommend me ten Chinese books. Use bulletpoint in your answer."
Response:
```json
{{
    "Complex": True,
    "Basic Query": "Recommend me books",
    "Stylistic Constraints": [
        {{
            "constraint": "Shakespeare's tone",
            "simplified query": "Recommend me ten Chinese books. Use bulletpoint in your answer."
        }}
    ],
    "Format Constraints": [
        {{
            "constraint": "Use bulletpoint in your answer.",
            "simplified query": "In Shakespeare's tone, recommend me ten Chinese books."
        }}
    ]，
    "Numerical Constraints": [
        {{
            "constraint": "ten Chinese books",
            "simplified query": "In Shakespeare's tone, recommend me Chinese books. Use bulletpoint in your answer."
        }}
    ]
    "Content Constraints": [
        {{
            "constraint": "Chinese books",
            "simplified query": "In Shakespeare's tone, recommend me ten books. Use bulletpoint in your answer."
        }}
    ]
}}
```
Explanation: The primary goal of the query is to recommend books. The query includes multiple constraints: a stylistic constraint to use "Shakespeare's tone," a format constraint to "use bullet points," a numerical constraint to recommend "ten books," and a content constraint to focus on "Chinese books."

Please only provide the response in JSON format.
""".strip()


def packing(data_path="your_data.jsonl"):
    data = list(jsonlines.open(data_path, "r"))
    for idx, d in enumerate(tqdm(data)):
        query = prompt_template.format(QUERY=d['query'])
        messages = [{'role': 'user', 'content': query}]
        # use vllm batch call for inference
        with jsonlines.open("./extract_constraint_prompts.jsonl", "a") as f:
            f.write({
                "custom_id": f"request-{idx}",
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
    
if __name__ == "__main__":
    packing("your_data.jsonl")
    # after packing, you need to run batch and extract the results to 
