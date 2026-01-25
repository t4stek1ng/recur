from openai import OpenAI
import tiktoken
import requests
import json

config = json.load(open('./config.json', 'r'))
api_key = config["api_key"]
model = config["models"][5]


client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

def ds_call():

    messages = [{"role": "user", "content": prompt}]
    response = client.chat.completions.create(
        model="deepseek-reasoner",
        messages=messages,
        max_tokens=16384,
        logprobs=20
    )

    text = response.choices[0].message.content
    logprobs = response.choices[0].logprobs
    reasoning_tokens = response.usage.completion_tokens_details.reasoning_tokens
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens - reasoning_tokens

    print("input_tokens: ", input_tokens)
    print("output_tokens: ", output_tokens)
    print("reasoning_tokens: ", reasoning_tokens)

    return {'text': text, 
            'input tokens': input_tokens,
            'output tokens': output_tokens,
            'reasoning tokens':reasoning_tokens, 
            "entire respose":response}