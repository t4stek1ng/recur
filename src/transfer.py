import json
import os

# configurations
config = json.load(open('./config.json', 'r'))
api_key = config["api_key"]
target_model = "deepseek-reasoner"
# target_model = "o1-2024-12-17"
# target_model = "gemini-2.5-flash"
# target_model = "grok-4-fast-reasoning"
source_models = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-14B", "QwQ-32B"]
temperature = 0
reasoning_effort = "high" #low, medium, high
max_length = 16384 * 4

# DeepSeek
if target_model == "deepseek-reasoner":
    from openai import OpenAI

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

    def transfer_attack(target_model:str, source_model:str, id: int, reasoning_effort: str, temperature: float, prompt: str):
        thought = ""
        output = ""
        messages = [{"role": "user", "content": prompt}]
        response = client.chat.completions.create(
            model=target_model,
            messages=messages,
            max_tokens=max_length,
            temperature=temperature
        )

        thought = response.choices[0].message.reasoning_content
        output = response.choices[0].message.content
        thought_tokens = response.usage.completion_tokens_details.reasoning_tokens
        output_tokens = response.usage.completion_tokens - thought_tokens

        return {'id': id, "source": source_model, "temperature": temperature, "thought tokens": thought_tokens, "output tokens": output_tokens, "thought": thought, "output": output}

# OpenAI
if target_model == "o1-2024-12-17":
    client = OpenAI(api_key=api_key, base_url="")

    def transfer_attack(target_model:str, source_model:str, id: int, reasoning_effort: str, temperature: float, prompt: str):
        thought = ""
        output = ""
        response = client.responses.create(
            model=target_model,
            input=prompt,
            reasoning={"effort": reasoning_effort},
            temperature=temperature
        )

        output = response.output_text
        thought_tokens = response.usage.output_tokens_details.reasoning_tokens
        output_tokens = response.usage.output_tokens

        return {'id': id, "source": source_model, "reasoning effort": reasoning_effort, "temperature": temperature, "thought tokens": thought_tokens, "output tokens": output_tokens, "thought": thought, "output": output}


# Google Gemini 2.5 Flash
if target_model == "gemini-2.5-flash":
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)

    def transfer_attack(target_model:str, source_model:str, id: int, reasoning_effort: str, temperature: float, prompt: str):
        thought = ""
        output = ""
        response = client.models.generate_content(
            model=target_model,
            contents=prompt,
            config=types.GenerateContentConfig(
                thinking_config=types.ThinkingConfig(
                    include_thoughts=True
                ),
                temperature=temperature,
                # max_output_tokens = max_length
            )
        )

        for part in response.candidates[0].content.parts:
            if not part.text:
                continue
            if part.thought:
                thought = part.text
            else:
                output = part.text

        thoughts_tokens = response.usage_metadata.thoughts_token_count
        output_tokens = response.usage_metadata.candidates_token_count

        return {"id": id, "source": source_model, "temperature": temperature, "thought tokens": thoughts_tokens, "output tokens": output_tokens, "thought": thought, "output": output}

# xAI
if target_model == "grok-4":
    from xai_sdk import Client as xClient
    from xai_sdk.chat import user

    client = xClient(api_key=api_key)

    def transfer_attack(target_model: str, source_model: str, id: int, reasoning_effort: str, temperature: float, prompt: str):
        thought = ""
        output = ""
        chat = client.chat.create(
            model=target_model,
            temperature=temperature
        )

        chat.append(user(prompt))
        try:
            response = chat.sample()

        except Exception as e:
            raise e

        output = response.content
        output_tokens = response.usage.completion_tokens
        thoughts_tokens = response.usage.reasoning_tokens

        return {"id": id, "source": source_model, "temperature": temperature, "thought tokens": thoughts_tokens, "output tokens": output_tokens, "thought": thought, "output": output}

if os.path.exists(f"./exp/{target_model}"):
    pass
else:
    os.mkdir(f"./exp/{target_model}")

if os.path.exists(f"./exp/{target_model}/results.jsonl"):
    pass
else:
    with open(f"./exp/{target_model}/results.jsonl", "w") as f:
        pass

if __name__ == "__main__":
    for source_model in source_models:
        with open(f"./exp/{source_model}/promtps.jsonl") as f:
            prompts = [json.loads(line) for line in f.readlines() if json.loads(line)["generation"] == False]

        for data in prompts:
            prompt = data["prompt"]
            id = data["id"]

            result = transfer_attack(target_model, source_model, id, reasoning_effort, temperature, prompt)

            with open(f"./exp/{target_model}/results.jsonl", "a") as f:
                f.write(json.dumps(result)+"\n")