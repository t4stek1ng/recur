import json
import os

temperature = 0
target_models = ["deepseek-reasoner", "o1-2024-12-17", "gemini-2.5-flash", "grok-4-fast-reasoning"]
source_model = None

with open("/home/guge_wzw/project/recur/dataset/ours/gsm8k_sample.jsonl", "r") as f:
    gsm8k_dataset: list[dict[str, list[str]] | str | int] = [json.loads(line) for line in f.readlines()]

for i, data in enumerate(gsm8k_dataset):
    gsm8k_dataset[i] = {
        "id": i,
        "question": data["question"],
        "options": None,
        "option": None,
        "answer": data["answer"].split("\n#### ")[1],
        "incorrect": []
    }


for model in target_models:
    if model == "deepseek-reasoner":
        # DeepSeek
        from openai import OpenAI

        client = OpenAI(api_key="", base_url="https://api.deepseek.com")

        def deepseek_attack(target_model:str, source_model:str, id: int, temperature: float, prompt: str):
            thought = ""
            output = ""
            messages = [{"role": "user", "content": prompt}]
            response = client.chat.completions.create(
                model=target_model,
                messages=messages,
                # max_tokens=max_length,
                temperature=temperature
            )

            thought = response.choices[0].message.reasoning_content
            output = response.choices[0].message.content
            thought_tokens = response.usage.completion_tokens_details.reasoning_tokens
            output_tokens = response.usage.completion_tokens - thought_tokens

            return {'id': id, "source": source_model, "temperature": temperature, "thought tokens": thought_tokens, "output tokens": output_tokens, "thought": thought, "output": output}

    elif model == "o1-2024-12-17":
        # OpenAI
        client = OpenAI(api_key="", base_url="https://turingai.plus/v1")
        reasoning_efforts = ["medium", "high"]
        def gpt_attack(target_model:str, source_model:str, id: int, reasoning_effort: str, temperature: float, prompt: str):
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

    elif model == "gemini-2.5-flash":
        # Google Gemini 2.5 Flash
        from google import genai
        from google.genai import types

        client = genai.Client(api_key="")

        def gemini_attack(target_model:str, source_model:str, id: int, temperature: float, prompt: str):
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

    else:
        # xAI
        from xai_sdk import Client as xClient
        from xai_sdk.chat import user

        client = xClient(api_key="")

        def grok_attack(target_model: str, source_model: str, id: int, temperature: float, prompt: str):
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

    if os.path.exists(f"/home/guge_wzw/project/recur/exp/baseline/{model}_gsm8k_results.jsonl"):
        pass
    else:
        with open(f"/home/guge_wzw/project/recur/exp/baseline/{model}_gsm8k_results.jsonl", "w") as f:
            pass

    for id, data in enumerate(gsm8k_dataset):
        content = data["question"]
        if model == "deepseek-reasoner":
            result = deepseek_attack(model, source_model, id, temperature, content)
            with open(f"/home/guge_wzw/project/recur/exp/baseline/{model}_gsm8k_results.jsonl", "a") as f:
                f.write(json.dumps(result)+"\n")
        elif model == "o1-2024-12-17":
            for reasoning_effort in reasoning_efforts:
                result = gpt_attack(model, source_model, id, reasoning_effort, temperature, content)
                with open(f"/home/guge_wzw/project/recur/exp/baseline/{model}_gsm8k_results.jsonl", "a") as f:
                    f.write(json.dumps(result)+"\n")
        elif model == "gemini-2.5-flash":
            result = gemini_attack(model, source_model, id, temperature, content)
            with open(f"/home/guge_wzw/project/recur/exp/baseline/{model}_gsm8k_results.jsonl", "a") as f:
                f.write(json.dumps(result)+"\n")
        else:
            result = grok_attack(model, source_model, id, temperature, content)
            with open(f"/home/guge_wzw/project/recur/exp/baseline/{model}_gsm8k_results.jsonl", "a") as f:
                f.write(json.dumps(result)+"\n")