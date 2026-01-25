# DeepSeek
from openai import OpenAI

def deepseek_attack(prompt: str) -> tuple[str, int]:
    pass

# OpenAI

def gpt_attack(prompt: str) -> tuple[str, int]:
    pass

# Google Gemini 2.5 Flash

from google import genai
from google.genai import types

client = genai.Client()

def gemini_attack(prompt: str) -> tuple[str, int]:
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(
                include_thoughts=True
            )
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

    return {"thought tokens": thoughts_tokens, "output tokens": output_tokens, "thought": thought, "output": output}

# xAI

def grok_attack(prompt: str) -> tuple[str, int]:
    pass