import asyncio
from googletrans import Translator

async def en_to_zh(text: str) -> str:
    translator = Translator()
    result = await translator.translate(text, src='en', dest='zh-cn')
    return result.text

# 在 Jupyter 中需要这样调用
print(asyncio.run(en_to_zh("Hello, how are you today?")))
