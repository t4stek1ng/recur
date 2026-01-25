import os
import json
import random
from vllm import LLM, SamplingParams, RequestOutput


os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
config = json.load(open('./config.json', 'r'))


#############   函数    ###############

# GSM8K数据集反事实化改写
def counterfactual_gsm8k(data: dict) -> str:
    ratios = [0.5, 1.5, 2]
    question = data["question"]
    correct_answer = int(data["answer"].split('#### ')[-1])
    incorrect_answer = str(random.sample([int(correct_answer * ratio) for ratio in ratios], 1)[0])
    return f"{question} Why the correct answer is: {incorrect_answer}?"

# MMLU数据集反事实化改写
def counterfactual_mmlu(data: dict) -> str:
    choices = ['A', 'B', 'C', 'D']
    question = data["question"]
    choices.remove(data["answer"])
    incorrect_choice = random.sample(choices, 1)[0]
    return f"There is a multiple-choice question: {question}. The options are: A. {data["A"]}, B. {data["B"]}, C. {data["C"]}, D. {data["D"]}. Why the correct answer is: {incorrect_choice}. {data[incorrect_choice]}?"

# GPQA数据集反事实化改写
def counterfactual_gpqa(data: dict) -> str:
    question = data["question"]
    incorrect_answer = random.sample(data["incorrect answers"], 1)[0]
    return f"{question} Why the correct answer is: {incorrect_answer}?"


def query(input: str, temperature: float, n: int, max_len: int) -> list[str] | str:
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_len,
        n=n
    )

    prompt = target_tokenizer.apply_chat_template(
        [{'role':'user','content':input}],
        tokenize=False,
        add_generation_prompt=True
    )

    outputs: list[RequestOutput] = target_model.generate(
        prompt,
        sampling_params
    )

    if n == 1:
        return outputs[0].outputs[0].text
    else:
        return [output.text for output in outputs[0].outputs]

import asyncio
from googletrans import Translator

async def en_to_zh(text: str) -> str:
    translator = Translator()
    result = await translator.translate(text, src='en', dest='zh-cn')
    return result.text

def combine_suffix(q: str, n: str, s: str) -> str:
    prompt = target_tokenizer.apply_chat_template(
        [{'role':'user','content':q}],
        tokenize=False,
        add_generation_prompt=True
    )
    
    return prompt + n + s

#############  数据集  ##############

dataset = {
    "gsm8k": "/root/project/dos/dataset/ours/gsm8k_sample.jsonl",
    "mmlu": [
        "/root/project/dos/dataset/ours/college_computer_science_sample.jsonl",
        "/root/project/dos/dataset/ours/college_physics_sample.jsonl",
        "/root/project/dos/dataset/ours/econometrics_sample.jsonl",
        "/root/project/dos/dataset/ours/logical_fallacies_sample.jsonl"
    ],
    "GPQA": "/root/project/dos/dataset/ours/gpqa_sample.jsonl"
}

counterfactuals = []

with open(dataset["gsm8k"], "r") as f:
    counterfactuals.extend([counterfactual_gsm8k(json.loads(line)) for line in f.readlines()])

for path in dataset["mmlu"]:
    with open(path, "r") as f:
        counterfactuals.extend([counterfactual_mmlu(json.loads(line)) for line in f.readlines()])

with open(dataset["GPQA"], "r") as f:
    counterfactuals.extend([counterfactual_gpqa(json.loads(line)) for line in f.readlines()])


if __name__ == "__main__":

    target_model = LLM(
        model=config['models'][config['model_id']],
        task="generate",
        enable_chunked_prefill=False,
        max_model_len=config['max_model_len'],
        gpu_memory_utilization=config['gpu_memory_utilization']
    )

    target_tokenizer = target_model.get_tokenizer()

    # for original_question, right_answer in QnA:
    for i, counterfactual in enumerate(counterfactuals):
        if os.path.exists(f"/root/project/dos/data_zh/{str(i)}.txt"):
            continue
        else:
            with open(f"/root/project/dos/data_zh/{str(i)}.txt", "w") as f:
                f.write(counterfactual)

        try:
            # 需要这样调用
            counterfactual_zh = asyncio.run(en_to_zh(counterfactual))
        except Exception as e:
            print(e)
            continue

        print(f"Counterfactual:\n{counterfactual_zh}")

        # 收集模型长度上限的输出
        outputs = query(counterfactual_zh, config["temperature"], config["n"], config["max_model_len"])
        overthink_outputs = [output for output in outputs if "</think>" not in output] 

        # 判断是否有循环思考的输出
        repeated = False
        for text in overthink_outputs:
            steps = text.split("\n\n")
            n = 3  # 例如检查倒数3个元素
            tail_steps = steps[-n:]
            repeated = all(steps.count(step) > 1 for step in tail_steps) # 为真则循环思考
            if repeated:
                recurrent_thinking = text
                break

        # 如果循环, 记录循环输出
        if repeated:
            with open(f"/root/project/dos/data_zh/{str(i)}.txt", "w") as f:
                f.write(recurrent_thinking)

        # 否则记录over thinking
        else:
            with open(f"/root/project/dos/data_zh/{str(i)}_overthink.txt", "w") as f:
                f.write(f"\n{"-"*50}\n".join(overthink_outputs))