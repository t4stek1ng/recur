import json
config = json.load(open('./config.json', 'r'))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
# os.environ['RANK'] = '0'
# os.environ['VLLM_USE_V1'] = '0'
# os.environ['OMP_NUM_THREADS'] = '1'

import re
from collections import Counter

# load the dict
with open("/root/project/dos/words_new.json", "r") as f:
    lib = json.loads(f.read())

head = lib['head']
error = lib['error']
solution = lib['solution']
twist = lib['twist']

# sorting
sorted_head = sorted(head, key=len, reverse=True)
sorted_error = sorted(error, key=len, reverse=True)
sorted_solution = sorted(solution, key=len, reverse=True)

# 用 re.escape 转义特殊字符，避免它们被当成正则元字符
pattern_head = "|".join(re.escape(w) for w in sorted_head)
pattern_head = r"^(?:" + pattern_head + r")"
pattern_error = "|".join(re.escape(w) for w in sorted_error)
pattern_solution = "|".join(re.escape(w) for w in sorted_solution)

head_re = re.compile(pattern_head)
error_re = re.compile(pattern_error)
solution_re = re.compile(pattern_solution)

PH1 = '{head}'
PH2 = '{error}'
PH3 = '{solution}'

#############   functions    ###############
import random

## 问题重写方法：使回答变得模棱两可，奖励模型评分对废话的奖励高于给出确定答案的奖励（鲁棒性、可迁移性）
# 1. 指令为解释错误的答案（Why the answer is xxx）
# 2. 指令为否定正确答案（xxx is not the correct answer）
# 3. 指令为设置不存在的陷阱（There is a trap in the question, you won't figure out the correct answer at once）
## 在不同模型上提高成功率的做法：
# 1. 重复尝试
# 2. 尝试每种不同的重写方法
## 无法 over-think 的原因
# 1. 忽视指令，生成正确答案
# 2. 彻底接受指令，生成错误答案
# 3. 钻指令的空子，生成不违反指令字面意思的正确答案
## 不同模型之间的差异
# 1. 指令遵循能力

# GSM8K 数据集问题改写
def counterfactual_gsm8k(data: dict) -> str:
    ratios = [0.5, 1.5, 2]
    question = data["question"]
    correct_answer = int(data["answer"].split('#### ')[-1])
    incorrect_answer = str(random.sample([int(correct_answer * ratio) for ratio in ratios], 1)[0])
    return f"{question} Why the correct answer is {incorrect_answer}?"

# MMLU 数据集问题改写
def counterfactual_mmlu(data: dict) -> str:
    choices = ['A', 'B', 'C', 'D']
    question = data["question"]
    choices.remove(data["answer"])
    incorrect_choice = random.sample(choices, 1)[0]
    return f"There is a multiple-choice question: {question}. The options are: A. {data["A"]}, B. {data["B"]}, C. {data["C"]}, D. {data["D"]}. Why the correct answer is {incorrect_choice}. {data[incorrect_choice]}?"

# GPQA 数据集问题改写
def counterfactual_gpqa(data: dict) -> str:
    question = data["question"]
    incorrect_answer = random.sample(data["incorrect answers"], 1)[0]
    return f"{question}\nWhy the correct answer is {incorrect_answer}?"

def query(input: str | list[str], apply_chat_template: bool, temperature: float, n: int, max_len: int):
    if apply_chat_template:
        if isinstance(input, str):
            prompt = target_tokenizer.apply_chat_template(
                [{'role':'user','content':input}],
                tokenize=False,
                add_generation_prompt=True
            )

        else:
            prompt = [
                target_tokenizer.apply_chat_template(
                    [{'role':'user','content':content}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for content in input
            ]
    else:
        prompt = input

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_len,
        n=n
    )

    outputs: list[RequestOutput] = target_model.generate(
        prompt,
        sampling_params
    )

    # 将输出与输入拼接为完整序列
    if isinstance(input, str):
        if n == 1:
            return prompt + outputs[0].outputs[0].text
        else:
            return [[p+response.text for response in output.outputs] for p, output in zip([prompt], outputs)]
    else:
        return [[p+response.text for response in output.outputs] for p, output in zip(prompt, outputs)]


def tokenize(sentence):
    # str to words
    return re.findall(r"[A-Za-z]+", sentence.lower())

def similarity_rate_count(s1, s2):
    words1 = tokenize(s1)
    words2 = tokenize(s2)

    c1 = Counter(words1)
    c2 = Counter(words2)

    # 多重交集：每个词取两句出现次数的 min
    common_count = sum(min(c1[w], c2[w]) for w in c1)

    # 重复率 = 共同词汇数量（按出现次数） / 第一句总词数
    rate = min(common_count / len(words1), common_count / len(words2)) if (len(words1)>0 and len(words2)>0) else 0
    return rate

def replace_with_placeholders(text: str) -> str:
    # 按顺序对三个列表依次替换
    # 如果同一段文本可能同时匹配多个列表，先后顺序会影响结果
    text = head_re.sub(PH1, text)
    text = error_re.sub(PH2, text)
    text = solution_re.sub(PH3, text)
    return text

def imitate_overthink(overthink_steps: list[str]) -> list[str]:
    overthink_template = []
    for step in overthink_steps:
        overthink_template.append(replace_with_placeholders(step))

    return overthink_template

import random

class SafeDict(dict):
    def __missing__(self, key):
        return "{" + key + "}"

def overthinking_gen(overthink_template: list[str]) -> list[str]:
    overthinking = []
    for step in overthink_template:
        # sample may larger than population
        try:
            overthinking.append(step.format_map(SafeDict({"head":random.sample(sorted(head_local),1)[0], "error":random.sample(sorted(error_local),1)[0], "solution":random.sample(sorted(solution_local),1)[0]})))
        except:
            try:
                step = re.sub(r'\{head\}', re.escape(random.sample(sorted(head_local),1)[0]), step)
                step = re.sub(r'\{error\}', re.escape(random.sample(sorted(error_local),1)[0]), step)
                step = re.sub(r'\{solution\}', re.escape(random.sample(sorted(solution_local),1)[0]), step)
                overthinking.append(step)
            except:
                step = re.sub(r'\{head\}', re.escape(random.sample(head,1)[0]), step)
                step = re.sub(r'\{error\}', re.escape(random.sample(error,1)[0]), step)
                step = re.sub(r'\{solution\}', re.escape(random.sample(solution,1)[0]), step)
                overthinking.append(step)

    return overthinking

def combine_suffix(q: str, original: str, s: str) -> str:
    prompt = target_tokenizer.apply_chat_template(
        [{'role':'user','content':q}],
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = prompt + original + s
    if len(target_tokenizer.encode(inputs)) >= config["max_model_len"]:
        return None
    else:
        return inputs

def recurrent_check(text: str) -> bool:
    """
    检查输出是否产生循环
    
    :param text: 输出文本
    :type text: str
    :return: 检查结果
    :rtype: bool
    """

    steps = text.split("\n\n")[:-1]
    test_step = steps[-1]
    rev = steps[-2::-1]
    if test_step in rev:
        # 往前找到第一个和 test step 相同的思考步骤
        idx_rev = rev.index(test_step)
        # 换算成正序索引
        same_idx = len(rev) - 1 - idx_rev # 重复步骤索引
        test_idx = len(steps) - 1 # 最后一步索引
        distance = test_idx - same_idx
        if same_idx > distance: # 回溯重复模式时不会越界:
            if distance > 1: # 检查整个模式是否相同
                if all(steps[index] == steps[index - distance] for index in range(same_idx, test_idx)):
                    return True
                else:
                    return False
            else: # 重复思考相邻，可能存在非循环情况，至少重复10次
                if all(steps[index] == steps[test_step] for index in range(test_step - 10, test_idx)):
                    return True
                else:
                    False
        else:
            return False

def overthink_check(text: str) -> bool:
    """
    检查输出是否过度思考
    
    :param text: 输出文本
    :type text: str
    :return: 检查结果
    :rtype: bool
    """

    steps = text.split("\n\n")
    
def overthink_choose(texts: list[str]) -> str:
    """
    选择过度思考文本
    
    :param texts: list of overthinks
    :type texts: list[str]
    :return: choosed overthink
    :rtype: str
    """
    pass



#############  dataset  ##############

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

#############  main  ##############

if __name__ == "__main__":
    from vllm import LLM, SamplingParams, RequestOutput

    target_model = LLM(
        model=config['models'][config['model_id']],
        task="generate",
        enable_chunked_prefill=False,
        max_model_len=config['max_model_len'],
        gpu_memory_utilization=config['gpu_memory_utilization']
    )

    target_tokenizer = target_model.get_tokenizer()

    for i in range(0, len(counterfactuals), config["batch_size"]):
        # build batch
        print("\n----- Overthinking Generation Phase -----\n")
        batch = []
        for j in range(config["batch_size"]):
            if os.path.exists(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+j)}_overthink.txt") or os.path.exists(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+j)}_recurrent.txt") or os.path.exists(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+j)}_fail.txt"):
                continue

            print(f"\n{str(i+j)}: {counterfactuals[i+j]}\n")
            batch.append((j, counterfactuals[i+j]))

        # generate overtinkings
        inputs = [item for _, item in batch]
        idxs = [item for item, _ in batch]
        outputs = query('tokenize', inputs, config["temperature"], config["n"], config["max_model_len"]) # outputs: list[list[str]]
        batch_overthinkings = []
        for thinkings in outputs:
            overthinkings = []
            for thinking in thinkings:
                steps = thinking.split("\n\n")
                count = 0
                for step in steps:
                    for t in twist:
                        if step.startswith(t):
                            count += 1
                    if count >= 2:
                        break
                if count >= 2:
                    overthinkings.append(thinking)
            overthinkings = sorted(overthinkings, key=lambda s: len(target_tokenizer.encode(s)), reverse=True)
            batch_overthinkings.append(overthinkings)

        # 对 batch 中每一组输出
        for idx, tks in zip(idxs, batch_overthinkings):
            repeated = []
            # 判断是否有循环思考的输出
            for text in tks:
                print(f"\n[output {str(i+idx)}] {text[:50]} ... {text[-50:]}\n")
                steps = text.split("\n\n")
                test_step = steps[-2]
                prefix = steps[:-2]
                rev = prefix[::-1]
                if test_step in rev:
                    # 在倒序列表中的索引
                    idx_rev = rev.index(test_step)
                    # 换算成原列表的索引
                    index = len(prefix) - 1 - idx_rev
                    index_test = len(steps) - 1
                    distance = index_test - index
                    if distance > 1:
                        if prefix[index - distance] == test_step and steps[-3] == steps[index-1]:
                            repeated.append(text)
                    else:
                        if steps[-3] == test_step and steps[-4] == test_step and steps[-5] == test_step and steps[-6] == test_step and steps[-7] == test_step:
                            repeated.append(text)

            # 如果循环, 记录循环输出
            if repeated != []:
                with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+idx)}_recurrent.txt", "w") as f:
                    f.write(f"[recurrent thinking]\n{'\n[recurrent thinking]\n'.join(repeated)}")
                with open("/root/project/dos/log", "a") as f:
                    f.write(f"[INFO] [context length: {str(config["max_model_len"])}] question {str(i+idx)} generated recurrent output\n")

            # 否则记录over thinking
            else:
                if tks != []:
                    with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+idx)}_overthink.txt", "w") as f:
                        f.write(f"[overthinking]\n{'\n[overthinking]\n'.join(tks)}")
                else:
                    with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+idx)}_fail.txt", "w") as f:
                        pass
                    with open("/root/project/dos/log", "a") as f:
                        f.write(f"[INFO] question {str(i+idx)} failed\n")

        # 加载过度思考
        print("\n----- Overthinking Extention Phase -----\n")
        extention = []
        for j in range(config["batch_size"]):
            if os.path.exists(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+j)}_recurrent.txt"):
                continue

            try:
                with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+j)}_overthink.txt", "r") as f:
                    overthinkings = f.read().split(f"[overthinking]\n")
                    overthinking = overthinkings[1]
            except:
                continue
        
            thinking_steps = overthinking.split("\n\n")[:-1]

            # the second twist to the third twist
            count = 0
            for k in range(len(thinking_steps)):
                if "</think>" in thinking_steps[k]:
                    break

                for t in twist:
                    if thinking_steps[k].startswith(t):
                        count += 1
                        if count == 2:
                            p_1 = k

                if count == 3:
                    p_2 = k
                    break

            if count < 3:
                with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+j)}_fail.txt", "w") as f:
                    pass
                with open("/root/project/dos/log", "a") as f:
                    f.write(f"[INFO] question {str(i+j)} failed\n")
                continue

            overthink_steps = thinking_steps[p_1:p_2]
            temp = imitate_overthink(overthink_steps)

            # 各思考步骤相似性
            # similarities = []

            # for l in range(k+1, k+11):
            #     value = similarity_rate_count(thinking_steps[k], thinking_steps[l])
            #     similarities.append(value)

            # overthink_offset = similarities.index(max(similarities)) + 1
            # overthink_steps = thinking_steps[k:k+overthink_offset]
            # temp = imitate_overthink(overthink_steps)

            # 提取模式
            head_local = set()
            error_local = set()
            solution_local = set()

            for thinking in overthinkings:
                overthinking_steps = thinking.split("\n\n")
                for text in overthinking_steps:
                    ## [head]
                    match = re.match(r"^([A-Za-z]+),", text)
                    if match:
                        head_local.add(match.group(1))

                    ## [error]
                    pattern = re.compile(r"\b(I must|maybe I)[^.,]*", re.IGNORECASE)
                    m = pattern.search(text)
                    if m:
                        error_local.add(m.group(0))
                    ## [solution]
                    pattern = re.compile(r"\b(Let me)[^.,;:!?()\[\]\"'…]*", re.IGNORECASE)
                    match = pattern.match(text)
                    if match:
                        solution_local.add(match.group(0).strip())

            rewrited_overthinkings = []
            for _ in range(2):
                rewrited_overthinkings.append(overthinking_gen(temp))

            insertion = "\n\n".join(["\n\n".join([step for step in steps]) for steps in rewrited_overthinkings]) + "\n\n"
            original_thinking = "\n\n".join(thinking_steps[:k]) + "\n\n"
            input_str = combine_suffix(counterfactuals[i+j], original_thinking, insertion)
            if input_str != None:
                print(f"\nQuestion: {str(i+j)}\nThinking: {input_str}\n")
                extention.append((j, input_str))
            else:
                continue

        inputs = [item for _, item in extention]
        idxs = [item for item, _ in extention]
        outputs = query('direct', inputs, config["temperature"], config["n"], config["max_model_len"])
        extention_overthinkings = [[thinking for thinking in thinkings if "</think>" not in thinking] for thinkings in outputs] # 收集那些达到模型长度上限的输出

        for idx, tks in zip(idxs, extention_overthinkings):
            # 判断是否有循环思考的输出
            repeated = []
            for text in tks:
                print(f"\n[output {str(i+idx)}] {text[:50]} ... {text[-50:]}\n")
                steps = text.split("\n\n")
                test_step = steps[-2]
                prefix = steps[:-2]
                rev = prefix[::-1]
                if test_step in rev:
                    # 在倒序列表中的索引
                    idx_rev = rev.index(test_step)
                    # 换算成原列表的索引
                    index = len(prefix) - 1 - idx_rev
                    index_test = len(steps) - 1
                    distance = index_test - index
                    if distance > 1:
                        if prefix[index - distance] == test_step and steps[-3] == steps[index-1]:
                            repeated.append(text)
                    else:
                        if steps[-3] == test_step and steps[-4] == test_step and steps[-5] == test_step and steps[-6] == test_step and steps[-7] == test_step:
                            repeated.append(text)

            # 如果循环, 记录循环输出
            if repeated != []:
                with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+idx)}_recurrent.txt", "w") as f:
                    f.write(f"[recurrent thinking]\n{'\n[recurrent thinking]\n'.join(repeated)}")
                with open("/root/project/dos/log", "a") as f:
                    f.write(f"[INFO] [context length: {str(config["max_model_len"])}] question {str(i+idx)} generated recurrent output\n")

            # 如果达到长度上限但未循环，记录潜在的成功诱导循环的输入，用于更长长度测试
            else:
                if tks != []:
                    with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+idx)}_latent.txt", "w") as f:
                        f.write([input_str for index, input_str in extention if index == idx][0])
                    with open("/root/project/dos/log", "a") as f:
                        f.write(f"[INFO] [context length: {str(config["max_model_len"])}] question {str(i+idx)} has latent input\n")
                else:
                    with open(f"/root/project/dos/exp/8k_{config['models'][config['model_id']].split('/')[-1]}/{str(i+idx)}_fail.txt", "w") as f:
                        pass
                    with open("/root/project/dos/log", "a") as f:
                        f.write(f"[INFO] question {str(i+idx)} failed\n")