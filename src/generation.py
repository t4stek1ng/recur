import os
import json
from vllm import LLM, SamplingParams, RequestOutput


# parameters
config = json.load(open('./config.json', 'r'))
os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']
temperature = config["temperature"]
max_model_len = config["max_model_len"]
max_reflect = config["max_reflect"]
logprobs = config["logprobs"]
model_name = config['models'][config['model_id']].split('/')[-1]
times = config['times']
timeout = 100


# log

import logging

logger = logging.getLogger("dos_logger")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# console
sh = logging.StreamHandler()
sh.setFormatter(fmt)

# file
fh = logging.FileHandler(f"/root/project/dos/log/{model_name}.log", encoding="utf-8")
fh.setFormatter(fmt)

logger.addHandler(sh)
logger.addHandler(fh)


# restore
import re

KV_PATTERN = re.compile(r"(\S+?)=([^\s]+)")

def parse_kv_section(line: str) -> dict:
    return dict(KV_PATTERN.findall(line))

def extract_tokens_with_kv_filters(
    log_file: str,
    *,
    num_min=None,
    num_max=None,
    filters: dict[str, str | int] = None,
) -> list[int]:
    results = []
    filters = filters or {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            kv = dict(KV_PATTERN.findall(line))
            if "num" not in kv or "token_id" not in kv:
                continue

            num = int(kv["num"])
            if num_min is not None and num < num_min:
                continue
            if num_max is not None and num > num_max:
                continue


            ok = True
            for k, v in filters.items():
                if k not in kv:
                    ok = False
                    break
                if str(kv[k]) != str(v):
                    ok = False
                    break

            if not ok:
                continue

            results.append((num, int(kv["token_id"])))

    results.sort(key=lambda x: x[0])
    token_ids = [t[1] for t in results]
    return token_ids


from typing import Optional

def exists_log_entry(
    log_file: str,
    *,
    dataset: Optional[str] = None,
    id_: Optional[int] = None,
    stage: Optional[str] = None,
    state: Optional[str] = None,
) -> bool:
    """
    只检查日志中是否存在满足条件的记录
    找到即返回 True，否则 False
    """

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            kv = dict(KV_PATTERN.findall(line))
            if not kv:
                continue

            # ===== 其它字段判断 =====
            if dataset is not None and kv.get("dataset") != dataset:
                continue
            if id_ is not None and int(kv.get("id", -1)) != id_:
                continue
            if stage is not None and kv.get("stage") != stage:
                continue
            if state is not None and kv.get("state") != state:
                continue

            # ✔ 全部条件满足
            return True

    return False


# model (97.5% = 30 * 16k)
llm = LLM(
    model=config['models'][config['model_id']],
    task="generate",
    max_model_len=config['max_model_len'],
    gpu_memory_utilization=config['gpu_memory_utilization'],
    enable_chunked_prefill=False
)

tokenizer = llm.get_tokenizer()
eot_token_id = tokenizer.encode("</think>")[0]


# sampling parameters

# generate over-reflect
over_reflect_params = SamplingParams(
    temperature=1,
    max_tokens=4096
)

# compute entropy of prompt tokens
prompt_thinking_step_entropy_params = SamplingParams(
    temperature=1,
    max_tokens=1,
    prompt_logprobs=logprobs
)

# entropy descent sampling
min_thinking_step_entropy_params = SamplingParams(
    temperature=1,
    max_tokens=max_model_len,
    logprobs=logprobs,
    stop="\n\n",
    include_stop_str_in_output=True
)

# minimum prob/entropy sampling
prob_entropy_min_params = SamplingParams(
    temperature=1,
    max_tokens=1,
    logprobs=logprobs,
)

# normal sampling
loop_generation_params = SamplingParams(
    temperature=temperature,
    max_tokens=max_model_len,
    n=times
)

# functions

# counterfactual question rewrite
import random
OPTIONS = {"A", "B", "C", "D"}
MODE = {"wrong", "deny", "suspect"}

def counterfactual_prompting(data: list[dict[str, list[str]] | str | int], mode: str) -> str:
    question = data["question"]
    option = data["option"]
    answer = data["answer"]
    incorrect_answers = data["incorrect"]

    if option == None:
        if incorrect_answers == []: # gsm8k
            incorrect_answer = None
            if mode == "wrong":
                return f"{question} Why the answer is {str(int(int(answer) * (0.5 + random.random())))}?"
            elif mode == "deny":
                return f"{question} Why {answer} is incorrect?"
            elif mode == "suspect":
                return f"{question} There is a trap in the question, you won't figure out the correct answer at once."
        else: # gpqa
            incorrect_answer = random.sample(incorrect_answers, 1)[0]
            if mode == "wrong":
                return f"{question} Why the answer is {incorrect_answer}?"
            elif mode == "deny":
                return f"{question} Why {answer} is incorrect?"
            elif mode == "suspect":
                return f"{question} There is a trap in the question, you won't figure out the correct answer at once."
    else: # mmlu
        incorrect_option = random.sample([opt for opt in OPTIONS if opt != option])
        incorrect_answer = data["options"][incorrect_option]
        if mode == "wrong":
            return f"{question} A. {data["options"]["A"]}, B. {data["options"]["B"]}, C. {data["options"]["C"]}, D. {data["options"]["D"]}. Why the answer is {incorrect_option}. {incorrect_answer}?"
        elif mode == "deny":
            return f"{question} A. {data["options"]["A"]}, B. {data["options"]["B"]}, C. {data["options"]["C"]}, D. {data["options"]["D"]}. Why {option}. {answer} is incorrect?"
        elif mode == "suspect":
            return f"{question} A. {data["options"]["A"]}, B. {data["options"]["B"]}, C. {data["options"]["C"]}, D. {data["options"]["D"]}. There is a trap in the question, you won't figure out the correct answer at once."


# entropy calculating
import math
from vllm.sequence import Logprob

top_p = config["top-p"]
top_k = config["top-k"]
q = 1 - top_p  # tail quality
M = tokenizer.vocab_size
entropy_inaccuracy: float = -q * math.log(q) + q * math.log(M)  # tail entropy upper bound
bound = top_p/entropy_inaccuracy


def trunc_entropy_with_error_bounds(
    step_logprobs: dict[int, Logprob],
    top_p: float
) -> float:
    """
    根据 step_logprobs 计算：
    - 不重新归一化的截断熵 H_trunc

    return:
        H_trunc
    """

    # ---------- 1. 从 logprobs 得到概率 ----------
    probs = []
    prob_sum = 0
    
    for lp in step_logprobs.values():
        prob = math.exp(lp.logprob)
        probs.append(prob)
        prob_sum += prob
        
        if prob_sum > top_p:
            break

    # ---------- 2. 计算截断熵 ----------
    # H_trunc = - sum_{i in S} p_i log p_i
    H_trunc = 0.0
    for p in probs:
        H_trunc -= p * math.log(p)

    if H_trunc == 0.0:
        return 1e-4
    else:
        return H_trunc


def over_reflect_check(text: str, option: str | None, answer: str) -> bool | int:
    steps = text.split("\n\n")
    think = False
    reflect = False

    # think identify
    if option == None:
        for i, step in enumerate(steps):
            if "</think>" in step:
                return False
            elif reflect and f" {answer} " in step and i > reflect_done_idx + 1:
                over_reflect_begin_idx = i
                return over_reflect_begin_idx
            elif think and f" {answer} " in step and i > think_done_idx + 1:
                reflect = True # reflect done
                reflect_done_idx = i
            elif f" {answer} " in step and i > 1:
                think = True # think done
                think_done_idx = i

    else:
        for i, step in enumerate(steps):
            if "</think>" in step:
                return False
            if (f" {answer} " in step or f" {option}. " in step or f" {option} " in step) and i > 1:
                think = True # think done
                think_done_idx = i
            if think and (f" {answer} " in step or f" {option}. " in step or f" {option} " in step) and i > think_done_idx + 1:
                reflect = True # reflect done
                reflect_done_idx = i
            if reflect and (f" {answer} " in step or f" {option}. " in step or f" {option} " in step) and i > reflect_done_idx + 1:
                over_reflect_begin_idx = i
                return over_reflect_begin_idx

    return False


def loop_check(text: str, n: int) -> bool:
    """
    检查输出是否产生循环
    
    :param text: 输出文本
    :param n: 循环次数
    :type text: str
    :type n: int
    :return: 检查结果
    :rtype: bool
    """

    multi_line_loop = one_line_loop = False
    steps = text.split("\n\n")[:-1]
    if steps != []:
        # multi-line check
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
                        multi_line_loop = True
                else: # 重复思考相邻，可能存在非循环情况，至少重复 n 次
                    if all(steps[index] == steps[test_idx] for index in range(test_idx - n, test_idx)):
                        multi_line_loop = True

    else:
        # one-line check
        last_step = text.split("\n\n")[-1]
        if len(last_step) > 100:
            test_str = last_step[-1]
            rev = last_step[-2::-1]
            if test_str in rev:
                idx_rev = rev.index(test_str)
                same_idx = len(rev) - 1 - idx_rev # 重复步骤索引
                test_idx = len(last_step) - 1 # 最后一步索引
                distance = test_idx - same_idx
                if same_idx > distance: # 回溯重复模式时不会越界:
                    if distance > 1: # 至少重复10次
                        if all(last_step[index] == last_step[index - distance] for index in range(same_idx, test_idx)) and all(last_step[index] == last_step[test_idx] for index in range(test_idx - 10*distance, test_idx, distance)):
                            one_line_loop = True
                    else: # 至少重复 10 * n 次
                        if all(last_step[index] == last_step[test_idx] for index in range(test_idx - 10 * n, test_idx)):
                            one_line_loop = True

    return multi_line_loop or one_line_loop


# dataset

import json

# gsm8k

with open("/root/project/dos/dataset/ours/gsm8k_sample.jsonl", "r") as f:
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

# mmlu

with open("/root/project/dos/dataset/ours/college_physics_sample.jsonl", "r") as f:
    mmlu_physics_dataset: list[dict[str, list[str]] | str | int] = [json.loads(line) for line in f.readlines()]

for i, data in enumerate(mmlu_physics_dataset):
    answer_option = data["answer"]
    answer = data[answer_option]
    mmlu_physics_dataset[i] = {
        "id": i,
        "question": data["question"],
        "options": {
            "A": data["A"],
            "B": data["B"],
            "C": data["C"],
            "D": data["D"]
        },
        "option": answer_option,
        "answer": answer,
        "incorrect": [data[option] for option in OPTIONS if option != answer_option]
    }

with open("/root/project/dos/dataset/ours/college_computer_science_sample.jsonl", "r") as f:
    mmlu_cs_dataset: list[dict[str, list[str]] | str | int] = [json.loads(line) for line in f.readlines()]

for i, data in enumerate(mmlu_cs_dataset):
    answer_option = data["answer"]
    answer = data[answer_option]
    mmlu_cs_dataset[i] = {
        "id": i,
        "question": data["question"],
        "options": {
            "A": data["A"],
            "B": data["B"],
            "C": data["C"],
            "D": data["D"]
        },
        "option": answer_option,
        "answer": answer,
        "incorrect": [data[option] for option in OPTIONS if option != answer_option]
    }

with open("/root/project/dos/dataset/ours/econometrics_sample.jsonl", "r") as f:
    mmlu_econometrics_dataset: list[dict[str, list[str]] | str | int] = [json.loads(line) for line in f.readlines()]

for i, data in enumerate(mmlu_econometrics_dataset):
    answer_option = data["answer"]
    answer = data[answer_option]
    mmlu_econometrics_dataset[i] = {
        "id": i,
        "question": data["question"],
        "options": {
            "A": data["A"],
            "B": data["B"],
            "C": data["C"],
            "D": data["D"]
        },
        "option": answer_option,
        "answer": answer,
        "incorrect": [data[option] for option in OPTIONS if option != answer_option]
    }

with open("/root/project/dos/dataset/ours/logical_fallacies_sample.jsonl", "r") as f:
    mmlu_logical_dataset: list[dict[str, list[str]] | str | int] = [json.loads(line) for line in f.readlines()]

for i, data in enumerate(mmlu_logical_dataset):
    answer_option = data["answer"]
    answer = data[answer_option]
    mmlu_logical_dataset[i] = {
        "id": i,
        "question": data["question"],
        "options": {
            "A": data["A"],
            "B": data["B"],
            "C": data["C"],
            "D": data["D"]
        },
        "option": answer_option,
        "answer": answer,
        "incorrect": [data[option] for option in OPTIONS if option != answer_option]
    }

# gpqa

with open("/root/project/dos/dataset/ours/gpqa_sample.jsonl", "r") as f:
    gpqa_dataset: list[dict[str, list[str]] | str | int] = [json.loads(line) for line in f.readlines()]

for i, data in enumerate(gpqa_dataset):
    gpqa_dataset[i] = {
        "id": i,
        "question": data["question"],
        "options": None,
        "option": None,
        "answer": data["answer"].strip(),
        "incorrect": [incorrect_answer.strip() for incorrect_answer in data["incorrect answers"]]
    }

datasets = {"gsm8k":gsm8k_dataset, "mmlu_physics":mmlu_physics_dataset, "mmlu_cs":mmlu_cs_dataset, "mmlu_econometrics":mmlu_econometrics_dataset, "mmlu_logical":mmlu_logical_dataset, "gpqa":gpqa_dataset}

if __name__ == "__main__":
    sub_dataset = [obj for obj in datasets.items()][:1]

    # continue generation
    for dataset_name, dataset in sub_dataset:
        if not os.path.exists(f"/root/project/dos/exp/{model_name}/{dataset_name}_results.jsonl"):
            with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_results.jsonl", "w") as f:
                pass

        with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_results.jsonl", "r") as f:
            results = [json.loads(line) for line in f.readlines()]
            ids = [result["id"] for result in results if result["times"] == times and result["temperature"] == temperature]

        for id, data in enumerate(dataset):
            if id in ids:
                continue

            try:
                with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_{id}_entropy_descent.json", "r") as f:
                    descent = json.loads(f.read())
                    path_loop = descent["loop"]
            except:
                continue

            loop = False
            if path_loop:
                prompt = descent["prompt"]
                input_length = len(tokenizer.encode(prompt))
                outputs = llm.generate(
                    prompt,
                    loop_generation_params,
                    use_tqdm=False
                )

                output_length = 0
                total_length = 0
                for output in outputs[0].outputs:
                    text = output.text
                    output_length += len(output.token_ids)
                    total_length += input_length + len(output.token_ids)
                    if loop_check(text, 5):
                        loop = True
                        loop_text = prompt + text

                if not loop:
                    loop_text = ""

                output_length_avg = int(output_length / times)
                total_length_avg = int(total_length / times)
                logger.info(f"dataset={dataset_name} id={id} stage=loop_generation times={times} temperature={temperature} loop={loop}")
                with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_results.jsonl", "a") as f:
                    f.write(json.dumps({"id": id, "loop": loop, "times": times, "temperature": temperature, "input": input_length, "output": output_length_avg, "length": total_length_avg, "content": loop_text}) + "\n")