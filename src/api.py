import os
import json
import copy
from openai import OpenAI

# parameters
config = json.load(open('./config.json', 'r'))
temperature = config["temperature"]
max_model_len = config["max_model_len"]
max_reflect = config["max_reflect"]
logprobs = config["logprobs"]
model_name = "DeepSeek"
timeout = 60
api_key = config["api_key"]
reasoning_model = config["models"][5]
chat_model = config["models"][6]
client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com/beta")

# log

import logging

logger = logging.getLogger("dos_logger")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")


# 控制台
sh = logging.StreamHandler()
sh.setFormatter(fmt)

# 文件
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

            # 动态字段匹配
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


import re
from typing import Iterator

# 用于识别日志一条记录的开头（时间戳 + 竖线）
TS_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \|")

def iter_log_records(path: str) -> Iterator[str]:
    """
    按“逻辑日志条目”迭代，而不是按物理行。
    多行组成的一条日志会被合并成一个长字符串返回。
    """
    current = None

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if TS_PATTERN.match(line):
                # 开始新记录
                if current is not None:
                    yield current
                current = line.rstrip("\n")
            else:
                # 续行：拼接到上一条记录后面
                if current is None:
                    # 文件一开始就遇到孤立行，跳过
                    continue
                current += "\n" + line.rstrip("\n")

        # 文件结束，别忘最后一个
        if current is not None:
            yield current

from typing import Optional, Dict, Tuple

# 解析一般的 key=value（不含 token 那一段）
KV_PATTERN = re.compile(r"(\S+?)=([^\s]+)")

# 解析 token=... 和 prob/entropy=... 之间的 token 字符串
TOKEN_PATTERN = re.compile(
    r"token=(?P<token>.*?)(?= prob/entropy=|$)",
    re.DOTALL,
)

def parse_log_line(line: str) -> Tuple[Dict[str, str], Optional[str]]:
    """
    输入一条“逻辑日志行”（已合并多行），返回 (kv_dict, token_str)
    - kv_dict：除 token 外的所有 key=value（dataset/id/stage/state/num/prob/entropy 等）
    - token_str：token= 和 prob/entropy= 之间的原始字符串（可能含空格/换行）
    """
    token_str = None

    m_tok = TOKEN_PATTERN.search(line)
    if m_tok:
        token_str = m_tok.group("token")
        start, end = m_tok.span()
        cleaned = line[:start] + line[end:]   # 把 token=... 这一段去掉，剩下的再解析 kv
    else:
        cleaned = line

    kv = dict(KV_PATTERN.findall(cleaned))
    return kv, token_str

from typing import List, Tuple

def extract_token_strings_with_filters(
    log_file: str,
    *,
    num_min: Optional[int] = None,
    num_max: Optional[int] = None,
    dataset: Optional[str] = None,
    id_: Optional[int] = None,
    stage: Optional[str] = None,
    state: Optional[str] = None,
    strip_token: bool = False,
) -> List[Tuple[int, str]]:
    """
    从日志中提取 token 字符串（支持多行 token），并按条件筛选。
    返回 [(num, token_str), ...]，按日志出现顺序。
    """
    results: List[Tuple[int, str]] = []

    for record in iter_log_records(log_file):
        kv, token = parse_log_line(record)
        if token is None:
            continue

        # 处理 num
        if "num" not in kv:
            continue
        try:
            num = int(kv["num"])
        except ValueError:
            continue

        # num 区间筛选
        if num_min is not None and num < num_min:
            continue
        if num_max is not None and num > num_max:
            continue

        # 其它字段筛选
        if dataset is not None and kv.get("dataset") != dataset:
            continue

        if id_ is not None:
            try:
                if int(kv.get("id", -1)) != id_:
                    continue
            except ValueError:
                continue

        if stage is not None and kv.get("stage") != stage:
            continue

        if state is not None and kv.get("state") != state:
            continue

        token_use = token.strip() if strip_token else token
        results.append((num, token_use))

    return results


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

top_p = config["top-p"]
top_k = config["top-k"]
q = 1 - top_p  # tail quality
M = 128000
entropy_inaccuracy: float = -q * math.log(q) + q * math.log(M)  # tail entropy upper bound
bound = top_p/entropy_inaccuracy


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


def loop_check(text: str) -> bool:
    """
    检查输出是否产生循环
    
    :param text: 输出文本
    :type text: str
    :return: 检查结果
    :rtype: bool
    """

    # multi-line check
    multi_line_loop = False
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
                    multi_line_loop = True
            else: # 重复思考相邻，可能存在非循环情况，至少重复10次
                if all(steps[index] == steps[test_idx] for index in range(test_idx - 10, test_idx)):
                    multi_line_loop = True

    # one-line check
    one_line_loop = False
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
                else: # 至少重复100次
                    if all(last_step[index] == last_step[test_idx] for index in range(test_idx - 100, test_idx)):
                        one_line_loop = True

    return multi_line_loop or one_line_loop


def trunc_entropy_with_error_bounds_closed(
    step_logprobs: list[tuple[str, float]]
) -> float:
    """
    根据 step_logprobs 计算：
    - 不重新归一化的截断熵 H_trunc

    return:
        H_trunc
    """

    # ---------- 1. 从 logprobs 得到概率 ----------
    probs = []
    Hs = []
    prob_sum = 0
    H_trunc = 0.0
    
    for lp in step_logprobs:
        prob = math.exp(lp[1])
        probs.append(prob)
        h = prob * lp[1]
        Hs.append(h)
        prob_sum += prob
        H_trunc -= h

    return H_trunc


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


# main

if __name__ == "__main__":
    sub_dataset = [obj for obj in datasets.items()][:1]
    for dataset_name, dataset in sub_dataset:
        for id, data in enumerate(dataset):
            if os.path.exists(f"/root/project/dos/exp/{model_name}/{dataset_name}_{id}_reflect.jsonl"):
                over_reflect_flag = True
                with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_{id}_reflect.jsonl", "r") as f:
                    over_reflect_prompt = [json.loads(line) for line in f.readlines()]

            else:
                # generating over reflect
                over_reflect_flag = False
                for mode in MODE:
                    option = data["option"]
                    answer = data["answer"]
                    content = counterfactual_prompting(data, mode)
                    messages = [{"role":"user", "content":content}]

                    for _ in range(max_reflect):
                        # closed source
                        response = client.chat.completions.create(
                            model=reasoning_model,
                            messages=messages,
                            max_tokens=4096
                        )

                        text = response.choices[0].message.reasoning_content + response.choices[0].message.content

                        over_reflect_begin_idx = over_reflect_check(text, option, answer)
                        if over_reflect_begin_idx:
                            over_reflect_flag = True
                            break

                    if over_reflect_flag:
                        break

                if over_reflect_flag:
                    logger.info(f"dataset={dataset_name} id={id} stage=over_reflect_generation result=succeed")
                    over_reflect_steps = text.split("\n\n")[:over_reflect_begin_idx+1]
                    over_reflect_output = "\n\n".join(over_reflect_steps) + "\n\n"
                    over_reflect_prompt = messages.copy()
                    over_reflect_prompt.append({"role":"assistant", "content":"<think>\n"+over_reflect_output, "prefix":True})
                    with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_{id}_reflect.jsonl", "w") as f:
                        f.writelines([json.dumps(item) + "\n" for item in over_reflect_prompt])

            if over_reflect_flag:
                count = 0

                # 检索日志
                if exists_log_entry(f"/root/project/dos/log/{model_name}.log", dataset=dataset_name, id_=id, stage="entropy_descent_sampling", state="down"):
                    continue
                elif exists_log_entry(f"/root/project/dos/log/{model_name}.log", dataset=dataset_name, id_=id, stage="entropy_descent_sampling", state="running"):
                    # 恢复日志
                    pairs = extract_token_strings_with_filters(
                        f"/root/project/dos/log/{model_name}.log",
                        dataset=dataset_name,
                        id_=id,
                        stage="entropy_descent_sampling",
                        strip_token=False,   # 如果你想保留原来的前导/后缀空格，就设 False
                    )

                    # 按 num 排序，得到完整 token 序列
                    pairs_sorted = sorted(pairs, key=lambda x: x[0])
                    tokens = [t for _, t in pairs_sorted]
                    count = len(tokens)
                    generated_text = "".join(tokens)
                    over_reflect_prompt[1]["content"] += generated_text

                # 比较 top-k 个 token 后续预测的熵值
                # closed source
                response = client.chat.completions.create(
                    model=chat_model,
                    messages=over_reflect_prompt,
                    max_tokens=1,
                    logprobs=True,
                    top_logprobs=20,
                    temperature=1
                )

                candidate_token_logprobs = [(top_logprob.token,top_logprob.logprob) for top_logprob in sorted(response.choices[0].logprobs.content[0].top_logprobs, key=lambda x: x.logprob, reverse=True)[:top_k]] # sorted

                while(True):
                    PROB_1_ENTROPY_0 = False
                    PROB_1_ENTROPY = False
                    PROB_ENTROPY_0 = False
                    PROB_0_ENTROPY_0 = False
                    descent: list[tuple[str, float, list[tuple[str, float]]]] = []
                    for token, logprob in candidate_token_logprobs:
                        prob = math.exp(logprob)
                        over_reflect_prompt_temp = copy.deepcopy(over_reflect_prompt)
                        over_reflect_prompt_temp[1]["content"] += token

                        response = client.chat.completions.create(
                            model=chat_model,
                            messages=over_reflect_prompt_temp,
                            max_tokens=1,
                            logprobs=True,
                            top_logprobs=20,
                            temperature=1
                        )

                        if response.choices[0].logprobs != None:
                            next_step_logprobs = [(top_logprob.token,top_logprob.logprob) for top_logprob in sorted(response.choices[0].logprobs.content[0].top_logprobs, key=lambda x: x.logprob, reverse=True)] # sorted
                        else:
                            continue

                        entropy = trunc_entropy_with_error_bounds_closed(next_step_logprobs)
                        if prob == 1.0:
                            if entropy == 0.0:
                                PROB_1_ENTROPY_0 = True
                                next_token = token
                                next_prob_entropy = prob
                                candidate_token_logprobs = next_step_logprobs
                                break
                            else:
                                PROB_1_ENTROPY = True
                                descent.append((token, prob, entropy, prob/entropy, next_step_logprobs))
                        elif prob == 0.0:
                            if entropy == 0.0:
                                PROB_0_ENTROPY_0 = True
                                descent.append((token, prob, entropy, 1.0, next_step_logprobs))
                                if PROB_1_ENTROPY:
                                    break
                            else:
                                descent.append((token, 0.0, entropy, 0.0, next_step_logprobs))
                        else:
                            if entropy == 0.0:
                                PROB_ENTROPY_0 = True
                                next_token = token
                                next_prob_entropy = prob
                                candidate_token_logprobs = next_step_logprobs
                                break
                            else:
                                descent.append((token, prob, entropy, prob/entropy, next_step_logprobs))

                    if PROB_1_ENTROPY_0 or PROB_ENTROPY_0:
                        pass
                    else:
                        if PROB_1_ENTROPY and PROB_0_ENTROPY_0:
                            response = client.chat.completions.create(
                                model=chat_model,
                                messages=over_reflect_prompt,
                                max_tokens=1,
                                temperature=1
                            )
                            next_token = response.choices[0].message.content
                            over_reflect_prompt_temp = copy.deepcopy(over_reflect_prompt)
                            over_reflect_prompt_temp[1]["content"] += next_token

                            response = client.chat.completions.create(
                                model=chat_model,
                                messages=over_reflect_prompt_temp,
                                max_tokens=1,
                                logprobs=True,
                                top_logprobs=20,
                                temperature=1
                            )

                            if response.choices[0].finish_reason != "stop" and response.choices[0].logprobs != None:
                                candidate_token_logprobs = [(top_logprob.token,top_logprob.logprob) for top_logprob in sorted(response.choices[0].logprobs.content[0].top_logprobs, key=lambda x: x.logprob, reverse=True)][:top_k] # sorted
                            else:
                                logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=eos")
                                break

                            next_prob_entropy = 1.0

                        else:
                            descent.sort(key=lambda x: x[3], reverse=True)
                            next_token = descent[0][0]
                            next_prob_entropy = descent[0][3]
                            candidate_token_logprobs = descent[0][4][:top_k]

                    over_reflect_prompt[1]["content"] += next_token
                    count += 1

                    logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=running num={count} token={next_token} prob/entropy={next_prob_entropy}")

                    if next_token == "<｜end▁of▁sentence｜>" or over_reflect_prompt[1]["content"].endswith("</think>\n\n"):
                        logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=eos")
                        break

                    elif loop_check(over_reflect_prompt[1]["content"]):
                        logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=True stop_reason=loop")
                        break

                    elif count >= 2048:
                        logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=length_limit")
                        break

                with open(f"/root/project/dos/exp/{model_name}/{dataset_name}_{id}_entropy_descent.jsonl", "w") as f:
                    f.writelines([json.dumps(item) + "\n" for item in over_reflect_prompt])

            else:
                logger.info(f"dataset={dataset_name} id={id} stage=over_reflect_generation state=failed")