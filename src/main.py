import os
import json
import time
from vllm import LLM, SamplingParams, RequestOutput


# configurations
config = json.load(open('./config.json', 'r'))
os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']


# parameters
temperature = config["temperature"]
max_model_len = config["max_model_len"]
max_reflect = config["max_reflect"]
logprobs = config["logprobs"]
model_name = config['model']
model_path = f"./models/{model_name}"
timeout = 100


# log

import logging

logger = logging.getLogger("recur_logger")
logger.setLevel(logging.INFO)

fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

# console
sh = logging.StreamHandler()
sh.setFormatter(fmt)

# file
fh = logging.FileHandler(f"./log/{model_name}.log", encoding="utf-8")
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
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            kv = dict(KV_PATTERN.findall(line))
            if not kv:
                continue

            if dataset is not None and kv.get("dataset") != dataset:
                continue
            if id_ is not None and int(kv.get("id", -1)) != id_:
                continue
            if stage is not None and kv.get("stage") != stage:
                continue
            if state is not None and kv.get("state") != state:
                continue

            return True

    return False


# model
llm = LLM(
    model=model_path,
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

# recursive entropy guided sampling
prob_entropy_min_params = SamplingParams(
    temperature=1,
    max_tokens=1,
    logprobs=logprobs,
)

# normal sampling
loop_generation_params = SamplingParams(
    temperature=temperature,
    max_tokens=max_model_len
)

# functions

# counterfactual questions
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


def trunc_entropy_with_error_bounds(
    step_logprobs: dict[int, Logprob],
    top_p: float
) -> float:
    probs = []
    prob_sum = 0
    
    for lp in step_logprobs.values():
        prob = math.exp(lp.logprob)
        probs.append(prob)
        prob_sum += prob
        
        if prob_sum > top_p:
            break

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
            elif reflect and (f" {answer} " in step or f" {answer}." in step or f" {answer}," in step or f" {answer}\n" in step) and i > reflect_done_idx + 1:
                over_reflect_begin_idx = i
                return over_reflect_begin_idx
            elif think and (f" {answer} " in step or f" {answer}." in step or f" {answer}," in step or f" {answer}\n" in step) and i > think_done_idx + 1:
                reflect = True # reflect done
                reflect_done_idx = i
            elif (f" {answer} " in step or f" {answer}." in step or f" {answer}," in step or f" {answer}\n" in step) and i > 1:
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
    # multi-line check
    multi_line_loop = False
    steps = text.split("\n\n")[:-1]
    test_step = steps[-1]
    rev = steps[-2::-1]
    if test_step in rev:
        idx_rev = rev.index(test_step)
        same_idx = len(rev) - 1 - idx_rev
        test_idx = len(steps) - 1
        distance = test_idx - same_idx
        if same_idx > distance:
            if distance > 1:
                if all(steps[index] == steps[index - distance] for index in range(same_idx, test_idx)):
                    multi_line_loop = True
            else:
                if all(steps[index] == steps[test_idx] for index in range(test_idx - n, test_idx)):
                    multi_line_loop = True

    # one-line check
    one_line_loop = False
    last_step = text.split("\n\n")[-1]
    if len(last_step) > 100:
        test_str = last_step[-1]
        rev = last_step[-2::-1]
        if test_str in rev:
            idx_rev = rev.index(test_str)
            same_idx = len(rev) - 1 - idx_rev
            test_idx = len(last_step) - 1
            distance = test_idx - same_idx
            if same_idx > distance:
                if distance > 1:
                    if all(last_step[index] == last_step[index - distance] for index in range(same_idx, test_idx)) and all(last_step[index] == last_step[test_idx] for index in range(test_idx - 10*distance, test_idx, distance)):
                        one_line_loop = True
                else:
                    if all(last_step[index] == last_step[test_idx] for index in range(test_idx - 10 * n, test_idx)):
                        one_line_loop = True

    return multi_line_loop or one_line_loop


# dataset

import json

# gsm8k

with open("./dataset/ours/gsm8k_sample.jsonl", "r") as f:
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

datasets = {"gsm8k":gsm8k_dataset}


# main

if __name__ == "__main__":
    if os.path.exists(f"./exp/{model_name}"):
        pass
    else:
        os.mkdir(f"./exp/{model_name}")

    sub_dataset = [obj for obj in datasets.items()][:1]
    for dataset_name, dataset in sub_dataset:
        for id, data in enumerate(dataset):
            LOOP = False
            if os.path.exists(f"./exp/{model_name}/{dataset_name}_{id}_reflect.json"):
                with open(f"./exp/{model_name}/{dataset_name}_{id}_reflect.json", "r") as f:
                    info = json.loads(f.read())
                    over_reflect_flag = info["over_reflect"]
                    over_reflect_prompt = info["prompt"]
            else:
                # generating over reflect
                over_reflect_flag = False
                for mode in MODE:
                    option = data["option"]
                    answer = data["answer"]
                    content = counterfactual_prompting(data, mode)
                    prompt = tokenizer.apply_chat_template(
                        [{"role":"user", "content":content}],
                        tokenize=False,
                        add_generation_prompt=True
                    )

                    for _ in range(max_reflect):
                        outputs: list[RequestOutput] = llm.generate(
                            prompt,
                            over_reflect_params,
                            use_tqdm=False
                        )

                        output = outputs[0]
                        text = output.outputs[0].text
                        over_reflect_begin_idx = over_reflect_check(text, option, answer)
                        if over_reflect_begin_idx:
                            over_reflect_flag = True
                            break

                    if over_reflect_flag:
                        break

                input_length = len(tokenizer.encode(prompt))
                output_length = len(tokenizer.encode(text))
                prompt_length = input_length + output_length
                if over_reflect_flag:
                    logger.info(f"dataset={dataset_name} id={id} stage=over_reflect_generation result=succeed")
                    over_reflect_steps = text.split("\n\n")[:over_reflect_begin_idx+1]
                    over_reflect_output = "\n\n".join(over_reflect_steps) + "\n\n"
                    over_reflect_prompt = prompt + over_reflect_output

                    with open(f"./exp/{model_name}/{dataset_name}_{id}_reflect.json", "w") as f:
                        f.write(json.dumps({
                            "over_reflect": True,
                            "input_length": input_length,
                            "output_length": output_length,
                            "prompt_length": prompt_length,
                            "prompt": over_reflect_prompt
                        }))
                else:
                    logger.info(f"dataset={dataset_name} id={id} stage=over_reflect_generation result=failed")

                    with open(f"./exp/{model_name}/{dataset_name}_{id}_reflect.json", "w") as f:
                        f.write(json.dumps({
                            "over_reflect": False,
                            "input_length": input_length,
                            "output_length": output_length,
                            "prompt_length": prompt_length,
                            "prompt": prompt + text
                        }))

            if over_reflect_flag:
                count = 0

                # retrieve log
                if exists_log_entry(f"./log/{model_name}.log", dataset=dataset_name, id_=id, stage="entropy_descent_sampling", state="down"):
                    continue
                elif exists_log_entry(f"./log/{model_name}.log", dataset=dataset_name, id_=id, stage="entropy_descent_sampling", state="running"):
                    # restore from log
                    additional_token_ids = extract_tokens_with_kv_filters(f"./log/{model_name}.log", filters={"dataset":dataset_name, "id":id, "stage":"entropy_descent_sampling"})
                    count = len(additional_token_ids)
                    additional_prompt = tokenizer.decode(additional_token_ids)
                    over_reflect_prompt += additional_prompt

                prompt_length = len(tokenizer.encode(over_reflect_prompt))


                outputs = llm.generate(
                    over_reflect_prompt,
                    prob_entropy_min_params,
                    use_tqdm=False
                )

                candidate_token_logprobs = sorted(outputs[0].outputs[0].logprobs[0].items(), key=lambda x: x[1].rank)[:top_k]


                while(True):
                    descent: list[tuple[int, str, float, dict[int, Logprob]]] = []
                    for token_id, logprob in candidate_token_logprobs:
                        token = logprob.decoded_token
                        prob = math.exp(logprob.logprob)
                        over_reflect_prompt_temp = over_reflect_prompt + token

                        t0 = time.perf_counter()
                        outputs = llm.generate(
                            over_reflect_prompt_temp,
                            prob_entropy_min_params,
                            use_tqdm=False
                        )
                        dt = time.perf_counter() - t0

                        next_step_logprobs = outputs[0].outputs[0].logprobs[0]
                        entropy = trunc_entropy_with_error_bounds(next_step_logprobs, 0.99)
                        descent.append((token_id, token, prob/entropy, next_step_logprobs, prob, entropy))

                    descent.sort(key=lambda x: x[2], reverse=True)
                    next_token_id = descent[0][0]
                    next_token = descent[0][1]
                    over_reflect_prompt += next_token
                    next_prob_entropy = descent[0][2]
                    prob = descent[0][4]
                    entropy = descent[0][5]
                    count += 1
                    prompt_length += 1
                    logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=running num={count} token_id={next_token_id} prob={prob} entropy={entropy} prob/entropy={next_prob_entropy}")

                    candidate_token_logprobs = sorted(descent[0][3].items(), key=lambda x: x[1].rank)[:top_k]
                    if next_token_id == tokenizer.eos_token_id or next_token_id == eot_token_id:
                        logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=eos")
                        stop_reason = "eos"
                        break

                    elif loop_check(over_reflect_prompt, 10):
                        logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=True stop_reason=loop")
                        stop_reason = "loop"
                        LOOP = True
                        break

                    elif dt > timeout:
                        logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=slow")
                        stop_reason = "slow"
                        break

                    elif prompt_length >= 4096:
                        if loop_check(over_reflect_prompt, 5):
                            logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=True stop_reason=length_limit")
                            LOOP = True
                        else:
                            logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=length_limit")
                        stop_reason = "length_limit"
                        break

                    elif count >= 2048:
                        if loop_check(over_reflect_prompt, 5):
                            logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=True stop_reason=count_limit")
                            LOOP = True
                        else:
                            logger.info(f"dataset={dataset_name} id={id} stage=entropy_descent_sampling state=down loop=False stop_reason=count_limit")
                        stop_reason = "count_limit"
                        break

                with open(f"./exp/{model_name}/{dataset_name}_{id}_entropy_descent.json", "w") as f:
                    f.write(json.dumps({
                        "loop": LOOP,
                        "stop_reason": stop_reason,
                        "prompt": over_reflect_prompt
                    }))