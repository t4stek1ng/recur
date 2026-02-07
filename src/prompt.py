import json
import os
from vllm import LLM, SamplingParams, RequestOutput

# configurations
config = json.load(open('./config.json', 'r'))
os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = config['gpu']


# parameters
temperature = 0
max_model_len = config["max_model_len"]
model_name = config['model']
model_path = f"./models/{model_name}"

# mode
llm = LLM(
    model=model_path,
    task="generate",
    max_model_len=max_model_len,
    gpu_memory_utilization=config['gpu_memory_utilization'],
    enable_chunked_prefill=False
)

tokenizer = llm.get_tokenizer()
eot_token_id = tokenizer.encode("</think>")[0]


# sampling parameters
sampling_params = SamplingParams(
    temperature=temperature,
    max_tokens=16384
)

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

if __name__ == "__main__":
    with open(f"./exp/{model_name}/promtps.jsonl", "r") as f:
        datas = [json.loads(line) for line in f.readlines()]
        prompts = [data for data in datas if data["generation"] == False]

    if os.path.exists(f"./exp/{model_name}/gsm8k_results.jsonl"):
        pass
    else:
        with open(f"./exp/{model_name}/gsm8k_results.jsonl", "w") as f:
            pass

    length = 0
    lengths = []
    loop_ids = []
    for data in prompts:
        id = data["id"]
        prompt = tokenizer.apply_chat_template(
            [{"role":"user", "content":data["prompt"]}],
            tokenize=False,
            add_generation_prompt=True
        )

        outputs: list[RequestOutput] = llm.generate(
            prompt,
            sampling_params,
            use_tqdm=False
        )

        output = outputs[0]
        text = output.outputs[0].text
        loop = loop_check(text, 5)
        if loop:
            loop_ids.append(id)
        length += len(output.outputs[0].token_ids)
        lengths.append(len(output.outputs[0].token_ids))
        with open(f"./exp/{model_name}/gsm8k_results.jsonl", "a") as f:
            f.write(json.dumps({"id": id, "loop": loop, "temperature": temperature, "output": len(output.outputs[0].token_ids)})+"\n")

    with open(f"./exp/{model_name}/gsm8k_results.jsonl", "a") as f:
        f.write(json.dumps({"id": loop_ids, "loop": len(loop_ids), "temperature": temperature, "avg": int(length/len(prompts)), "max": max(lengths)})+"\n")