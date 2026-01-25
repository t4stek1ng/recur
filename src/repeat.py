import json
import os
from vllm import LLM, SamplingParams, RequestOutput

os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
temperature = 0
model = "DeepSeek-R1-Distill-Llama-8B"
times = 3

# model (97.5% = 30 * 16k)
llm = LLM(
    model=f"/root/project/models/{model}",
    task="generate",
    max_model_len=16384,
    gpu_memory_utilization=0.95,
    enable_chunked_prefill=False
)

tokenizer = llm.get_tokenizer()
eot_token_id = tokenizer.encode("</think>")[0]


# sampling parameters

# generate over-reflect
sampling_params = SamplingParams(
    temperature=temperature,
    max_tokens=16384
)

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
            else: # 重复思考相邻，可能存在非循环情况，至少重复 n 次
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

if __name__ == "__main__":
    with open(f"/root/project/dos/exp/{model}/promtps.jsonl", "r") as f:
        datas = [json.loads(line) for line in f.readlines()]
        prompts = [data for data in datas if data["generation"] == False]

    ids = [4, 8, 6, 0]


    for data in prompts:
        id = data["id"]
        if id not in ids:
            continue

        length = 0
        loop_flag = False
        for _ in range(times):
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
                loop_flag = True
            length += len(output.outputs[0].token_ids)

        with open(f"/root/project/dos/exp/{model}/gsm8k_results.jsonl", "a") as f:
            f.write(json.dumps({"id": id, "loop": loop_flag, "temperature": temperature, "times":times, "avg": int(length/times), "trim": False})+"\n")

    # print(f"avg length: {int(length/len(prompts))}")
    # with open(f"/root/project/dos/exp/{model}/gsm8k_results.jsonl", "a") as f:
    #     f.write(json.dumps({"id": loop_ids, "loop": len(loop_ids), "temperature": temperature, "avg": int(length/len(prompts)), "max": max(lengths), "trim": False})+"\n")