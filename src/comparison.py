import os
import json

comparisons = os.listdir("/root/project/dos/comparison")

from vllm import LLM, SamplingParams, RequestOutput

os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
temperature = 0
model = "QwQ-32B"

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

for comparison in comparisons:
    if os.path.exists(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl"):
        pass
    else:
        with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "w") as f:
            pass

    # if comparison == "LoopLLM":
    #     files = os.listdir(f"/root/project/dos/comparison/{comparison}")[:20]
    #     prompts = []
    #     for file in files:
    #         with open(f"/root/project/dos/comparison/{comparison}/{file}", "r") as f:
    #             data: dict = json.loads(f.read())
    #             result = [value for _, value in data.items()][-1]
    #             prompt = result["adv_prompt"]
            
    #         prompts.append(prompt)

    #     length = 0
    #     lengths = []
    #     for id, content in enumerate(prompts):
    #         prompt = tokenizer.apply_chat_template(
    #             [{"role":"user", "content":content}],
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )

    #         outputs: list[RequestOutput] = llm.generate(
    #             prompt,
    #             sampling_params,
    #             use_tqdm=False
    #         )

    #         output = outputs[0]
    #         text = output.outputs[0].text
    #         length += len(output.outputs[0].token_ids)
    #         lengths.append(len(output.outputs[0].token_ids))
    #         with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
    #                 f.write(json.dumps({"id": id, "temperature": temperature, "output": len(output.outputs[0].token_ids), "content": text})+"\n")

    #     avg = length/len(prompts)
    #     with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
    #         f.write(json.dumps({"temperature": temperature, "avg": int(avg), "max": max(lengths)})+"\n")

    # if comparison == "AutoDoS":
    #     with open(f"/root/project/dos/comparison/AutoDoS/gpt-4o_Meta-Llama-3.1-8B_10_subtask.json", "r") as f:
    #         prompts = eval(f.read())

    #     length = 0
    #     lengths = []
    #     for id, content in enumerate(prompts):
    #         prompt = tokenizer.apply_chat_template(
    #             [{"role":"user", "content":content}],
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )

    #         outputs: list[RequestOutput] = llm.generate(
    #             prompt,
    #             sampling_params,
    #             use_tqdm=False
    #         )

    #         output = outputs[0]
    #         text = output.outputs[0].text
    #         length += len(output.outputs[0].token_ids)
    #         lengths.append(len(output.outputs[0].token_ids))
    #         with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
    #                 f.write(json.dumps({"id": id, "temperature": temperature, "output": len(output.outputs[0].token_ids), "content": text})+"\n")

    #     avg = length/len(prompts)
    #     with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
    #         f.write(json.dumps({"temperature": temperature, "avg": int(avg), "max": max(lengths)})+"\n")

    # if comparison == "GCG":
    #     prompts = []
    #     files = os.listdir(f"/root/project/dos/comparison/{comparison}")
    #     for file in files:
    #         with open(f"/root/project/dos/comparison/{comparison}/{file}", "r") as f:
    #             prompt = json.loads(f.read())["prompt"]
    #             prompts.append(prompt)

    #     length = 0
    #     lengths = []
    #     for id, content in enumerate(prompts):
    #         prompt = tokenizer.apply_chat_template(
    #             [{"role":"user", "content":content}],
    #             tokenize=False,
    #             add_generation_prompt=True
    #         )

    #         outputs: list[RequestOutput] = llm.generate(
    #             prompt,
    #             sampling_params,
    #             use_tqdm=False
    #         )

    #         output = outputs[0]
    #         text = output.outputs[0].text
    #         length += len(output.outputs[0].token_ids)
    #         lengths.append(len(output.outputs[0].token_ids))
    #         with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
    #                 f.write(json.dumps({"id": id, "temperature": temperature, "output": len(output.outputs[0].token_ids), "content": text})+"\n")

    #     avg = length/len(prompts)
    #     with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
    #         f.write(json.dumps({"temperature": temperature, "avg": int(avg), "max": max(lengths)})+"\n")

    if comparison == "overthink":
        with open(f"/root/project/dos/comparison/overthink/prompt.json", "r") as f:
            prompts = eval(f.read())

        length = 0
        lengths = []
        for id, content in enumerate(prompts):
            prompt = tokenizer.apply_chat_template(
                [{"role":"user", "content":content}],
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
            length += len(output.outputs[0].token_ids)
            lengths.append(len(output.outputs[0].token_ids))
            with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
                    f.write(json.dumps({"id": id, "temperature": temperature, "output": len(output.outputs[0].token_ids), "content": text})+"\n")

        avg = length/len(prompts)
        with open(f"/root/project/dos/exp/comparison/{comparison}_{model}_results.jsonl", "a") as f:
            f.write(json.dumps({"temperature": temperature, "avg": int(avg), "max": max(lengths)})+"\n")