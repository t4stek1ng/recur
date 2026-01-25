import json
import os
from vllm import LLM, SamplingParams, RequestOutput

os.environ['RANK'] = '0'
os.environ['VLLM_USE_V1'] = '0'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
temperature = 0.5
model = "QwQ-32B"

if os.path.exists(f"/root/project/dos/exp/baseline/{model}_gsm8k_results.jsonl"):
    pass
else:
    with open(f"/root/project/dos/exp/baseline/{model}_gsm8k_results.jsonl", "w") as f:
        pass

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

length = 0
lengths = []
for id, data in enumerate(gsm8k_dataset):
    content = data["question"]
    print(f"Question: {content}\n")
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
    print(f"Assistant: {text}\n")
    length += len(output.outputs[0].token_ids)
    lengths.append(len(output.outputs[0].token_ids))
    with open(f"/root/project/dos/exp/baseline/{model}_gsm8k_results.jsonl", "a") as f:
            f.write(json.dumps({"id": id, "temperature": temperature, "output": len(output.outputs[0].token_ids), "content": text})+"\n")

avg = length/len(gsm8k_dataset)
with open(f"/root/project/dos/exp/baseline/{model}_gsm8k_results.jsonl", "a") as f:
    f.write(json.dumps({"temperature": temperature, "avg": int(avg), "max": max(lengths)})+"\n")