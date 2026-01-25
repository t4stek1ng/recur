import json
config = json.load(open('./config.json', 'r'))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = config['latent']['gpu']
rate = config['latent']['rate']
n = config['latent']['n']
batch_size = config['latent']['batch_size']

import glob

latent = sorted(glob.glob(f"/root/project/dos/exp/8k_{config['models'][config['latent']['model_id']].split('/')[-1]}" + "/*_latent.txt"))

# load the dict
with open("/root/project/dos/words_new.json", "r") as f:
    lib = json.loads(f.read())

head = lib['head']
error = lib['error']
solution = lib['solution']
twist = lib['twist']

def query(mode: str, input: str | list[str], temperature: float, n: int, max_len: int) -> list[str] | str:
    if mode == 'tokenize':
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
    elif mode == 'direct':
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

    if n == 1:
        return outputs[0].outputs[0].text
    else:
        return [[response.text for response in output.outputs] for output in outputs]

if __name__ == "__main__":
    from vllm import LLM, SamplingParams, RequestOutput

    target_model = LLM(
        model=config['models'][config['latent']['model_id']],
        task="generate",
        enable_chunked_prefill=False,
        max_model_len=config['max_model_len'] * 2 ** rate,
        gpu_memory_utilization=config['gpu_memory_utilization']
    )

    target_tokenizer = target_model.get_tokenizer()

    for i in range(0, len(latent), batch_size):
        # build batch
        print("\n----- latent input try -----\n")
        batch = []
        for j in range(batch_size):
            try:
                path = latent[i+j]
            except:
                continue
            rank = path.split("/")[-1].split("_")[0]
            if os.path.exists(f"/root/project/dos/exp/16k_{config['models'][config['latent']['model_id']].split('/')[-1]}/{rank}_fail.txt"):
                continue
            with open(path, 'r') as f:
                input_str = f.read()
            print(f"\n{rank}: {input_str}\n")
            batch.append((rank, input_str))

        # generate
        inputs = [item for _, item in batch]
        ranks = [item for item, _ in batch]
        outputs = query('direct', inputs, config["temperature"], n, config["max_model_len"] * 2 ** rate) # outputs: list[list[str]]
        batch_overthinkings = [[thinking for thinking in thinkings if "</think>" not in thinking] for thinkings in outputs] # 收集那些达到模型长度上限的输出

        # 对 batch 中每一组输出
        for rank, tks in zip(ranks, batch_overthinkings):
            repeated = []
            # 判断是否有循环思考的输出
            for text in tks:
                print(f"\n[output {rank}] {text[:50]} ... {text[-50:]}\n")
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
                with open(f"/root/project/dos/exp/16k_{config['models'][config['latent']['model_id']].split('/')[-1]}/{rank}_recurrent.txt", "w") as f:
                    f.write(f"[recurrent thinking]\n{'\n[recurrent thinking]\n'.join(repeated)}")
                with open("/root/project/dos/log", "a") as f:
                    f.write(f"[INFO] [context length: {str(config["max_model_len"] * 2 ** rate)}] question {rank} generated recurrent output\n")

            # 否则记录latent
            else:
                if tks != []:
                    with open(f"/root/project/dos/exp/16k_{config['models'][config['latent']['model_id']].split('/')[-1]}/{rank}_latent.txt", "w") as f:
                        f.write([input_str for index, input_str in batch if index == rank][0])
                    with open(f"/root/project/dos/exp/16k_{config['models'][config['latent']['model_id']].split('/')[-1]}/{rank}_overthink.txt", "w") as f:
                        f.write(f"[overthinking]\n{'\n[overthinking]\n'.join(tks)}")
                    with open("/root/project/dos/log", "a") as f:
                        f.write(f"[INFO] [context length: {str(config["max_model_len"] * 2 ** rate)}] question {rank} has latent input\n")
                else:
                    with open(f"/root/project/dos/exp/16k_{config['models'][config['latent']['model_id']].split('/')[-1]}/{rank}_fail.txt", "w") as f:
                        pass
                    with open("/root/project/dos/log", "a") as f:
                        f.write(f"[INFO] question {rank} failed\n")