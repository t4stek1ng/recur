import json
import glob
import os

# configurations
config = json.load(open('./config.json', 'r'))
model_name = config['model']
model_path = f"./models/{model_name}"

def over_reflect_trim(text: str, option: str | None, answer: str) -> bool | tuple[str, str]:
    text = text.split("<think>\n")[1]
    steps = text.split("\n\n")
    think = False
    reflect = False

    # think identify
    if option == None:
        for i, step in enumerate(steps):
            if "</think>" in step:
                return False
            elif reflect and (f" {answer} " in step or f" {answer}." in step or f" {answer}," in step or f" {answer}\n" in step) and i > reflect_done_idx + 1:
                return "\n\n".join(steps[:i+1]), steps[i+1]
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
                return "\n\n".join(steps[:i+1]), steps[i+1]

    return False

def thinking_loop_trim(text: str, token: str) -> str:
    steps = text.split("\n\n")
    all_steps = steps[:-1]
    test_step = all_steps[-1]
    rev = all_steps[::-1]
    for idx, step in enumerate(rev):
        if step != test_step and step.startswith(token):
            idx_1 = len(rev) - 1 - idx
            break
    if steps[-1] == "":
        for idx, step in enumerate(all_steps):
            if step == test_step:
                idx_2 = idx
                break

        return "\n\n".join(all_steps[idx_1:idx_2+1])
    else:
        return "\n\n".join(steps[idx_1:])

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        use_fast=True,
        trust_remote_code=True
    )

    if os.path.exists(f"./exp/{model_name}/promtps.jsonl"):
        pass
    else:
        with open(f"./exp/{model_name}/promtps.jsonl", "w") as f:
            pass

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

    paths = glob.glob(f"./exp/{model_name}/*entropy*.json")

    for path in paths:
        with open(path,"r") as f:
            generation = json.loads(f.read())

        if not generation["loop"]:
            continue

        id = int(path.split("/")[-1].split("_")[1])
        thinking = generation["prompt"]
        answer = gsm8k_dataset[id]["answer"]

        over_reflect, step = over_reflect_trim(thinking, None, answer)
        tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(step))#[1:]
        token = tokenizer.convert_tokens_to_string(tokens[:3])
        thinking_loop = thinking_loop_trim(thinking, token)
        prompt = over_reflect + "\n\n" + thinking_loop + "\n\n"

        with open(f"./exp/{model_name}/promtps.jsonl", "a") as f:
            f.write(json.dumps({"id":id, "generation": False, "tokens": 3, "token": token, "prompt": prompt})+"\n")