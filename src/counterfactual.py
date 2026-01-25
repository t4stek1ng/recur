import os
import json

# counterfactual question

import random
OPTIONS = {"A", "B", "C", "D"}
MODE = {"directed", "reverse", "undirected"}

def counterfactual_prompting(data: list[dict[str, list[str]] | str | int], mode: str) -> str:
    """
    将数据集中的问题转化为反事实问题
    """
    question = data["question"]
    option = data["option"]
    answer = data["answer"]
    incorrect_answers = data["incorrect"]

    if option == None:
        if incorrect_answers == []: # gsm8k
            incorrect_answer = None
            if mode == "directed":
                return f"{question} Why the answer is {str(int(int(answer) * (0.5 + random.random())))}?"
            elif mode == "reverse":
                return f"{question} Why {answer} is incorrect?"
            elif mode == "undirected":
                return f"{question} There is a trap in the question, you won't figure out the correct answer at once."
        else: # gpqa
            incorrect_answer = random.sample(incorrect_answers, 1)[0]
            if mode == "directed":
                return f"{question} Why the answer is {incorrect_answer}?"
            elif mode == "reverse":
                return f"{question} Why {answer} is incorrect?"
            elif mode == "undirected":
                return f"{question} There is a trap in the question, you won't figure out the correct answer at once."
    else: # mmlu
        incorrect_option = random.sample([opt for opt in OPTIONS if opt != option])
        incorrect_answer = data["options"][incorrect_option]
        if mode == "directed":
            return f"{question} A. {data["options"]["A"]}, B. {data["options"]["B"]}, C. {data["options"]["C"]}, D. {data["options"]["D"]}. Why the answer is {incorrect_option}. {incorrect_answer}?"
        elif mode == "reverse":
            return f"{question} A. {data["options"]["A"]}, B. {data["options"]["B"]}, C. {data["options"]["C"]}, D. {data["options"]["D"]}. Why {option}. {answer} is incorrect?"
        elif mode == "undirected":
            return f"{question} A. {data["options"]["A"]}, B. {data["options"]["B"]}, C. {data["options"]["C"]}, D. {data["options"]["D"]}. There is a trap in the question, you won't figure out the correct answer at once."

if __name__ == "__main__":
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