# process dataset
# sequence: 
# 0-19: math(gsm8k)
# 20-39: physics(mmlu)
# 40:59: computer science(mmlu)
# 60-79: econometrics(mmlu)
# 80-99: logical(mmlu)
# 100-119: hard sicience problem(gpqa)

dataset = {
    "gsm8k": "/root/project/dos/dataset/ours/gsm8k_sample.jsonl",
    "mmlu": [
        "/root/project/dos/dataset/ours/college_physics_sample.jsonl",
        "/root/project/dos/dataset/ours/college_computer_science_sample.jsonl",
        "/root/project/dos/dataset/ours/econometrics_sample.jsonl",
        "/root/project/dos/dataset/ours/logical_fallacies_sample.jsonl"
    ],
    "GPQA": "/root/project/dos/dataset/ours/gpqa_sample.jsonl"
}

