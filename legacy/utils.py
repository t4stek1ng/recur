from vllm import LLM, SamplingParams, RequestOutput
from vllm.transformers_utils.tokenizer import AnyTokenizer
import random

"""
outputs: list[RequestOutput]
RequestOutput:
    finished: bool
    outputs: list[CompletionOutput]
    prompt: str
    prompt_logprobs: numpy
    prompt_token_ids: list[int]
    request_id: str
CompletionOutput:
    finish_reason: str
    text: str
    token_ids: list[int]
"""

def query(input_str: str | list[str], model: LLM, tokenizer: AnyTokenizer, apply_chat_template: bool, temperature: float, n: int, max_len: int) -> dict[str, str | int | list[float]]:
    if apply_chat_template:
        if isinstance(input_str, str):
            prompt = tokenizer.apply_chat_template(
                [{'role':'user','content':input_str}],
                tokenize=False,
                add_generation_prompt=True
            )

        else: # batch
            prompt = [
                tokenizer.apply_chat_template(
                    [{'role':'user','content':content}],
                    tokenize=False,
                    add_generation_prompt=True
                ) for content in input_str
            ]
    else:
        prompt = input_str

    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_len,
        n=n,
        logprobs=1,
        prompt_logprobs=1
    )

    outputs: list[RequestOutput] = model.generate(
        prompt,
        sampling_params
    )

    if isinstance(input_str, str):
        output = outputs[0]
        return [{"id":str(id),"num_tokens":len(completion.token_ids),"text":completion.text,"prompt":prompt,"logits":completion.logprobs} for id, completion in enumerate(output.outputs)]
    else: # batch
        return [[{"id":str(id),"num_tokens":len(completion.token_ids),"text":completion.text,"prompt":prompt} for id, completion in enumerate(output.outputs)] for output in outputs]