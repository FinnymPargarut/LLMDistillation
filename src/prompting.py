def build_prompt(example: dict) -> list[dict[str, str]]:
    """
    Два случая: COUNTDOWN датасет и тест соревнования.
    В первом есть датасетный промпт, во втором нужно собрать из nums и target.
    """
    if "prompt" in example:
        return example["prompt"]

    prompt = [
        {
            "content": "You are a helpful assistant. You first think about the reasoning process in the mind and then provide the user with the answer.",
            "role": "system"
        },
        {
            "content": f"Using the numbers {example['nums']}, create an equation that equals {example['target']}. You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. Show your work in <think> </think> tags. And return the final equation and answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 = 1 </answer>.",
            "role": "user"
        }
    ]
    return prompt
