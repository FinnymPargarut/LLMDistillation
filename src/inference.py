import json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from src.validate import extract_equation_from_llm_response, validate_equation


def load_model_and_tokenizer(model_name, load_in_4bit=False):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    if load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
        )
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=dtype,
            device_map="auto",
        )
    return model, tokenizer


def run_inference_on_dataset(model, tokenizer, dataset, prompt_builder,
                             max_new_tokens=1024, batch_size=8,
                             generation_kwargs=None,
                             chat_template_kwargs=None):
    model.eval()
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    gen_kwargs = {"do_sample": False}
    if generation_kwargs:
        gen_kwargs.update(generation_kwargs)
    ct_kwargs = chat_template_kwargs or {}

    for batch_start in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset.select(range(batch_start, min(batch_start + batch_size, len(dataset))))

        prompts = [
            tokenizer.apply_chat_template(
                prompt_builder(ex), tokenize=False, add_generation_prompt=True,
                **ct_kwargs,
            )
            for ex in batch
        ]

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
                **gen_kwargs,
            )

        new_tokens = outputs[:, inputs.input_ids.shape[1]:]
        responses = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        yield batch, responses


def evaluate_model(inference_generator, dataset_len, results_filename, print_first_n=5):
    results = []
    extracted, correct = 0, 0
    total_response_chars = 0
    global_idx = 0

    for batch, responses in inference_generator:
        for example, response in zip(batch, responses):
            equation = extract_equation_from_llm_response(response)
            if equation is not None:
                extracted += 1
                val_res = validate_equation(equation, example["nums"], example["target"])
                valid = val_res.ok
                reason = val_res.reason
            else:
                valid = False
                reason = "Failed to extract equation from response"

            if valid:
                correct += 1
            total_response_chars += len(response)

            if global_idx < print_first_n:
                print(f"\n--- {global_idx} ---")
                print(f"nums={example['nums']}, target={example['target']}")
                print(f"response: {response!r}")
                print(f"extracted={equation}, correct={valid} ({reason})")

            results.append({
                "iteration": global_idx,
                "nums": example["nums"],
                "target": example["target"],
                "model_output": response,
                "equation": equation,
                "correct": valid,
                "reason": reason,
            })
            global_idx += 1

    summary = {
        "dataset_len": dataset_len,
        "equation_extracted": extracted,
        "equation_correct": correct,
        "extracted_rate": extracted / dataset_len,
        "accuracy": correct / dataset_len,
        "avg_response_chars": total_response_chars / dataset_len,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    out_dir = Path(__file__).resolve().parent.parent / "outputs"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"{results_filename}.json", "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_example": results}, f, indent=2, ensure_ascii=False)
    return summary, results