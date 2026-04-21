import time

import torch
from transformers import TrainerCallback
from unsloth import FastLanguageModel

from src.prompting import build_prompt
from src.validate import extract_equation_from_llm_response, validate_equation


def _render_dev_prompts(dev_examples, tokenizer):
    return [
        tokenizer.apply_chat_template(
            build_prompt({"nums": list(ex["nums"]), "target": ex["target"]}),
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in dev_examples
    ]


@torch.no_grad()
def _generate_batched(model, tokenizer, prompts, max_new_tokens, gen_batch_size):
    FastLanguageModel.for_inference(model)
    model.eval()
    all_outputs = []
    for i in range(0, len(prompts), gen_batch_size):
        batch = prompts[i:i + gen_batch_size]
        enc = tokenizer(
            batch, return_tensors="pt", padding=True, truncation=True,
            max_length=256, padding_side="left",
        ).to(model.device)
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
        gen = out[:, enc["input_ids"].shape[1]:]
        all_outputs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    FastLanguageModel.for_training(model)
    model.train()
    return all_outputs


def evaluate_accuracy_on_dev(model, tokenizer, dev_examples,
                             max_new_tokens=256, gen_batch_size=16,
                             return_samples_n: int = 0):
    """
    Прогоняет модель на dev и возвращает (metrics, samples).

    metrics: dict с dev_accuracy, dev_extracted_rate, n_dev, dev_eval_time_s
    samples: первые N примеров {nums, target, response, equation} для логов
    """
    prompts = _render_dev_prompts(dev_examples, tokenizer)

    t0 = time.time()
    responses = _generate_batched(model, tokenizer, prompts, max_new_tokens, gen_batch_size)
    elapsed = time.time() - t0

    n = len(responses)
    extracted = 0
    correct = 0
    samples = []
    for i, (resp, ex) in enumerate(zip(responses, dev_examples)):
        eq = extract_equation_from_llm_response(resp)
        if eq is not None:
            extracted += 1
            vr = validate_equation(eq, ex["nums"], ex["target"])
            if vr.ok:
                correct += 1
        if i < return_samples_n:
            samples.append({
                "nums": list(ex["nums"]), "target": ex["target"],
                "response": resp[:300], "equation": eq,
            })

    metrics = {
        "dev_accuracy": correct / n if n else 0.0,
        "dev_extracted_rate": extracted / n if n else 0.0,
        "n_dev": n,
        "dev_eval_time_s": round(elapsed, 1),
    }
    return metrics, samples


class CountdownAccuracyCallback(TrainerCallback):
    """
    Триггерится по step'ам, делает accuracy eval через генерацию.

    Триггерится на on_step_end, а не на on_evaluate, чтобы не зависеть от
    Trainer.evaluate() (тот бы посчитал loss).

    Поддерживает два размера: eval_n_quick для промежуточных чекпоинтов
    (быстрый sanity), eval_n_final для финального eval.
    """

    def __init__(self, dev_examples, tokenizer,
                 eval_steps: int,
                 eval_n_quick: int | None = None,
                 max_new_tokens: int = 256,
                 gen_batch_size: int = 16,
                 verbose_n: int = 3):
        self.dev = list(dev_examples)
        self.tokenizer = tokenizer
        self.eval_steps = eval_steps
        self.quick_slice = (
            self.dev[:eval_n_quick] if eval_n_quick else self.dev
        )
        self.max_new_tokens = max_new_tokens
        self.gen_batch_size = gen_batch_size
        self.verbose_n = verbose_n

    def _log_and_print(self, metrics, samples, step):
        print(
            f"\n[step {step}] "
            f"dev_acc={metrics['dev_accuracy']:.3f}  "
            f"ext={metrics['dev_extracted_rate']:.3f}  "
            f"n={metrics['n_dev']}  "
            f"({metrics['dev_eval_time_s']}s)"
        )
        for s in samples:
            print(f"  nums={s['nums']} target={s['target']}")
            print(f"    got: {s['response'][:120]!r}")
            print(f"    eq:  {s['equation']!r}")
        try:
            import wandb
            if wandb.run is not None:
                wandb.log(metrics, step=step)
        except ImportError:
            pass

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step == 0 or state.global_step % self.eval_steps != 0:
            return control
        metrics, samples = evaluate_accuracy_on_dev(
            model, self.tokenizer, self.quick_slice,
            max_new_tokens=self.max_new_tokens,
            gen_batch_size=self.gen_batch_size,
            return_samples_n=self.verbose_n,
        )
        state.log_history.append({**metrics, "step": state.global_step})
        self._log_and_print(metrics, samples, state.global_step)
        return control