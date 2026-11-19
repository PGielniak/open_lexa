"""Train a LoRA adapter for tool calling and score it on the held-out split.

Base: Qwen2-1.5B-Instruct, the model Hailo publishes a LoRA-ready HAR for
(``qwen2_1.5b_instruct.q.har``, see docs/LLM_FINETUNING.md). The adapter only
targets the MLP projections, the layout used by the known working Hailo LoRA
flow; check the DFC LoRA tutorial before widening it. Runs on a CUDA machine:

    uv run --with 'torch' --with 'transformers>=4.51,<5' --with 'peft>=0.15,<1' \\
        --with 'trl>=0.20,<1' --with 'datasets>=3,<5' --with accelerate \\
        python hailo-llm/tools/train_tool_lora.py --data ~/tool-data --out ~/qwen2-lexa-lora

    # score only (base model, or base + adapter)
    ... train_tool_lora.py --data ~/tool-data --eval-only [--adapter ~/qwen2-lexa-lora]

The score is a GPU float result. The quantized model on the Hailo-10H can be much
worse; repeat the evaluation on the device before switching the receiver.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

BASE = 'Qwen/Qwen2-1.5B-Instruct'
TARGETS = ['gate_proj', 'up_proj', 'down_proj']


def parse(text: str):
    """The model's JSON answer as (name, arguments); None when unparsable."""
    line = text.strip().splitlines()[0] if text.strip() else ''
    try:
        data = json.loads(line)
    except ValueError:
        return None
    if not isinstance(data, dict) or 'name' not in data:
        return None
    arguments = data.get('arguments') or {}
    return data['name'], {k: str(v).strip().lower() for k, v in arguments.items()}


def score(model, tokenizer, rows) -> dict:
    import torch

    counts = {'total': 0, 'exact': 0, 'tool': 0, 'invalid': 0, 'false_trigger': 0, 'negatives': 0}
    model.eval()
    for row in rows:
        prompt, expected = row['messages'][:2], parse(row['messages'][2]['content'])
        ids = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors='pt').to(model.device)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=64, do_sample=False)
        got = parse(tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True))
        counts['total'] += 1
        counts['invalid'] += got is None
        if got is not None:
            counts['tool'] += got[0] == expected[0]
            counts['exact'] += got == expected
        if expected[0] is None:
            counts['negatives'] += 1
            counts['false_trigger'] += got is not None and got[0] is not None
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='Output of export_tool_dataset.py')
    parser.add_argument('--base', default=BASE)
    parser.add_argument('--out', type=Path, help='Adapter output directory')
    parser.add_argument('--adapter', type=Path, help='Existing adapter for --eval-only')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--epochs', type=float, default=3)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--rank', type=int, default=16)
    args = parser.parse_args()
    if not args.eval_only and not args.out:
        parser.error('--out is required for training')

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16, device_map='auto')
    test_rows = [json.loads(line) for line in (args.data / 'test.jsonl').read_text(encoding='utf-8').splitlines()
                 if line.strip()]

    if args.eval_only:
        if args.adapter:
            model = PeftModel.from_pretrained(model, str(args.adapter))
        print(json.dumps(score(model, tokenizer, test_rows)))
        return

    def to_prompt_completion(row):
        return {'prompt': row['messages'][:2], 'completion': row['messages'][2:]}

    dataset = load_dataset('json', data_files={'train': str(args.data / 'train.jsonl')})['train']
    dataset = dataset.map(to_prompt_completion, remove_columns=['messages'])
    print('baseline', json.dumps(score(model, tokenizer, test_rows)))
    trainer = SFTTrainer(
        model=model, processing_class=tokenizer, train_dataset=dataset,
        peft_config=LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                               target_modules=TARGETS, task_type='CAUSAL_LM'),
        args=SFTConfig(output_dir=str(args.out), num_train_epochs=args.epochs, learning_rate=args.lr,
                       per_device_train_batch_size=8, gradient_accumulation_steps=2,
                       lr_scheduler_type='cosine', warmup_ratio=0.05, logging_steps=10,
                       save_strategy='epoch', save_total_limit=1, bf16=True,
                       completion_only_loss=True, report_to=[]))
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))
    print('fine-tuned', json.dumps(score(trainer.model, tokenizer, test_rows)))


if __name__ == '__main__':
    main()
