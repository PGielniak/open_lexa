"""Fine-tune a Polish Whisper checkpoint on the exported audiofolder dataset.

Runs on a CUDA machine (the cluster is ARM64 Pis without a GPU). The
dependencies are not part of the project; pull them in for this run only:

    uv run --with 'transformers>=4.44,<5' --with 'datasets[audio]>=2.20,<4' \\
        --with accelerate --with evaluate --with jiwer --with torch \\
        python whisper-api/tools/finetune_whisper.py \\
        --data ~/whisper-data --model openai/whisper-small --out ~/whisper-small-pl-lexa

With little data keep the encoder frozen (default) and the epoch count low;
watch the test WER, not the training loss.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, required=True, help='Output of export_finetune_dataset.py')
    parser.add_argument('--model', default='openai/whisper-small')
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=float, default=5)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--train-encoder', action='store_true', help='Also update the encoder weights')
    args = parser.parse_args()

    import evaluate
    import torch
    from datasets import Audio, load_dataset
    from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration,
                              WhisperProcessor)

    processor = WhisperProcessor.from_pretrained(args.model, language='polish', task='transcribe')
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.generation_config.language = 'polish'
    model.generation_config.task = 'transcribe'
    model.generation_config.forced_decoder_ids = None
    if not args.train_encoder:
        model.freeze_encoder()

    dataset = load_dataset('audiofolder', data_dir=str(args.data))
    dataset = dataset.cast_column('audio', Audio(sampling_rate=16000))

    def prepare(batch):
        audio = batch['audio']
        batch['input_features'] = processor.feature_extractor(
            audio['array'], sampling_rate=audio['sampling_rate']).input_features[0]
        batch['labels'] = processor.tokenizer(batch['transcription']).input_ids
        return batch

    dataset = dataset.map(prepare, remove_columns=dataset['train'].column_names)

    @dataclass
    class Collator:
        def __call__(self, features):
            inputs = processor.feature_extractor.pad(
                [{'input_features': f['input_features']} for f in features], return_tensors='pt')
            labels = processor.tokenizer.pad(
                [{'input_ids': f['labels']} for f in features], return_tensors='pt')
            ids = labels['input_ids'].masked_fill(labels.attention_mask.ne(1), -100)
            # The tokenizer already adds <|startoftranscript|>; the model adds it again.
            if (ids[:, 0] == model.config.decoder_start_token_id).all():
                ids = ids[:, 1:]
            inputs['labels'] = ids
            return inputs

    wer = evaluate.load('wer')

    def metrics(prediction):
        labels = prediction.label_ids
        labels[labels == -100] = processor.tokenizer.pad_token_id
        predicted = processor.batch_decode(prediction.predictions, skip_special_tokens=True)
        reference = processor.batch_decode(labels, skip_special_tokens=True)
        return {'wer': wer.compute(predictions=predicted, references=reference)}

    has_test = 'test' in dataset
    training = Seq2SeqTrainingArguments(
        output_dir=str(args.out), per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size, learning_rate=args.lr, warmup_ratio=0.1,
        num_train_epochs=args.epochs, fp16=torch.cuda.is_available(),
        eval_strategy='epoch' if has_test else 'no', save_strategy='epoch', save_total_limit=2,
        load_best_model_at_end=has_test, metric_for_best_model='wer', greater_is_better=False,
        predict_with_generate=True, generation_max_length=96, logging_steps=10, report_to=[])
    trainer = Seq2SeqTrainer(
        model=model, args=training, train_dataset=dataset['train'],
        eval_dataset=dataset['test'] if has_test else None, data_collator=Collator(),
        compute_metrics=metrics if has_test else None, processing_class=processor.feature_extractor)
    if has_test:
        print('baseline', trainer.evaluate())
    trainer.train()
    if has_test:
        print('fine-tuned', trainer.evaluate())
    trainer.save_model(str(args.out))
    processor.save_pretrained(str(args.out))


if __name__ == '__main__':
    main()
