import os
import click
import torch
import evaluate
import numpy as np
import copy
from dataclasses import dataclass, field
from torch.nn.utils.rnn import pad_sequence
from tqdm.auto import tqdm
from typing import Dict, Optional, Sequence
from datasets import load_dataset
from transformers import (
    Trainer,
    Seq2SeqTrainer,
    PreTrainedTokenizerBase,
    Seq2SeqTrainingArguments)
import transformers
import lm_eval

IGNORE_INDEX = -100
MMLU_DATA_BASE_DIR = "/nfs/pool002/users/mmakni/datasets/mmlu-evals"

@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    source_max_len: int
    target_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [f"{self.tokenizer.bos_token}{example['input']}" for example in instances]
        targets = [f"{example['output']}{self.tokenizer.eos_token}" for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.source_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.target_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt['input_ids'],
            tokenized_targets['input_ids']
        ):
            if not self.predict_with_generate:
                input_ids.append(torch.tensor(tokenized_source + tokenized_target))
                if not self.train_on_source:
                    labels.append(
                        torch.tensor([IGNORE_INDEX for _ in range(len(tokenized_source))] + copy.deepcopy(tokenized_target))
                    )
                else:
                    labels.append(torch.tensor(copy.deepcopy(tokenized_source + tokenized_target)))
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX) if not self.predict_with_generate else None
        data_dict = {
            'input_ids': input_ids,
            'attention_mask':input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict['labels'] = labels
        return data_dict

def _run_evaluation(
    trainer: Seq2SeqTrainer,
    tokenizer: PreTrainedTokenizerBase,
    mmlu_name: Optional[str] = None,
    mmlu_split: Optional[str] = None,
    max_mmlu_samples: Optional[int] = None,
    mmlu_source_max_len: int = 2048,
) -> Dict[str, float]:
    # Defaults
    if mmlu_name is None:
        mmlu_name = "mmlu-fs"
    if mmlu_split is None:
        mmlu_split = "test"

    if mmlu_name == "mmlu-zs":
        mmlu_dataset = load_dataset("json", data_files={
            "eval": os.path.join(MMLU_DATA_BASE_DIR, "zero_shot_mmlu_val.json"),
            "test": os.path.join(MMLU_DATA_BASE_DIR, "zero_shot_mmlu_test.json"),
        })
        mmlu_dataset = mmlu_dataset.remove_columns("subject")
    # MMLU Five-shot (Eval/Test only)
    elif mmlu_name == "mmlu" or mmlu_name == "mmlu-fs":
        mmlu_dataset = load_dataset("json", data_files={
            "eval": os.path.join(MMLU_DATA_BASE_DIR, "five_shot_mmlu_val.json"),
            "test": os.path.join(MMLU_DATA_BASE_DIR, "five_shot_mmlu_test.json"),
        })
        # mmlu_dataset = mmlu_dataset.remove_columns("subject")
    mmlu_dataset = mmlu_dataset[mmlu_split]
    if max_mmlu_samples is not None:
        mmlu_dataset = mmlu_dataset.select(range(max_mmlu_samples))
    abcd_idx = [
        tokenizer("A", add_special_tokens=False).input_ids[0],
        tokenizer("B", add_special_tokens=False).input_ids[0],
        tokenizer("C", add_special_tokens=False).input_ids[0],
        tokenizer("D", add_special_tokens=False).input_ids[0],
    ]
    accuracy = evaluate.load("accuracy")

    def on_evaluate() -> Dict[str, float]:
        data_loader = trainer.get_eval_dataloader(mmlu_dataset)
        # source_max_len = trainer.data_collator.source_max_len
        # trainer.data_collator.source_max_len = args.mmlu_source_max_len
        if trainer.data_collator.source_max_len != mmlu_source_max_len:
            raise ValueError

        trainer.model.eval()
        preds, refs = [], []
        loss_mmlu = 0
        for batch in tqdm(data_loader, total=len(data_loader)):
            (loss, logits, labels) = trainer.prediction_step(trainer.model,batch,prediction_loss_only=False,)
            # There are two tokens, the output, and eos token.
            for i, logit in enumerate(logits):
                label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                logit_abcd = logit[label_non_zero_id-1][abcd_idx]
                preds.append(torch.argmax(logit_abcd).item())
            labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:,0]
            refs += [abcd_idx.index(label) for label in labels.tolist()]
            loss_mmlu += loss.item()
        # Extract results by subject.
        results = {"mmlu_loss":loss_mmlu/len(data_loader)}
        subject = mmlu_dataset["subject"]
        subjects = {s:{"refs":[], "preds":[]} for s in set(subject)}
        for s,p,r in zip(subject, preds, refs):
            subjects[s]["preds"].append(p)
            subjects[s]["refs"].append(r)
        subject_scores = []
        for subject in subjects:
            subject_score = accuracy.compute(
                references=subjects[subject]["refs"],
                predictions=subjects[subject]["preds"]
            )["accuracy"]
            results[f"mmlu_{mmlu_split}_accuracy_{subject}"] = subject_score
            subject_scores.append(subject_score)
        results[f"mmlu_{mmlu_split}_accuracy"] = np.mean(subject_scores)
        trainer.log(results)
        # trainer.data_collator.source_max_len = source_max_len
        return results

    return on_evaluate()


def run_evaluation(trainer: Trainer) -> Dict[str, float]:
    # Some patches to make stuff run
    trainer.args.remove_unused_columns = False
    # https://github.com/facebookresearch/llama-recipes/pull/196/files
    trainer.tokenizer.pad_token = trainer.tokenizer.eos_token

    seq2seq_trainer = Seq2SeqTrainer(
        model=trainer.model,
        tokenizer=trainer.tokenizer,
        args=Seq2SeqTrainingArguments(
            **trainer.args.to_dict()
        ),
        data_collator=DataCollatorForCausalLM(
            tokenizer=trainer.tokenizer,
            source_max_len=2048,
            target_max_len=512,
            train_on_source=False,
            predict_with_generate=False,
        ),
    )

    # Run the evaluation
    return _run_evaluation(
        trainer=seq2seq_trainer,
        tokenizer=seq2seq_trainer.tokenizer)


def calculate_avg_accuracy(task_names: str, results: dict) -> float:
    n_tasks = len(task_names)
    acc_cumul = sum(
        result.get('acc_norm,none', result['acc,none']) for task, result in results.items() if 'mmlu' not in task
    )

    questions_per_mmlu_task = {
        task_name: lm_eval.tasks.get_task_dict([task_name])[task_name].dataset["test"].num_rows
        for task_name in task_names
        if 'mmlu' in task_name
    }

    if not questions_per_mmlu_task:
        return acc_cumul / n_tasks

    # Calculate average accuracy for mmlu tasks, weighted by number of questions in each task
    acc_mmlu = sum(
        result.get('acc_norm,none', result['acc,none']) * questions_per_mmlu_task[task]
        for task, result in results.items()
        if 'mmlu' in task
    )
    acc_mmlu_avg = acc_mmlu / sum(questions_per_mmlu_task.values())

    return (acc_cumul + acc_mmlu_avg) / (n_tasks - len(questions_per_mmlu_task) + 1)
