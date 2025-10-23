"""
Calibration and test data loaders for the 3BASiL framework.

This module provides data loaders compatible with SparseGPT, ALPS, HASSLE-free, and other
compression algorithms. The calibration and test loaders follow the same format used in
these baseline methods for fair comparison.
"""

import random

from datasets import load_dataset


def get_c4_data(split='train'):
    return load_dataset(
        "allenai/c4",
        data_files={
            "train": "en/c4-train.00000-of-01024.json.gz",
            "test": "en/c4-validation.00000-of-00008.json.gz"
        },
        split=split,
        trust_remote_code=True
    )

def get_wikitext2_data(split='train'):
    return load_dataset('wikitext', 'wikitext-2-raw-v1', split=split, trust_remote_code=True)

def get_ptb_data(split='train'):
    return load_dataset('ptb_text_only', 'penn_treebank', split=split, trust_remote_code=True)

def get_c4_calibrationloader(nsamples=128, tokenizer=None, seqlen=2048, seed=0):
    traindata = get_c4_data(split='train')
    random.seed(seed)
    calibration_loader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['text'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        calibration_loader.append((inp, tar))
    return calibration_loader

def get_c4_testloader(tokenizer, seqlen=2048):
    testdata = get_c4_data(split='test')
    testenc = tokenizer(' '.join(testdata[:1100]['text']), return_tensors='pt')
    testenc = testenc.input_ids[:, :(256 * seqlen)]
    class TokenizerWrapper:
        def __init__(self, input_ids):
            self.input_ids = input_ids
    testenc = TokenizerWrapper(testenc)
    return testenc

def get_wikitext2_calibrationloader(nsamples=128, tokenizer=None, seqlen=2048, seed=0):
    traindata = get_wikitext2_data(split='train')
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    random.seed(seed)
    calibration_loader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        calibration_loader.append((inp, tar))
    return calibration_loader


def get_wikitext2_testloader(tokenizer):
    testdata = get_wikitext2_data(split='test')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    return testenc

def get_ptb_calibrationloader(nsamples=128, tokenizer=None, seqlen=2048, seed=0):
    traindata = get_ptb_data(split='train')
    random.seed(seed)
    calibration_loader = []
    for _ in range(nsamples):
        while True:
            i = random.randint(0, len(traindata) - 1)
            trainenc = tokenizer(traindata[i]['sentence'], return_tensors='pt')
            if trainenc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        calibration_loader.append((inp, tar))
    return calibration_loader

def get_ptb_testloader(tokenizer):
    testdata = get_ptb_data(split='test')
    testenc = tokenizer(" ".join(testdata['sentence']), return_tensors='pt')
    return testenc

def get_ppl_testloader(dataset_name, tokenizer, seqlen=2048):
    """
    Get test loader for perplexity evaluation based on dataset name.
    
    Args:
        dataset_name: Name of the dataset ('wikitext2', 'ptb', or 'c4')
        tokenizer: Tokenizer to use for encoding
        seqlen: Sequence length (used for C4 dataset)
    
    Returns:
        Test data loader for the specified dataset
    """
    if dataset_name == "wikitext2":
        return get_wikitext2_testloader(tokenizer)
    elif dataset_name == "ptb":
        return get_ptb_testloader(tokenizer)
    elif dataset_name == "c4":
        return get_c4_testloader(tokenizer, seqlen)
    else:
        raise ValueError(f"Not supported dataset name: {dataset_name}")
