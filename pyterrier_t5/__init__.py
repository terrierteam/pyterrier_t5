import pyterrier as pt
import pandas as pd
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from transformers import T5Config, T5Tokenizer, T5ForConditionalGeneration
from pyterrier.transformer import TransformerBase
from more_itertools import chunked
from typing import List
import re


class MonoT5ReRanker(TransformerBase):    
    def __init__(self, 
                 tok_model='t5-base',
                 model='castorini/monot5-base-msmarco',
                 batch_size=4,
                 text_field='text',
                 verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def __str__(self):
        return "MonoT5"

    def transform(self, run):
        scores = []
        queries, texts = run['query'], run[self.text_field]
        it = range(0, len(queries), self.batch_size)
        prompts = self.tokenizer.batch_encode_plus([f'Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')
        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]
        if self.verbose:
            it = pt.tqdm(it, desc='monoT5 batches')
        for start_idx in it:
            rng = slice(start_idx, start_idx+self.batch_size) # same as start_idx:start_idx+self.batch_size
            enc = self.tokenizer.batch_encode_plus([f'Query: {q} Document: {d}' for q, d in zip(queries[rng], texts[rng])], return_tensors='pt', padding='longest')
            for key, enc_value in list(enc.items()):
                enc_value = enc_value[:, :-1] # chop off end of sequence token-- this will be added with the prompt
                enc_value = enc_value[:, :max_vlen] # truncate any tokens that will not fit once the prompt is added
                enc[key] = torch.cat([enc_value, prompts[key][:enc_value.shape[0]]], dim=1) # add in the prompt to the end
            enc['decoder_input_ids'] = torch.full(
                (len(queries[rng]), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            scores += F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=scores)
        run = add_ranks(run)
        return run
