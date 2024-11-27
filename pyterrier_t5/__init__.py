__version__ = '0.1.0'

import math
import warnings
import itertools
import pyterrier as pt
from collections import defaultdict
from pyterrier.model import add_ranks
import torch
from torch.nn import functional as F
from transformers import T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration


class MonoT5ReRanker(pt.Transformer):
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
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]

    def __str__(self):
        return f"MonoT5({self.model_name})"

    def transform(self, run):
        scores = []
        queries, texts = run['query'], run[self.text_field]
        it = range(0, len(queries), self.batch_size)
        prompts = self.tokenizer.batch_encode_plus(['Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')
        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]
        if self.verbose:
            it = pt.tqdm(it, desc='monoT5', unit='batches')
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


class DuoT5ReRanker(pt.Transformer):
    def __init__(self, 
                 tok_model='t5-base',
                 model='castorini/duot5-base-msmarco',
                 batch_size=4,
                 text_field='text',
                 verbose=True,
                 agg='sum'):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = T5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('true')[0]
        self.NREL = self.tokenizer.encode('false')[0]
        assert agg == 'sum', 'DuoT5ReRanker only supports sum aggregation mode at this time'
        self.agg = agg

    def __str__(self):
        return f"DuoT5({self.model_name})"

    def transform(self, run):
        scores = defaultdict(lambda: 0.)
        prompts = self.tokenizer.batch_encode_plus(['Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')
        max_vlen = self.model.config.n_positions - prompts['input_ids'].shape[1]
        for batch in self._iter_duo_batches(run):
            enc_query = self.tokenizer.batch_encode_plus([f'Query: {q}' for q in batch['query']], return_tensors='pt', padding='longest')
            enc_text0 = self.tokenizer.batch_encode_plus([f'Document0: {q}' for q in batch['text0']], return_tensors='pt', padding='longest')
            enc_text1 = self.tokenizer.batch_encode_plus([f'Document1: {q}' for q in batch['text1']], return_tensors='pt', padding='longest')
            enc = {}
            for key in enc_query:
                query = enc_query[key][:, :-1] # chop off end of sequence token-- this will be added with the prompt
                text0 = enc_text0[key][:, :-1] # chop off end of sequence token-- this will be added with the prompt
                text1 = enc_text1[key][:, :-1] # chop off end of sequence token-- this will be added with the prompt
                # Do we need to truncate? If so, how many tokens per document?
                if query.shape[1] + text0.shape[1] + text1.shape[1] > max_vlen:
                    tokens_to_truncate = query.shape[1] + text0.shape[1] + text1.shape[1] - max_vlen
                    tokens_to_truncate_per_doc = math.ceil(tokens_to_truncate / 2)
                    text0 = text0[:, :-tokens_to_truncate_per_doc]
                    text1 = text1[:, :-tokens_to_truncate_per_doc]
                # Combine the components:
                enc[key] = torch.cat([
                    query,
                    text0,
                    text1,
                    prompts[key][:query.shape[0]]], dim=1)
            enc['decoder_input_ids'] = torch.full(
                (len(batch['ids']), 1),
                self.model.config.decoder_start_token_id,
                dtype=torch.long
            )
            enc = {k: v.to(self.device) for k, v in enc.items()}
            with torch.no_grad():
                result = self.model(**enc).logits
            result = result[:, 0, (self.REL, self.NREL)]
            result = F.log_softmax(result, dim=1)[:, 0].cpu().detach().tolist()
            for (qid, did1, did2), score in zip(batch['ids'], result):
                scores[qid, did1] += score
                scores[qid, did2] += (1 - score)

        score_list = []
        for record in run.itertuples(index=False):
            score_list.append(scores[record.qid, record.docno])

        run = run.drop(columns=['score', 'rank'], errors='ignore').assign(score=score_list)
        run = add_ranks(run)
        return run

    def _iter_duo_pairs(self, run):
        warned = False
        groups = run.groupby('qid')
        if self.verbose:
            groups = pt.tqdm(groups, desc='duoT5', unit='queries')
        for qid, group in groups:
            if not warned and len(group) > 50:
                warnings.warn(f'A large number of results per query was detected ({len(group)}). Since DuoT5 '
                               'is an O(n^2) operation, this will take a considerable amount of time to process. '
                               'Consider first reducing the size of the results using the % operator.')
                warned = True
            for row1, row2 in itertools.permutations(group.itertuples(index=False), 2):
                yield row1.qid, row1.query, getattr(row1, self.text_field), getattr(row2, self.text_field), row1.docno, row2.docno

    def _iter_duo_batches(self, run):
        batch = {'ids': [], 'query': [], 'text0': [], 'text1': []}
        for qid, query, text0, text1, did0, did1 in self._iter_duo_pairs(run):
            batch['ids'].append((qid, did0, did1))
            batch['query'].append(query)
            batch['text0'].append(text0)
            batch['text1'].append(text1)
            if len(batch['ids']) == self.batch_size:
                yield batch
                for v in batch.values():
                    v.clear()
        if len(batch['ids']) > 0:
            yield batch

class mT5ReRanker(pt.Transformer):
    def __init__(self, 
                 tok_model='unicamp-dl/mt5-base-mmarco-v2',
                 model='unicamp-dl/mt5-base-mmarco-v2',
                 batch_size=4,
                 text_field='text',
                 verbose=True):
        self.verbose = verbose
        self.batch_size = batch_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = T5Tokenizer.from_pretrained(tok_model)
        self.model_name = model
        self.model = MT5ForConditionalGeneration.from_pretrained(model)
        self.model.to(self.device)
        self.model.eval()
        self.text_field = text_field
        self.REL = self.tokenizer.encode('yes')[0]
        self.NREL = self.tokenizer.encode('no')[0]

    def __str__(self):
        return f"mT5({self.model_name})"

    def transform(self, run):
        scores = []
        queries, texts = run['query'], run[self.text_field]
        it = range(0, len(queries), self.batch_size)
        prompts = self.tokenizer.batch_encode_plus(['Relevant:' for _ in range(self.batch_size)], return_tensors='pt', padding='longest')
        max_vlen = 512 - prompts['input_ids'].shape[1] #mT5Config doesn't have n_positions so we fallback to 512
        if self.verbose:
            it = pt.tqdm(it, desc='monoT5', unit='batches')
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
