import numpy as np
from torch.nn import functional as F
import wandb
from transformers import T5ForConditionalGeneration, T5Tokenizer, AdamW
import json
import ir_datasets
import pandas as pd
import pyterrier as pt
pt.init()
from pyterrier_pisa import PisaIndex

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--negs', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()

rng = np.random.RandomState(args.seed)

wandb.init(
    project="t5-train",
    config={
      'model': 'monot5',
      'desc': 'debug',
      'negs': args.negs,
      'seed': args.seed,
    }
)

import torch
torch.manual_seed(0)

_logger = ir_datasets.log.easy()

bm25 = PisaIndex.from_dataset('msmarco_passage').bm25()

OUTPUTS = ['true', 'false']

def iter_train():
  dataset = ir_datasets.load('msmarco-passage/train')
  queries = {q.query_id: q.text for q in dataset.queries}
  docs = dataset.docs
  while True:
    for q in dataset.qrels:
      bm25_res = bm25.search(queries[q.query_id])
      bm25_res = bm25_res[bm25_res.docno != q.doc_id]
      if len(bm25_res) > 0:
        negs = bm25_res.sample(n=args.negs, replace=True, random_state=rng).docno
        if len(negs) > 0:
          yield 'Query: ' + queries[q.query_id] + ' Document: ' + docs.lookup(q.doc_id).text + ' Relevant:', OUTPUTS[0]
          for neg in negs:
            yield 'Query: ' + queries[q.query_id] + ' Document: ' + docs.lookup(neg).text + ' Relevant:', OUTPUTS[1]

train_iter = _logger.pbar(iter_train(), desc='total train samples')

model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
tokenizer = T5Tokenizer.from_pretrained("t5-base")
optimizer = AdamW(model.parameters(), lr=5e-5)

OUT_IDS = [tokenizer(t)['input_ids'][0] for t in OUTPUTS]

model.train()
for epoch in range(100):
    total_loss = 0
    correct = 0
    count = 0
    for _ in range(1024*16):
      inp, out = [], []
      for i in range(args.negs+1):
        i, o = next(train_iter)
        inp.append(i)
        out.append(o)
      inp_ids = tokenizer(inp, return_tensors='pt', padding=True).input_ids.cuda()
      out_ids = tokenizer(out, return_tensors='pt', padding=True).input_ids.cuda()
      model_out = model(input_ids=inp_ids, decoder_input_ids=torch.full_like(inp_ids[:,:1], model.config.decoder_start_token_id)).logits
      logprobs = F.log_softmax(model_out[:, :, OUT_IDS], dim=2)[:, 0, :]
      loss = -1 * (logprobs[0, 0] + logprobs[1:, 1].sum())
      loss.backward()
      optimizer.step()
      optimizer.zero_grad()
      total_loss = loss.item()
      count += 1
      c = (logprobs[0] > logprobs[1:]).sum() / args.negs
      correct += c
      total_loss += loss.item()
      count += 1
      wandb.log({'loss': loss.item(), "acc": c})
    model.save_pretrained(f'data/t5-base-{args.negs}-{args.seed}-{epoch}')
