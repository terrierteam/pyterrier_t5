## Training monoT5

Here we provide code we have used for training monoT5 models.

 - t5train.py - this uses the MSMARCO training triples for training monoT5. It also conducts validation.

 - t5-train-bm25negs.py - this uses the MSMARCO training queries for training monoT5. It adds negative samples obtained using BM25 from a Pisa index.

# Credits

Sean MacAvaney, University of Glasgow