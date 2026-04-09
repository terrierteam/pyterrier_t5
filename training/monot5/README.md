## Training monoT5

Here we provide code we have used for training monoT5 models.

 - t5train.py - this uses the MSMARCO training triples for training monoT5. It also conducts validation.

 - t5-train-bm25negs.py - this uses the MSMARCO training queries for training monoT5. It adds negative samples obtained using BM25 from a Pisa index.

## Reproducibility of monoT5

To replicate the performance of monoT5 base on MSMARCO from the original monoT5 paper [1], you should use t5train.py with the following command:

````
python t5train.py --train_type 'full' --steps 1
````

In this way, you can reproduce [monoT5-base-10k](https://huggingface.co/castorini/monot5-base-msmarco-10k) with the following results:

|  **Model** | **AP** | **RR** | **nDCG@10** |
|:----------:|:------:|:------:|:-----------:|
| MonoT5 [1] | 0.368  |  0.947 |    0.699    |
| Replicated | 0.367  |  0.944 |    0.700    |


## How to reproduce Light-MonoT5

To replicate the performance of Light-MonoT5 base [2] on MSMARCO, you should use t5train.py with the following configuration: 

````
python t5train.py --train_type 'light' --steps 10
````

# References

[1] [NOGUEIRA, Rodrigo, et al. Document Ranking with a Pretrained Sequence-to-Sequence Model. In: EMNLP 2020](https://aclanthology.org/2020.findings-emnlp.63/)

[2] Braga et al., 'Revealing MonoT5’s Learning Mechanisms via Prompt-Token Adaptation' in ECIR 2026

# Credits

Sean MacAvaney, University of Glasgow

Marco Braga, University of Milano-Bicocca
