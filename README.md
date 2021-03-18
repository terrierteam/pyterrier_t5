# PyTerrier_t5

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [Mono T5](https://github.com/castorini/docTTTTTquery) ranking approach [Nogueira21].

## Installation

This repostory can be installed using Pip.

    pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git


## Building a MonoT5 pipeline

You can use MonoT5 just like any other text-based re-ranker. By default, uses a MonoT5 model
trained on MS MARCO passage ranking.

```python
from pyterrier_t5 import MonoT5ReRanker
monoT5 = MonoT5ReRanker()

dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
pipeline = bm25 >> pt.text.get_text(dataset, "text") >> monoT5
```

Note that monoT5 requires the documnt text to be included in the dataframe (see `pt.text.get_text`).

MonoT5ReRanker has the following options:
 - `model` (default: `'castorini/monot5-base-msmarco'`). HGF model name. Defaults to a version trained on MS MARCO passage ranking.
 - `tok_model` (default: `'t5-base'`). HGF tokenizer name.
 - `batch_size` (default: `4`). How many documents to process at the same time.
 - `verbose` (default: `True`). Show progress bar.

## Examples

Checkout out the notebooks, even on Colab:

 - Vaswani [[Github](https://github.com/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_vaswani.ipynb)]
 - TREC-COVID [[Github](https://github.com/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_trec-covid.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_trec-covid.ipynb)]

## Implementation Details

We use a PyTerrier transformer to score documents using a T5 model.

Sequences longer than the model's maximum of 512 tokens are silently truncated.

## References

  - [Nogueira21]: Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. The Expando-Mono-Duo Design Pattern for Text Ranking withPretrained Sequence-to-Sequence Models. https://arxiv.org/pdf/2101.05667.pdf
  - [Macdonald20]: Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Sean MacAvaney, University of Glasgow
