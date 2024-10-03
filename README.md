# PyTerrier_t5

This is the [PyTerrier](https://github.com/terrier-org/pyterrier) plugin for the [Mono and Duo T5](https://arxiv.org/pdf/2101.05667.pdf) ranking approaches [[Nogueira21]](#Nogueira21).

Note that this package only supports scoring from a pretrained models (like [this one](https://huggingface.co/castorini/monot5-base-msmarco)).

## Installation

This repostory can be installed using Pip.

    pip install --upgrade git+https://github.com/terrierteam/pyterrier_t5.git


## Building T5 pipelines

You can use MonoT5 just like any other text-based re-ranker. By default, it uses a MonoT5 model previously
trained on MS MARCO passage ranking training queries.

```python
import pyterrier as pt
from pyterrier_t5 import MonoT5ReRanker, DuoT5ReRanker
monoT5 = MonoT5ReRanker() # loads castorini/monot5-base-msmarco by default
duoT5 = DuoT5ReRanker() # loads castorini/duot5-base-msmarco by default

dataset = pt.get_dataset("irds:vaswani")
bm25 = pt.BatchRetrieve(pt.get_dataset("vaswani").get_index(), wmodel="BM25")
mono_pipeline = bm25 >> pt.text.get_text(dataset, "text") >> monoT5
duo_pipeline = mono_pipeline % 5 >> duoT5 # apply a rank cutoff of 5 from monoT5 since duoT5 is too costly to run over the full result list
```

Note that both approaches require the document text to be included in the dataframe (see [pt.text.get_text](https://pyterrier.readthedocs.io/en/latest/text.html#pyterrier.text.get_text)).

MonoT5ReRanker and DuoT5ReRanker have the following options:
 - `model` (default: `'castorini/monot5-base-msmarco'` for mono, `'castorini/duot5-base-msmarco'` for duo). HGF model name. Defaults to a version trained on MS MARCO passage ranking.
 - `tok_model` (default: `'t5-base'`). HGF tokenizer name.
 - `batch_size` (default: `4`). How many documents to process at the same time.
 - `text_field` (default: `text`). The dataframe attribute in which the document text is stored.
 - `verbose` (default: `True`). Show progress bar.

## Examples

Checkout out the notebooks, even on Colab:

 - Reranking a dataframe [[Github](https://github.com/terrierteam/pyterrier_t5/blob/master/pyterrier_monoT5_direct_rerank.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_t5/blob/master/pyterrier_monoT5_direct_rerank.ipynb)]
 - Vaswani [[Github](https://github.com/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_vaswani.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_vaswani.ipynb)]
 - TREC-COVID [[Github](https://github.com/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_trec-covid.ipynb)] [[Colab](https://colab.research.google.com/github/terrierteam/pyterrier_t5/blob/master/pyterrier_t5_trec-covid.ipynb)]

## Implementation Details

We use a PyTerrier transformer to score documents using a T5 model.

Sequences longer than the model's maximum of 512 tokens are silently truncated. Consider splitting long texts
into passages and aggregating the results ([examples](https://pyterrier.readthedocs.io/en/latest/text.html#working-with-passages-rather-than-documents)).

## References

  - <a id="Nogueira21"/>Ronak Pradeep, Rodrigo Nogueira, and Jimmy Lin. The Expando-Mono-Duo Design Pattern for Text Ranking withPretrained Sequence-to-Sequence Models. https://arxiv.org/pdf/2101.05667.pdf
  - <a id="Macdonald20"/>Craig Macdonald, Nicola Tonellotto. Declarative Experimentation inInformation Retrieval using PyTerrier. Craig Macdonald and Nicola Tonellotto. In Proceedings of ICTIR 2020. https://arxiv.org/abs/2007.14271

## Credits

- Sean MacAvaney, University of Glasgow
