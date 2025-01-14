# Automated Note Linking Experiments

Using 3.7.1 global pyenv
Why do we weighted Tfidf with glove?
We use 2 versions of tfidf currently - the scikit learn version and the gensim version. Why? Not not just use the gensim tfidf?

- add tfidf scores
- label existing links and shared tags
- easier to read output
- ignore list for output
- add USE +
- do combined ranking
- figure how to handle tags
- connect to obsidian and add light randomization
- pairing ignores
- document ignores
- add keyword weighting
- clean up real documents
- pull more example graphs
- try training and learning
- add topological features
- try deep learning
- add link sequence features?
- and viewing pattern features?
- pull in online information features?

https://groups.google.com/g/gensim/c/8wCZp4Ievk4?pli=1
"You can combine similarity matrices by (weighted) averaging:

combined*similarity_matrix = SparseTermSimilarityMatrix(0.1 * first*similarity_matrix.matrix + 0.9 * second_similarity_matrix.matrix)

You could also view the similarity matrices as sparse directed graphs between words and apply e.g. power iteration to compute a denser closure, where we infer the similarities of previously unconnected words by taking e.g. the harmonic mean of the shortest path between them."

# doc-similarity (Original)

![Python application](https://github.com/4OH4/doc-similarity/workflows/Python%20application/badge.svg)

https://github.com/4OH4/doc-similarity

Find and rank relevant content in Python using NLP, TF-IDF and GloVe.

This repository includes two methods of ranking text content by similarity:

1.  Term Frequency - inverse document frequency (TF-idf)
2.  Semantic similarity, using GloVe word embeddings

Given a search query (text string) and a document corpus, these methods calculate a similarity metric for each document vs the query. Both methods exist as standalone modules, with explanation and demonstration code inside `examples.ipynb`.

There is an associated [blog post](https://towardsdatascience.com/how-to-rank-text-content-by-semantic-similarity-4d2419a84c32?source=friends_link&sk=a23730c6fad2744440eb8d4561088aa8) that explains the contents of this repository in more detail.

## Acknowledgements

The code in this repository utilises, is derived from and extends the excellent [Scikit-Learn](https://scikit-learn.org/), [Gensim](https://radimrehurek.com/gensim/) and [NLTK](https://www.nltk.org/) packages.

## Setup and requirements

Python 3 (v3.7 tested) and the following packages (all available via `pip`):

    pip install scikit-learn~=0.22
    pip install gensim~=3.8
    pip install nltk~=3.4

Or install via the `requirements.txt` file:

    pip install -r requirements.txt

## Running the example notebook

After installing the requirements (if necessary), open and run `examples.ipynb` using Jupyter Lab.

## Using the standalone TF-idf class

This module is a wrapper around the Scikit-Learn `TfidfVectorizer`, with some additional functionality from `nltk` to handle stopwords, lemmatization and cosine similarity calculation. To run:

    from tfidf import rank_documents

    document_scores = rank_documents(search_terms, documents)

## Using the standalone DocSim class

There is a self-contained class - DocSim - for running sematic similarity queries. This can be imported as a module and used without additional code:

    from docsim import DocSim

    docsim = DocSim(verbose=True)

    similarities = docsim.similarity_query(query_string, documents)

By default, a GloVe word embedding model is loaded (`glove-wiki-gigaword-50`), although a custom model can also be used.

The word embedding models can be quite large and slow to load, although subsequent operations are faster. The multi-threaded version of the class loads the model in the background, to avoid locking the main thread for a significant period of time. It is used in a similar way, although will raise an exception if the model is still loading so the status of the `model_ready` property should be checked first. The only difference is the import:

    from docsim import DocSim_threaded

## Running unit tests

To install the package requirements to run the unit tests:

    pip install -r requirements_unit_test.txt

To run all test cases, from the repository root:

    pytest

## Other

Comments and feedback welcome! Please raise an issue if you find any errors or omissions.
