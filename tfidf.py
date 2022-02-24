#!/usr/bin/env python

"""
Ranking documents using TF-IDF

@author: 4oh4
16/03/2020

Ranks documents based on similarity using overlapping terms
(Does not use semantic similarity, only matching words)

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Uncomment here to download the latest nltk stop words
import nltk
from nltk.corpus import stopwords

nltk.download("punkt")
stop_words = set(stopwords.words("english"))

# Or, just use these hard-coded instead
# stop_words = {'a','about','above','after','again','against','ain','all','am','an','and','any','are','aren',"aren't",'as','at','be','because','been','before','being','below','between','both','but','by','can','couldn',"couldn't",'d','did','didn',"didn't",'do','does','doesn',"doesn't",'doing','don',"don't",'down','during','each','few','for','from','further','had','hadn',"hadn't",'has','hasn',"hasn't",'have','haven',"haven't",'having','he','her','here','hers','herself','him','himself','his','how','i','if','in','into','is','isn',"isn't",'it',"it's",'its','itself','just','ll','m','ma','me','mightn',"mightn't",'more','most','mustn',"mustn't",'my','myself','needn',"needn't",'no','nor','not','now','o','of','off','on','once','only','or','other','our','ours','ourselves','out','over','own','re','s','same','shan',"shan't",'she',"she's",'should',"should've",'shouldn',"shouldn't",'so','some','such','t','than','that',"that'll",'the','their','theirs','them','themselves','then','there','these','they','this','those','through','to','too','under','until','up','ve','very','was','wasn',"wasn't",'we','were','weren',"weren't",'what','when','where','which','while','who','whom','why','will','with','won',"won't",'wouldn',"wouldn't",'y','you',"you'd","you'll","you're","you've",'your','yours','yourself','yourselves'}


class LemmaTokenizer:
    """
    Interface to the WordNet lemmatizer from nltk
    """

    ignore_tokens = [",", ".", ";", ":", '"', "``", "''", "`"]

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [
            self.wnl.lemmatize(t)
            for t in word_tokenize(doc)
            if t not in self.ignore_tokens
        ]


tokenizer = LemmaTokenizer()

# Lemmatize the stop words
stop_words = tokenizer(" ".join(stop_words))

vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenizer)


def rank_documents(search_terms: str, documents: list):
    """
    Search for search_terms in documents, and return a list of cosine similarity scores
    """

    try:
        vectors = vectorizer.fit_transform([search_terms] + documents)

        cosine_similarities = linear_kernel(vectors[0:1], vectors).flatten()

        document_scores = [
            item.item() for item in cosine_similarities[1:]
        ]  # convert back to native Python dtypes

    except ValueError:
        print(f"Unable to rank documents for search terms: {search_terms}")

        document_scores = [0.0 for _ in range(len(documents))]

    return document_scores


def top_pairs(keyToDoc: dict, topN):
    names, documents = [list(x) for x in zip(*keyToDoc.items())]

    vectors = vectorizer.fit_transform(documents)

    global_top_scores = []
    for first_idx, entry_vec in enumerate(vectors):
        offset = first_idx + 1
        # Do not record flipped duplicates. Must include self match for some reason?
        cosine_similarities = linear_kernel(
            vectors[offset - 1 : offset], vectors[offset - 1 :]
        ).flatten()

        # convert back to native Python dtypes
        document_scores = [item.item() for item in cosine_similarities[1:]]

        # Top scores for this subset.
        sorted_top_scores = sorted(
            enumerate(document_scores), reverse=True, key=lambda x: x[1]
        )[:topN]

        matches = [
            (first_idx, other_idx + offset, score)
            for other_idx, score in sorted_top_scores
        ]
        global_top_scores = global_top_scores + matches

    # Return highest scoring matches globally.
    top_index_matches = sorted(global_top_scores, reverse=True, key=lambda x: x[2])[
        :topN
    ]

    # Return the OG keys
    return [
        (names[first_idx], names[other_idx], score)
        for first_idx, other_idx, score in top_index_matches
    ]
