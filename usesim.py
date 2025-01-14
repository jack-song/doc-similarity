#!/usr/bin/env python

from absl import logging
import tensorflow_hub as hub
import tensorflow as tf
from sklearn.metrics.pairwise import linear_kernel
import numpy as np

logging.set_verbosity(logging.ERROR)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# messages = [
#     # Smartphones
#     "I like my phone",
#     "My phone is not good.",
#     "Your cellphone looks great.",
#     # Weather
#     "Will it snow tomorrow?",
#     "Recently a lot of hurricanes have hit the US",
#     "Global warming is real",
#     # Food and health
#     "An apple a day, keeps the doctors away",
#     "Eating strawberries is healthy",
#     "Is paleo better than keto?",
#     # Asking about age
#     "How old are you?",
#     "what is your age?",
# ]

# embeddings = embed(messages)

# for i, message_embedding in enumerate(np.array(embeddings).tolist()):
#     print("Message: {}".format(messages[i]))
#     print("Embedding size: {}".format(len(message_embedding)))
#     message_embedding_snippet = ", ".join((str(x) for x in message_embedding[:3]))
#     print("Embedding: [{}, ...]\n".format(message_embedding_snippet))

# sts_encode1 = tf.nn.l2_normalize(embed(tf.constant(batch["sent_1"].tolist())), axis=1)
# sts_encode2 = tf.nn.l2_normalize(embed(tf.constant(batch["sent_2"].tolist())), axis=1)
# cosine_similarities = tf.reduce_sum(tf.multiply(sts_encode1, sts_encode2), axis=1)
# clip_cosine_similarities = tf.clip_by_value(cosine_similarities, -1.0, 1.0)
# scores = 1.0 - tf.acos(clip_cosine_similarities) / math.pi


def top_pairs(keyToDoc: dict, topN):
    names, documents = [list(x) for x in zip(*keyToDoc.items())]

    embeddings = np.array(embed(documents)).tolist()

    global_top_scores = []
    for first_idx, entry_embedding in enumerate(embeddings):
        offset = first_idx + 1
        # Do not record flipped duplicates. Must include self match for some reason?
        cosine_similarities = linear_kernel(
            embeddings[offset - 1 : offset], embeddings[offset - 1 :]
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
