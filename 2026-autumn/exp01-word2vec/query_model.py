"""Query a model produced by this lab; explain out-of-vocabulary words."""
import argparse
from gensim.models import KeyedVectors
from lab_common import tokenize

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--vectors", required=True, help="Path to window_2/ or window_5/vectors.kv")
p.add_argument("words", nargs="+")
a = p.parse_args()
vectors = KeyedVectors.load(a.vectors)
for query in a.words:
    tokens = tokenize(query)
    if len(tokens) != 1 or tokens[0] not in vectors:
        print(f"{query}: OOV or not one token; this Word2Vec model has no vector for it.")
        continue
    print(query, vectors.most_similar(tokens[0], topn=5))
