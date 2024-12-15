# build_vocab.py

import pickle
from collections import Counter

with open("processed_data/train.pkl", "rb") as f:
    train_df = pickle.load(f)

# Build vocabulary from train tokens
all_tokens = [t for row in train_df['tokens'] for t in row]
counter = Counter(all_tokens)
# Keep a subset of vocab
vocab_size = 5000
most_common = counter.most_common(vocab_size-2) # reserve spots for <unk>, <pad>
vocab = {"<pad>":0, "<unk>":1}
for i, (w, _) in enumerate(most_common, start=2):
    vocab[w] = i

with open("processed_data/vocab.pkl", "wb") as f:
    pickle.dump(vocab, f)

print("Vocab built with size:", len(vocab))
