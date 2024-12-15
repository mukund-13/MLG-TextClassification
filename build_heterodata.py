import os
import pickle
import pandas as pd
import numpy as np
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from torch_geometric.data import HeteroData

# Load your data
with open("processed_data/train.pkl", "rb") as f:
    train_df = pickle.load(f)
with open("processed_data/val.pkl", "rb") as f:
    val_df = pickle.load(f)
with open("processed_data/test.pkl", "rb") as f:
    test_df = pickle.load(f)

# Concatenate all
all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)

# Check columns
# Should have: 'text', 'label', 'author_id', 'tags'
# We need to tokenize 'text' now.
nlp = spacy.load("en_core_web_sm")

def tokenize_text(text):
    doc = nlp(text)
    tokens = [t.lemma_.lower() for t in doc if not t.is_stop and t.is_alpha]
    return tokens

# Tokenize all documents
all_df['tokens'] = all_df['text'].apply(tokenize_text)

# Now we have a 'tokens' column
docs_text = [" ".join(tokens) for tokens in all_df['tokens']]

# Build TF-IDF features
vectorizer = TfidfVectorizer(min_df=5)
X_tfidf = vectorizer.fit_transform(docs_text)
doc_features = torch.tensor(X_tfidf.toarray(), dtype=torch.float)

# Create author and tag mappings
authors = all_df['author_id'].unique()
author2nid = {a: i for i,a in enumerate(authors)}
num_authors = len(authors)

# Extract all tags
all_tags_list = [t for taglist in all_df['tags'] for t in taglist]
unique_tags = list(set(all_tags_list))
tag2nid = {t: i for i,t in enumerate(unique_tags)}
num_tags = len(unique_tags)

# Labels and masks
labels = torch.tensor(all_df['label'].values, dtype=torch.long)

train_size = len(train_df)
val_size = len(val_df)
test_size = len(test_df)
doc_count = len(all_df)

train_mask = torch.zeros(doc_count, dtype=torch.bool)
val_mask = torch.zeros(doc_count, dtype=torch.bool)
test_mask = torch.zeros(doc_count, dtype=torch.bool)

train_mask[:train_size] = True
val_mask[train_size:train_size+val_size] = True
test_mask[train_size+val_size:] = True

# Build edges
doc_author_pairs = []
for i, row in all_df.iterrows():
    d_i = i
    a_id = row['author_id']
    doc_author_pairs.append((d_i, author2nid[a_id]))
doc_author_pairs = np.array(doc_author_pairs).T

doc_tag_pairs = []
for i, row in all_df.iterrows():
    d_i = i
    for t in row['tags']:
        doc_tag_pairs.append((d_i, tag2nid[t]))
doc_tag_pairs = np.array(doc_tag_pairs).T if len(doc_tag_pairs) > 0 else np.empty((2,0), dtype=int)

# Random features for authors and tags
author_features = torch.randn(num_authors, 64)
tag_features = torch.randn(num_tags, 64)

# Construct HeteroData
data = HeteroData()

data['document'].x = doc_features
data['document'].y = labels
data['document'].train_mask = train_mask
data['document'].val_mask = val_mask
data['document'].test_mask = test_mask

data['author'].x = author_features
data['tag'].x = tag_features

data['document', 'to', 'author'].edge_index = torch.tensor(doc_author_pairs, dtype=torch.long)
flipped_doc_author_pairs = doc_author_pairs[[1,0], :]
data['author', 'to', 'document'].edge_index = torch.tensor(flipped_doc_author_pairs, dtype=torch.long)

# data['author', 'to', 'document'].edge_index = torch.tensor(np.flip(doc_author_pairs, axis=0), dtype=torch.long)

# if doc_tag_pairs.size > 0:
#     data['document', 'to', 'tag'].edge_index = torch.tensor(doc_tag_pairs, dtype=torch.long)
#     data['tag', 'to', 'document'].edge_index = torch.tensor(np.flip(doc_tag_pairs, axis=0), dtype=torch.long)
if doc_tag_pairs.size > 0:
    data['document', 'to', 'tag'].edge_index = torch.tensor(doc_tag_pairs, dtype=torch.long)
    flipped_doc_tag_pairs = doc_tag_pairs[[1,0], :]  # This re-orders rows without np.flip
    data['tag', 'to', 'document'].edge_index = torch.tensor(flipped_doc_tag_pairs, dtype=torch.long)

torch.save(data, 'hetero_data_agnews.pt')
print("HeteroData built and saved.")
