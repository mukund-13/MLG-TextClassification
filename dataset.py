import os
import pickle
import numpy as np
from datasets import load_dataset
import pandas as pd

# 1. Load AG News dataset
ds = load_dataset("ag_news")

# ds['train'] and ds['test'] are available, each record is something like:
# {'text': "Some news article...", 'label': 2}

# 2. Define synthetic metadata generation functions

def generate_author_id(num_authors=100):
    # Return a random author ID from 0 to num_authors-1
    return np.random.randint(0, num_authors)

def generate_tags_from_text(text, num_tags=2):
    # A simple heuristic: take the first num_tags words as "tags"
    # In practice, you'd have a better strategy.
    words = text.lower().split()
    # Ensure at least num_tags words. If not, repeat some words or use <no_tag>
    if len(words) < num_tags:
        words = words + ["notag"]*(num_tags - len(words))
    # Just pick first num_tags words
    tags = words[:num_tags]
    return tags

def add_metadata(example):
    example['author_id'] = generate_author_id()  # random author
    example['tags'] = generate_tags_from_text(example['text'])
    return example

# 3. Map the dataset to add metadata
ds = ds.map(add_metadata)

# Now ds['train'] and ds['test'] have 'author_id' and 'tags' fields as well.
# For instance:
# {'text': "Some news ...", 'label': 2, 'author_id': 57, 'tags': ['some', 'news']}

# 4. Split train further if needed (e.g., create minimal supervision subsets).
# Here we just demonstrate saving the processed data as pickle files.

train_df = ds['train'].to_pandas()
test_df = ds['test'].to_pandas()

# Create minimal supervision training set
NUM_LABELED_PER_CLASS = 10
train_small = []
for c in range(4):  # classes are 0 to 3
    class_data = train_df[train_df['label'] == c]
    train_small.append(class_data.sample(NUM_LABELED_PER_CLASS, random_state=42))
train_small =  pd.concat(train_small)
remaining_train = train_df.drop(train_small.index)

# Let's do a simple validation split from remaining_train
from sklearn.model_selection import train_test_split
vali_df, unlabeled_df = train_test_split(remaining_train, test_size=0.8, random_state=42)

# 5. Save the processed data
os.makedirs("processed_data", exist_ok=True)
with open("processed_data/train.pkl", "wb") as f:
    pickle.dump(train_small, f)
with open("processed_data/val.pkl", "wb") as f:
    pickle.dump(vali_df, f)
with open("processed_data/test.pkl", "wb") as f:
    pickle.dump(test_df, f)
with open("processed_data/unlabeled.pkl", "wb") as f:
    pickle.dump(unlabeled_df, f)

# Save label2id mapping
label2id = {0:"World", 1:"Sports", 2:"Business", 3:"Sci/Tech"}
meta = {
    'label2id': label2id
}
with open("processed_data/mappings.pkl", "wb") as f:
    pickle.dump(meta, f)

print("Synthetic metadata added and data saved in 'processed_data' directory.")
