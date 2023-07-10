# Recommender-system
A book recommendation machine learning system

# Book Recommender System

This repository contains code for a book recommender system. The system is based on a dataset of books and uses a content-based filtering approach to recommend similar books based on their textual features.

## Installation

1. Clone the repository:


2. Install the required dependencies:


## Usage

1. Import the necessary libraries:

```python
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
df = pd.read_csv('BX-Books.csv', on_bad_lines="skip", encoding='latin-1', sep=';', low_memory=False)
# Remove duplicate book titles
df = df.drop_duplicates(subset='Book-Title')

# Randomly sample 15,000 rows
sample_size = 15000
df = df.sample(n=sample_size, replace=False, random_state=490)

# Clean text data
def clean_text(author):
    result = str(author).lower()
    return result.replace(' ', '')

df['Book-Author'] = df['Book-Author'].apply(clean_text)
df['Book-Title'] = df['Book-Title'].str.lower()
df['Publisher'] = df['Publisher'].str.lower()

# Combine text columns
df['data'] = df[df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)
vectorizer = CountVectorizer()
vectorized = vectorizer.fit_transform(df['data'])
similarities = cosine_similarity(vectorized)
df = pd.DataFrame(similarities, columns=df['Book-Title'], index=df['Book-Title']).reset_index()
input_book = "the general prologue to the canterbury tales"
recommendations = pd.DataFrame(df.nlargest(100, input_book)['Book-Title'])
recommendations = recommendations[recommendations['Book-Title'] != input_book]
print(recommendations)
