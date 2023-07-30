# Book Recommender System

<p align="center">
  <img src="/assets/download.png" width="80%">
</p>

Welcome to the Book Recommender System project! This repository contains code for a book recommender system that utilizes a content-based filtering approach to recommend similar books based on their textual features.

## Installation

1. Clone the repository:

`git clone https://github.com/your_username/Book-Recommender-System.git`

2. Install the required dependencies:

`pip install pandas scikit-learn`

## Usage

1. Import the necessary libraries:

`
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity`

2. Load the dataset and preprocess it:
   `df = pd.read_csv('BX-Books.csv', on_bad_lines="skip", encoding='latin-1', sep=';', low_memory=False)
df = df.drop_duplicates(subset='Book-Title')`

# Randomly sample 15,000 rows

`sample_size = 15000
df = df.sample(n=sample_size, replace=False, random_state=490)`

# Clean text data

`def clean_text(author):
    result = str(author).lower()
    return result.replace(' ', '')`

`df['Book-Author'] = df['Book-Author'].apply(clean_text)`
`df['Book-Title'] = df['Book-Title'].str.lower()`
`df['Publisher'] = df['Publisher'].str.lower()`

# Combine text columns

`df['data'] = df[df.columns[1:]].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)`

# Vectorize and calculate similarities:

`vectorizer = CountVectorizer()`
`vectorized = vectorizer.fit_transform(df['data'])`
`similarities = cosine_similarity(vectorized)`
`df = pd.DataFrame(similarities, columns=df['Book-Title'], index=df['Book-Title']).reset_index()`

# Input book for recommendation:

`input_book = "the general prologue to the canterbury tales"`

# Get recommendations:

`recommendations = pd.DataFrame(df.nlargest(100,input_book)['Book-Title'])`
`recommendations = recommendations[recommendations['Book-Title'] != input_book]`
`print(recommendations)`

## Recommendations

Based on the provided input book "the general prologue to the canterbury tales," the system recommends the following books:

- Book 1
- Book 2
- Book 3

## Contribution

We welcome contributions to enhance the project and encourage you to participate! Here's how you can contribute:

1. Fork the repository.

2. Create a new branch: git checkout -b feature/your_feature.

3. Make changes and commit them: git commit -m "Add your changes".

4. Push the changes to your forked repository: git push origin feature/your_feature.

5. Create a pull request to the main repository's main branch.

Please ensure that your contributions adhere to our code of conduct and are aligned with the project's goals.
