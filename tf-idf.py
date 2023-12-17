import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from sqlalchemy import create_engine
import pandas as pd
from nltk.corpus import wordnet as wn
import os
import re

# Function to summarize text
def summarize(text):
    stop_words = set(stopwords.words("english"))
    text = text.lower()
    text = text.strip()
    text = re.sub(r'[^\w\s]+', '', text)
    word_tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    # Remove stopwords and lemmatize
    filtered_tokens = []
    for token in word_tokens:
        if token not in stop_words:
            filtered_token = lemmatizer.lemmatize(token)
            filtered_tokens.append(filtered_token)

    cleaned_text = " ".join(filtered_tokens)

    # Vectorize text
    vectorizer = TfidfVectorizer()
    tf_matrix = vectorizer.fit_transform([cleaned_text])
    dict_of_tokens = {i[1]:i[0] for i in vectorizer.vocabulary_.items()}
    tfidf_vectors = []
    for row in tf_matrix:
        tfidf_vectors.append({dict_of_tokens[column]: value for (column, value) in zip(row.indices, row.data)})

    doc_sorted_tfidfs = []  # list of doc features each with tfidf weight
    # sort each dict of a document
    for dn in tfidf_vectors:
        newD = sorted(dn.items(), key=lambda x: x[1], reverse=True)
        newD = dict(newD)
        doc_sorted_tfidfs.append(newD)
    tfidf_kw = []
    for doc_tfidf in doc_sorted_tfidfs:
        ll = list(doc_tfidf.keys())
    tfidf_kw.append(ll)
    summary = " ".join(tfidf_kw[0][0:5])
    return summary

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

text_file_path = os.path.join(os.path.dirname(__file__) , "modules", "state_of_the_union.txt")
text = open(text_file_path, 'r').read()
text = text.replace('\n','')

# Summarize
summary = summarize(text)
print(f'Summary: {summary}')

# Print out hypernyms
hypernym_chains = {}

for word in summary.split(' '):
    # Get the Synset for the word
    print(word)
    ss = wn.synsets(word)[0]
    print(ss)
    chain = [ss]
    while ss.hypernyms():
        ss = ss.hypernyms()[0]
        chain.append(ss)
    hypernym_chains[word] = [s.name().split(".")[0] for s in chain]

# Print the hypernym chains
for word, chain in hypernym_chains.items():
    print(f"{word}: {' -> '.join(chain)}")

# Save to database
path = 'db/tfidf-summaries.db'
scriptdir = os.path.dirname(__file__)
db_path = os.path.join(scriptdir, path)
os.makedirs(os.path.dirname(db_path), exist_ok=True)

engine = create_engine(f'sqlite:///{db_path}')
summary_dict = {'text': [text], 'summary': [summary]}
df = pd.DataFrame(summary_dict)
df.to_sql('summaries', con=engine, if_exists='append', index=False)

# Retrieve summary
with engine.connect() as conn:
    df = pd.read_sql_table('summaries', con=conn)
    print(f'Retrieved summary: {df["summary"].values[0]}')

