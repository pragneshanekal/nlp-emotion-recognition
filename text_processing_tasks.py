import os
import json
import nltk
import gensim
import tensorflow as tf
from google.cloud import storage
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from gensim import corpora, models
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def download_data_from_gcs():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/pragneshanekal/Documents/Everything-Northeastern/\
Coursework/Summer-2024/MLOps/project_development/model-creek-425220-v1-5456036c90c6.json'
    storage_client = storage.Client()
    bucket_name = 'model-creek-425220-v1.appspot.com'
    bucket = storage_client.get_bucket(bucket_name)
    blob_name = 'train.txt'
    blob = bucket.blob(blob_name)
    content = blob.download_as_string().decode('utf-8')
    
    return content

def preprocess_text(content):
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    lines = content.split('\n')
    processed_data = []

    for line in lines:
        if line:
            text = line.strip()
            words = word_tokenize(text)
            words = [lemmatizer.lemmatize(word) for word in words if word.lower() not in stop_words and word.isalpha()]
            processed_data.append(' '.join(words))

    processed_text = '\n'.join(processed_data)
    with open('/tmp/processed_train.txt', 'w') as f:
        f.write(processed_text)
    
    return processed_data

def upload_processed_text_to_gcs():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/pragneshanekal/Documents/Everything-Northeastern/\
Coursework/Summer-2024/MLOps/project_development/model-creek-425220-v1-5456036c90c6.json'
    storage_client = storage.Client()
    bucket_name = 'model-creek-425220-v1.appspot.com'
    bucket = storage_client.get_bucket(bucket_name)
    processed_blob = bucket.blob('processed_train.txt')
    processed_blob.upload_from_filename('/tmp/processed_train.txt')

def analyze_emotions():
    nltk.download('vader_lexicon')
    sid = SentimentIntensityAnalyzer()

    with open('/tmp/processed_train.txt', 'r') as f:
        processed_data = f.read().split('\n')

    emotions = []
    for line in processed_data:
        if line:
            scores = sid.polarity_scores(line)
            emotions.append({'text': line, 'scores': scores})

    with open('/tmp/emotions.json', 'w') as f:
        json.dump(emotions, f)

def upload_emotions_to_gcs():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/pragneshanekal/Documents/Everything-Northeastern/\
Coursework/Summer-2024/MLOps/project_development/model-creek-425220-v1-5456036c90c6.json'
    storage_client = storage.Client()
    bucket_name = 'model-creek-425220-v1.appspot.com'
    bucket = storage_client.get_bucket(bucket_name)
    emotions_blob = bucket.blob('emotions.json')
    emotions_blob.upload_from_filename('/tmp/emotions.json')

def perform_topic_modeling():
    with open('/tmp/processed_train.txt', 'r') as f:
        processed_data = f.read().split('\n')

    tokenized_data = [line.split() for line in processed_data]
    dictionary = corpora.Dictionary(tokenized_data)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    corpus = [dictionary.doc2bow(text) for text in tokenized_data]
    lda_model = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=10)
    topics = lda_model.print_topics(num_words=4)

    with open('/tmp/topics.json', 'w') as f:
        json.dump(topics, f)

def upload_topics_to_gcs():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/pragneshanekal/Documents/Everything-Northeastern/\
Coursework/Summer-2024/MLOps/project_development/model-creek-425220-v1-5456036c90c6.json'
    storage_client = storage.Client()
    bucket_name = 'model-creek-425220-v1.appspot.com'
    bucket = storage_client.get_bucket(bucket_name)
    topics_blob = bucket.blob('topics.json')
    topics_blob.upload_from_filename('/tmp/topics.json')

def tokenize_and_pad_sequences():
    with open('/tmp/processed_train.txt', 'r') as f:
        processed_data = f.read().split('\n')

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(processed_data)
    sequences = tokenizer.texts_to_sequences(processed_data)
    max_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_length, padding='post')
    np.save('/tmp/padded_sequences.npy', padded_sequences)

def upload_padded_sequences_to_gcs():
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/pragneshanekal/Documents/Everything-Northeastern/\
Coursework/Summer-2024/MLOps/project_development/model-creek-425220-v1-5456036c90c6.json'
    storage_client = storage.Client()
    bucket_name = 'model-creek-425220-v1.appspot.com'
    bucket = storage_client.get_bucket(bucket_name)
    sequences_blob = bucket.blob('padded_sequences.npy')
    sequences_blob.upload_from_filename('/tmp/padded_sequences.npy')
