import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet') # download for lemmatization
nltk.download('punkt')

from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pickle


def load_data(database_filepath):
    """
    Load data from sql database file.
    Args:
        database_filepath (str): The file path for sql lite database file.
    Returns:
        Tuple(X, Y, category_names): the features (X), the multi-output labels (Y), and the category names
    """
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('resp_messages', con=engine)
    X = df['message']
    Y = df.iloc[:, 4:]
    
    return (X, Y, Y.columns.tolist())

def tokenize(text):
    """
    tokenize the input text.
    Args:
        text (str): The raw text to be tokenized.
    Returns:
        str: the cleaned and tokenized tokens
    """
    # regex for website (removal)
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # make it lower case and remove punctuations 
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # tokenize text
    words = word_tokenize(text)
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in words:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tok = lemmatizer.lemmatize(clean_tok, pos='v')
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model(tokenizer):
    """
    build the entire pipeline model, including the text preperation, and classification.
    Note that this function does *NOT* contain the GridSearch functions, which is included
    inside the jupyter notebook part, to speed up the runnning.
    Args:
        tokenizer: the tokenizer function used by CountVectorizer
    Returns:
        model: the pipeline model built
    """
    # the classification model
    model = MultiOutputClassifier(RandomForestClassifier(n_jobs=-1))
    pipeline = Pipeline([
            ('vect', CountVectorizer(tokenizer=tokenizer)),
            ('tfidf', TfidfTransformer()),
            ('clf', model)
            ])
    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate the model, given X_test and Y_test with the category names
    Args:
        model(Pipeline): the model
        X_test: feature test set
        Y_test: label test set
        category_names: the name for each output category
    Returns:
        dict: the generated report
    """
    # Make predictions
    Y_pred = model.predict(X_test)
    return classification_report(Y_test, Y_pred, target_names=category_names)

def save_model(model, model_filepath):
    """
    save the model to pickle file
    Args:
        model(Pipeline): the model
        X_test: feature test set
        Y_test: label test set
        category_names: the name for each output category
    Returns:
        dict: the generated report
    """
    with open(model_filepath, 'wb') as file:  
        pickle.dump(model, file)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model(tokenize)
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        performance = evaluate_model(model, X_test, Y_test, category_names)
        print(performance)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()