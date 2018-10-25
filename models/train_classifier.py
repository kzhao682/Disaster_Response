import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import sys
import random
import pickle
import warnings

import nltk
import re
from nltk.tokenize import word_tokenize, WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

sys.path.append('/home/kevinzhao/Udacity/data/')
warnings.simplefilter("ignore")

def load_data(database_filepath):
    """
    Load data from SQLite database and split into features and target
    """

    # load data from database
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table('message_categories', engine)

    # drop columns with null
    df = df[~(df.isnull().any(axis=1))|((df.original.isnull())&~(df.offer.isnull()))]

    # split features and targets
    X = df['message']
    y = df.iloc[:,4:]
    categories = y.columns

    return X, y, categories

def tokenize(text):
    """
    Remove capitalization and special characters and lemmatize texts
    """    

    # get tokens from text
    tokens = WhitespaceTokenizer().tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    # clean tokens
    processed_tokens = []
    for token in tokens:
        token = lemmatizer.lemmatize(token).lower().strip('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
        token = re.sub(r'\[[^.,;:]]*\]', '', token)
        
        # add token to compiled list if not empty
        if token != '':
            processed_tokens.append(token)
        
    return processed_tokens

def compute_text_length(data):
    """
    Compute the character length of texts
    """
    return np.array([len(text) for text in data]).reshape(-1, 1)

def build_model():
    """
    Build model with a pipeline
    """

    # create pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([('text', Pipeline([('vect', CountVectorizer(tokenizer=tokenize)),
                                                     ('tfidf', TfidfTransformer()),
                                                     ])),
                                  ('length', Pipeline([('count', FunctionTransformer(compute_text_length, validate=False))]))]
                                 )),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))])

    # use GridSearch to tune model with optimal parameters
    parameters = {'features__text__vect__ngram_range':[(1,2),(2,2)],
            'clf__estimator__n_estimators':[50, 100]
             }
    model = GridSearchCV(pipeline, parameters)

    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Show precision, recall, f1-score of model scored on testing set
    """    
    
    # make predictions with model
    Y_pred = model.predict(X_test)

    # print scores
    print(classification_report(Y_test.iloc[:,1:].values, np.array([x[1:] for x in Y_pred]), 
        target_names=category_names))


def save_model(model, model_filepath):
    """
    Pickle model to designated file
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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
