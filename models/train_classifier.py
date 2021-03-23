import sys

# import libraries
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
nltk.download('stopwords')

import re
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

import numpy as np
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier

from nltk.corpus import stopwords
import sqlite3
from sqlalchemy import create_engine

import pickle

def load_data(database_filepath):
    
    """Load datatable from SQL database and sets X and Y 
    Args:
        database_filepath (str): filepath of the sqlite database
    Returns:
        X (pandas dataframe): input data (messages)
        Y (pandas dataframe): Classification labels
        category_names (list): List of the category names
    """    
 
   
 #   e = create_engine('sqlite:///' + database_filepath) 
    e = create_engine('sqlite:///{}'.format(database_filepath))

   
    df = pd.read_sql_table(table_name='Messages', con=e)
    #df = pd.read_sql_table(table_name='Message', con=e)
    #df = df[:10000]
    df = df
    X = df['message']
    Y = df.loc[:,~df.columns.isin(['index','id','message','original','genre'])]
    return X,Y,Y.columns   

def tokenize(text):
    
    """ Steps using NLTK library to preprocess the text data
        Functions: 
            Tokenize Text 
            remove_stopwords
            lemmatize_text            
        Returns:
            Clean and preprocessed text 
    """  
    
    # Search for all non-letters and replace with spaces
    text = re.sub("[^a-zA-Z]"," ", str(text))
    
    tokens = []
    english_stops = stopwords.words('english')
    tokens = word_tokenize(text)
    tokens_sem_stop = [word for word in tokens if not word in english_stops]
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens_sem_stop:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens 



def build_model():
    """Returns the model
    Args:
        None
    Returns:
        model (scikit-learn KNeighborsClassifier()): Fit the model
    """
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),       
        ('clf', MultiOutputClassifier(KNeighborsClassifier()))
    ])
    
      #Uncomment for additional parameters
    grid_params = {
         'clf__estimator__n_neighbors': [3,5,7],
         'clf__estimator__weights': ['uniform','distance'],
        # 'tfidf__use_idf': (True, False),
         #'clf__estimator__metric': ['euclidean','manhattan']
    }
    print(sorted(pipeline.get_params().keys()))
    #Create Model
    GS = GridSearchCV(
        pipeline, 
        grid_params,
        verbose=1,
        cv=2,        
        n_jobs= -1,
        return_train_score=True
    )
   
  
    return GS
    
  


def evaluate_model(model, X_test, Y_test, category_names):
    
    """Prints multi-output classification report results 
    Args:
        model (dataframe): the scikit-learn fitted model
        X_text (dataframe): The X test set
        Y_test (dataframe): the Y test classifications
        category_names (list): the category names
    Returns:
        None
    """
    
    from sklearn.metrics import classification_report

    y_pred = model.predict(X_test)
    df_pred = pd.DataFrame.from_records(y_pred)
    
    cont = 0
    for category in category_names:
        y_true = Y_test[category]
        y_pred = df_pred[cont]
        print(category)
        print(classification_report(y_true, y_pred, digits=2,labels=np.unique(y_pred)))
        cont +=1

def save_model(model, model_filepath):
    
    """dumps the model to the given filepath
    Args:
        model (scikit-learn model): The fitted model
        model_filepath (string): the filepath to save the model to
    Returns:
        None
    """
    #joblib.dump(model, model_filepath)
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
