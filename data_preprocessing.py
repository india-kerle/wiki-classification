import glob
import nltk
import string
import pandas as pd
import numpy as np
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer 


def get_the_data(file_dir):
    '''takes as input file directory and outputs two different pandas
    dataframes: test and train.'''
    
    relevant_files = glob.glob(file_dir + "/*.csv")
    
    train_test = []
    for file in relevant_files:
        df = pd.read_csv(file, encoding = "utf-8")
        train_test.append(df)

    return train_test 

def data_preprocessing(text_column):
    '''takes as input the text data pandas column and outputs cleaned text column'''
    stemmer = PorterStemmer()
    stopwords = nltk.corpus.stopwords.words('english')
    remove = string.punctuation

    text_column = (
        text_column
        .str.lower() #to lowercase
        .str.replace(r"[{}]".format(remove.replace("!", "")), '') #remove punct expect ! 
        .str.replace(r'http\S+', '') #remove urls
        .str.replace(r'_\S+', '') 
        .str.replace(r'\d+', '') #remove numbers
        .apply(lambda x: [word for word in x.split() if word not in stopwords]) #remove stopwords
        .apply(lambda x: [stemmer.stem(word) for word in x]) #stem words
        .apply(lambda x: ' '.join(x))
    )

    return text_column

def vectorise(train, test, min_df, max_df): 
    '''takes as input train, test, the minimum document frequency and maximum document frequency and outputs vectorised X train and X test features labels.'''
    train_text = train.processed_text
    test_text = test.processed_text 
    all_text = pd.concat([train_text, test_text])
    
    vectoriser = TfidfVectorizer(min_df = min_df, max_df = max_df)
    
    vectoriser.fit(all_text)
    X_train = vectoriser.transform(train_text)
    X_test = vectoriser.transform(test_text)
    
    return X_train.todense(), X_test.todense(), vectoriser

def y_target(y_target):
    '''takes as input y_target pandas column and outputs y_target array.'''

    return np.array(y_target)

