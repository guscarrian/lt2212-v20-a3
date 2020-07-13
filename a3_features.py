import os
import sys
import argparse
import numpy as np
import pandas as pd
# Whatever other imports you need
import glob

from nltk.corpus import stopwords
from sklearn.feature_extraction import DictVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = stopwords.words('english')


def get_words(text):
    words = []
    words_split = text.split(' ')
    for word in words_split:
        if word.isalpha(): #and word not in stopwords: 
            words.append(word)
        #print(words)   
    return words


def tokenize_doc(inputdir):
    authors = [] #List of authors names. One string per each file inside each folder.
    file_pathname = [] # List of files names. One pathname per file
    tokenized_docs = []
    all_files = glob.glob("{}/*/*.*".format(inputdir))
    print(f"files: {len(all_files)}")
    for file in all_files:
        authors.append(file.split("/")[-2])
        file_pathname.append("{}/{}".format(authors[-1], file.split("/")[-1]))
        with open(file, "r") as f:
            #Get a list of words and transform it into a single string using ' '.join
            #Then insert it into tokenized_docs.
            #The result is a list of strings, one string per file.
            tokenized_docs.append(' '.join(get_words(f.read())))
    #print(authors)
    #print(file_pathname)
    print(len(tokenized_docs))

    return authors, file_pathname, tokenized_docs

#counting words + deleating words occurring less than x times    
def word_counter(tokenized_docs):
    counter_list = []
    word_count = {}
    for word in tokenized_docs:
        if word not in word_count:
            word_count[word] = 1
        else:
            word_count[word] += 1
    counter_list.append(word_count)
    #print(f'counter list: {counter_list}')
    return counter_list

def extract_features(tokenized_docs):
    counter = word_counter(tokenized_docs) 

    vectorizer = TfidfVectorizer()
    vectorized = vectorizer.fit_transform(tokenized_docs)

    #print('vectorized:' , vectorized)
    return vectorized


def reduce_dim(features,n):
    svd = TruncatedSVD(n_components=n)
    dim_red = svd.fit_transform(features)
    #print('dim_red: ' , dim_red)
    return dim_red


def create_dataframe(features, authors):
    #Creating dataframe
    df = pd.DataFrame(data=features)

    #Inserting column named Author
    df.insert(0, "Author", authors)

    return df
    
def insert_test_train(df, number_authors, testsize):
    #Create a random list of strings "Train"/"Test" with the proportion 0.8/0.2 respectively
    random_test_train_list = np.random.choice(["Train", "Test"], number_authors, p=[1-(testsize/100), testsize/100]).tolist()
    #Inserting a column with title "Train/Test"; as a value, we generate a random list of strings
    #"Train" -> 80% and "Test" -> 20%
    #This way, we have a column with values Test/Train randomized
    df.insert(0, "Train/Test", random_test_train_list)
    
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert directories into table.")
    parser.add_argument("inputdir", type=str, help="The root of the author directories.")
    parser.add_argument("outputfile", type=str, help="The name of the output file containing the table of instances.")
    parser.add_argument("dims", type=int, help="The output feature dimensions.")
    parser.add_argument("--test", "-T", dest="testsize", type=int, default="20", help="The percentage (integer) of instances to label as test.")

    args = parser.parse_args()

    print("Reading {}...".format(args.inputdir))
    # Do what you need to read the documents here.
    authors, file_pathname, tokenized_docs = tokenize_doc(args.inputdir)
    
    features_array = extract_features(tokenized_docs)
    reduced_features = reduce_dim(features_array, args.dims)
    
    print(f"authors:{len(authors)} reduced_features: {len(reduced_features)}")
    
    print("Constructing table with {} feature dimensions and {}% test instances...".format(args.dims, args.testsize))
    # Build the table here.
    df = create_dataframe(reduced_features, authors)
    df = insert_test_train(df=df, number_authors=len(authors),testsize=args.testsize)
    
    print("Writing to {}...".format(args.outputfile))
    # Write the table out here.
    df.to_csv(args.outputfile, index=False)
    print("Done!")
    
