"""
This script performs text preprocessing on the XML files found in the
input dataset.csv file and outputs the preprocessed text contents to a
file called preprocessed.csv

REFERENCE: https://medium.com/@bedigunjit/simple-guide-to-text-classification-nlp-using-svm-and-naive-bayes-with-python-421db3a72d34
"""

import xml.etree.cElementTree as et
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn
from collections import defaultdict
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import stopwords

import os
import pandas as pd

# Uncomment below lines the first time you run this script on another computer
# import nltk
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('averaged_perceptron_tagger')

np.random.seed(500)

# Pos tags to be used in lemmatization
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

wnl = WordNetLemmatizer()

# Top 10 publications based on number of articles
publications = ['National Intelligencer (Washington, DC)',
                'North American (Philadelphia, PA)',
                'Milwaukee Daily Sentinel (Milwaukee, WI)',
                'Bangor Daily Whig and Courier (Bangor, ME)',
                'Boston Daily Advertiser (Boston, MA)',
                'Cleveland Daily Herald (Cleveland, OH)',
                'Daily Evening Bulletin (San Francisco, CA)',
                'Rocky Mountain News (Denver, CO)',
                'Galveston Daily News (Houston, TX)',
                'Lowell Daily Citizen (Lowell, MA)']


def getTextContent(filename):
    """Extracts all relevant text content from a given XML file"""
    tree = et.parse(filename)
    root = tree.getroot()

    articles = []

    for article in root.iter('ocrText'):
        articles.append(article.text)

    print("Preprocessing", filename)
    return preprocessText(' '.join(articles))


def preprocessText(text: str):
    # Lowercase text
    text = text.lower()

    # Tokenization: break into set of words
    text = word_tokenize(text)

    # Remove stop words and perform word stemming
    finalWords = []

    # Try pos_tag_sent for more efficient tagging of more than one sentence.
    for word, tag in pos_tag(text):
        # Make sure it isn't a stop word and that it's in the alphabet
        if word not in stopwords.words("english") and word.isalpha():
            lemmatized = wnl.lemmatize(word, tag_map[tag[0]])
            finalWords.append(lemmatized)
    # print(finalWords)
    return finalWords


def main():
    df = pd.read_csv("articles.csv")
    # Uncomment two lines below to preprocess in one go
    # df["Text"] = df["Location"].apply(getTextContent)
    # df.to_csv("preprocessed.csv")

    # Loop to perform preprocessing in chunks since it takes hours
    for i in range(13000, len(df), 500):
        chunk = df.iloc[i: i + 500]  # chunks of 500 rows
        chunk["Text"] = chunk["Location"].apply(getTextContent)
        chunk.to_csv("Preprocessed/output_"+str(i)+".csv")

        print("Finished", i, i+500)
        res = input("Press enter to continue to the next" +
                    str(i+500) + "chunk or q to quit for now:")
        if res.lower() == 'q':
            print("Start the loop from", i+500, "next time to continue")
            break

    print("Finished preprocessing and saved to preprocessed.csv")


if __name__ == "__main__":
    main()
