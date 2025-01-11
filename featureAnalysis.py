 # ML Project
 # The anaylsis program that figures out the effectiveness of averageness
 #
 # Author: Aaron Ye
 # Version: 1.0

import spacy
from math import exp
import random
import numpy as np
from spacytextblob.spacytextblob import SpacyTextBlob
random.seed(1)
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("spacytextblob")
doc = nlp("Bad Quality.")

for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop, token.ent_type_)
print(doc._.blob.polarity)
#test code

# Features Extraction:
def featureExtract(sentence):
    stopCount = 0
    posCount = 0
    negCount = 0
    capCount = 0
    posWords = ["good", "excellent", "great", 
                "must", "impressed", "impress", 
                "highly", "recommend", "love", 
                "loved", "best", "ideal", "well", 
                "nice", "right"]
    negWords = ["no", "problem", "problems", 
                "waste", "wasted", "mere", 
                "odd", "fool", "fooled", 
                "misleading", "unacceptable", 
                "unusable", "hate", "poor", 
                "worthless"]
    for r in (sentence):
        for token in r:
            if token.is_stop:
                stopCount += 1
            for word in posWords:
                if token.text.lower() == word:
                    posCount += 1
            for word in negWords:
                if token.text.lower() == word:
                    negCount += 1
            if token.is_upper:
                capCount += 1
    
    print("There was a total of " + str(capCount/len(sentence)) + " cap words per sentence")
    print("There was a total of " + str(stopCount/len(sentence)) + " stop words per sentence")


# Method for Grabbing Data and converting to string array
def dataExtraction(fileName):
    file = open(fileName, "r")
    workinArray = file.readlines()
    return workinArray

# Essentially the Main Function
def evaluate(fileName):
    print("Evaluating: " + fileName)
    rawData = dataExtraction(fileName)

    # Training/Valid Arrays for Sentence and Scores used later
    trainSentence = []
    validSentence = []

    # Splitting the sentence into good and bad sentiment
    for line in rawData:
        splitter = line.split("\t")
        if splitter[1] == "0\n":
            trainSentence.append(nlp(splitter[0]))
        else:
            validSentence.append(nlp(splitter[0]))

    #Repruposed function name, finds the average of features between good aood and negative sentences
    featureExtract(trainSentence)
    featureExtract(validSentence)


# Model running on the 3 different datasets
evaluate("/home/ugrads/majors/aarony/CS4824/project/amazon_cells_labelled.txt")
evaluate("/home/ugrads/majors/aarony/CS4824/project/imdb_labelled.txt")
evaluate("/home/ugrads/majors/aarony/CS4824/project/yelp_labelled.txt")