 # ML Project
 # The main project file that takes input, performs linear regression, 
 # feature extracts, and assess sucess
 #
 # Author: Aaron Ye
 # Version: 1.0


#Run via "python3 CS4824/project/wordProc.py"
import spacy
from math import exp
import random

random.seed(1)
nlp = spacy.load("en_core_web_sm")

# Functions for Logistic Regression, heavily based on Hw2
def log(x):
    val = 1/(1 + exp(-x))
    return val

def dot(x, y):
    val = 0
    length = len(x)
    for cur in range(length):
        val += x[cur] * y[cur]
    return val

def predict(model, point):
    result = dot(model, point['features'])
    val = log(result)
    return val

# Functions for evluation
def accuracy(data, predictions):
    counter = 0
    correct = 0
    for x in data:
        if (predictions[counter] >= 0.5 and x["label"]) or (predictions[counter] < 0.5 and not x["label"]):
            correct += 1
        counter += 1
    return float(correct)/len(data)

def precision(data, predictions):
    counter = 0
    truePositive = 0
    falsePositive = 0
    for x in data:
        if (predictions[counter] >= 0.5 and x["label"]):
            truePositive += 1
        elif (predictions[counter] >= 0.5 and not x["label"]):
            falsePositive += 1
        counter += 1
    return float(truePositive)/(truePositive + falsePositive)

def recall(data, predictions):
    counter = 0
    truePositive = 0
    falseNegative = 0
    for x in data:
        if (predictions[counter] >= 0.5 and x["label"]):
            truePositive += 1
        elif (predictions[counter] < 0.5 and x["label"]):
            falseNegative += 1
        counter += 1
    return float(truePositive)/(truePositive + falseNegative)

def f1(precision, recall):
    return (2*precision * recall)/(precision + recall)


# Actual Model
def initModel(k):
    return [random.gauss(0, 1) for x in range(k)]

def train(data, epochs, rate, lam):
    model = initModel(len(data[0]['features']))
    for t in range(epochs):
        for i in range(len(data[0]['features'])):
            total = 0
            for l in range(len(data)):
                total += data[l]['features'][i]*(data[l]["label"]-predict(model, data[l]))
            newTotal = total*rate
            model[i] = model[i] - (rate*lam*model[i]) + newTotal
    return model

# Features Extraction:
def featureExtract(sentence, score):
    data = []
    for r in range(len(sentence)):
        point = {}
        point["label"] = (score[r] == "1\n")
        features = []
        features.append(1.)
        # Includes: number of punctuation, pos/neg count, stop words, and number of purely capitla words
        puncCount = 0
        posCount = 0
        negCount = 0
        stopCount = 0
        hasCaps = False
        posWords = ["good", "excellent", "great", "must", "impressed", "impress", "highly", "recommend", "recommended", "love", "loved", "best", "ideal", "well", "nice", "right"]
        negWords = ["no", "problem", "problems", "waste", "wasted", "mere", "odd", "fool", "fooled", "misleading", "unacceptable", "unusable", "hate", "poor", "worthless"]
        for token in sentence[r]:
            if token.pos_ == "PUNCT":
                puncCount += 1
            for word in posWords:
                if token.text.lower() == word:
                    posCount += 1
            for word in negWords:
                if token.text.lower() == word:
                    negCount += 1
            if token.is_upper:
                hasCaps = True
            if token.is_stop:
                stopCount += 1
        #features.append(stopCount <= 2)
        features.append(posCount >= 1)
        features.append(negCount <= 1)
        #features.append(posCount/len(sentence[r]))
        #features.append(negCount/len(sentence[r]))
        features.append(hasCaps)
        point['features'] = features
        data.append(point)
    return data

# Fine tuning linear regression model
def submission(data):
    random.seed(1)
    return train(data, 400, 1e-3, 1e-2)


# Method for Grabbing Data and converting to string array
def dataExtraction(fileName):
    file = open(fileName, "r")
    workinArray = file.readlines()
    return workinArray

# Essentially the Main Function
def evaluate(fileName):
    sentenceArray = []
    rawData = dataExtraction(fileName)
    trainIndex = len(rawData)*(2/3) # Index to seperate training data with valid data

    # Training/Valid Arrays for Sentence and Scores used later
    trainSentence = []
    validSentence = []
    trainScore = []
    validScore = []

    # Splitting the sentence into the sentence and score
    counter = 0
    for line in rawData:
        splitter = line.split("\t")
        sentenceArray.append(splitter[0])
        if counter <= trainIndex:
            trainScore.append(splitter[1])
        else:
            validScore.append(splitter[1])
        counter += 1

    # Utilizing spaCY converting all sentences into tokenized ones with features that can be utilized
    #Tokenization
    counter = 0
    for line in sentenceArray:
        if counter <= trainIndex:
            trainSentence.append(nlp(line))
        else:
            validSentence.append(nlp(line))
        counter += 1

    # Based on HW2
    train_data = featureExtract(trainSentence, trainScore)
    valid_data = featureExtract(validSentence, validScore)
    model = submission(train_data)
    trainPredictions = [predict(model, p) for p in train_data]
    print("Training Accuracy:", accuracy(train_data, trainPredictions))
    validPredictions = [predict(model, p) for p in valid_data]
    print("Validation Accuracy:", accuracy(valid_data, validPredictions))
    trainingPrecision = precision(train_data, trainPredictions)
    validPrecision = precision(valid_data, validPredictions)
    print("Training Precision:", trainingPrecision)
    print("Validation Precision:", validPrecision)
    trainingRecall = recall(train_data, trainPredictions)
    validRecall = recall(valid_data, validPredictions)
    print("Training Recall:", trainingRecall)
    print("Validation Recall:", validRecall)
    print("Training F1:", f1(trainingPrecision, trainingRecall))
    print("Validation F1:", f1(validPrecision, validRecall))
    print("") #Extra new line prints for spacing purposes
    print("")

# Model running on the 3 different datasets
print("Testing Amazon:")
evaluate("/home/ugrads/majors/aarony/CS4824/project/amazon_cells_labelled.txt")
print("Testing IMDB:")
evaluate("/home/ugrads/majors/aarony/CS4824/project/imdb_labelled.txt")
print("Testing Yelp:")
evaluate("/home/ugrads/majors/aarony/CS4824/project/yelp_labelled.txt")