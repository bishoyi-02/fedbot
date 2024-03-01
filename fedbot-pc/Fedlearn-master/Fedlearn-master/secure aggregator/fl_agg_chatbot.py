
import json
import string
import random
import nltk
import numpy as np
import tensorflow as tensorF

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
# from pickling import startPickling
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential 

import glob


def process_data():
    print("Processing data...")
    f=open('intents.json')
    data =json.load(f)
    lm = WordNetLemmatizer()
    ourClasses=[]
    newWords=[]
    docPattern=[]
    docTag=[]

    for intent in data['intents']:
        for pattern in intent['patterns']:
            ourNewTokens = nltk.word_tokenize(pattern)
            newWords.extend(ourNewTokens)
            docPattern.append(pattern)
            docTag.append(intent['tag'])
        if intent['tag'] not in ourClasses:
            ourClasses.append(intent['tag'])       


    newWords = [lm.lemmatize(word.lower()) for word in newWords if word not in string.punctuation]
    newWords = sorted(set(newWords))

    ourClasses=sorted(set(ourClasses))

    trainingData = []
    outEmpty=[0]*len(ourClasses)

    for i,doc in enumerate(docPattern):
        bagOfWords=[]
        text=lm.lemmatize(doc.lower())

        for word in newWords:
            bagOfWords.append(1) if word in text else bagOfWords.append(0)
        outputRow = list(outEmpty)
        outputRow[ourClasses.index(docTag[i])]=1
        trainingData.append([bagOfWords,outputRow])

    random.shuffle(trainingData)
    trainingData = np.array(trainingData,dtype=object)

    x = np.array(list(trainingData[:, 0]))
    y = np.array(list(trainingData[:, 1]))
    return x, y,newWords,ourClasses,data,lm

def load_models():
    arr = []
    models = glob.glob("client_models/*.npy")
    print(models)
    for i in models:
        arr.append(np.load(i, allow_pickle=True))

    return np.array(arr)

def fl_average():
    # FL average
    arr = load_models()
    fl_avg = np.average(arr, axis=0)

    # for i in fl_avg:
    #     print(i.shape)

    return fl_avg


def build_model(avg,x,y):
    iShape = (len(x[0]),)
    oShape = len(y[0])
    model =Sequential()
    model.add(Dense(128,input_shape=iShape,activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64,activation="relu"))
    model.add(Dropout(0.3))
    model.add(Dense(oShape,activation='softmax'))
    md = tensorF.keras.optimizers.Adam(learning_rate=0.01,decay=1e-6)
    model.compile(loss='categorical_crossentropy',
                        optimizer=md,
                        metrics=['accuracy'])
    model.set_weights(avg)
    model.compile(loss='categorical_crossentropy',
                        optimizer=md,
                        metrics=['accuracy'])
    return model


def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def save_agg_model(model):
    model.save("persistent_storage/agg_model.h5")
    print("Model written to storage!")

def model_aggregation():
    x,y,_,_,_,_ =  process_data()
    avg = fl_average()
    # print(avg)
    model = build_model(avg,x,y)
    evaluate_model(model, x, y)
    save_agg_model(model)













