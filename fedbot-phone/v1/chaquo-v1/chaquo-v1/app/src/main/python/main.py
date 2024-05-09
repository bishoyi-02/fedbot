# from six.moves import input
import nltk
import tensorflow as tensorF
import json
import string
import random
import numpy as np
import os

from nltk.stem import WordNetLemmatizer
from os.path import dirname,join
from os import path
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout


def main():
    nltk.download("punkt")
    nltk.download("wordnet")
    nltk.download('omw-1.4')
    def process_data():
        print("Processing data...")
        f=open(join(dirname(__file__),'intents.json'))
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


        return x,y,newWords,ourClasses,data,lm

    def evaluate_model(model, x_test, y_test):
        score = model.evaluate(x_test, y_test, verbose=0)
        print('Test loss:', score[0])
        print('Test accuracy:', score[1])

    def save_local_model_update(model):
        mod1 = model.get_weights()
        save_dir=join(dirname(__file__),'../assets/local_model')
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # os.makedirs(save_dir)
        if os.path.exists(join(save_dir,'mod1.npy')):
            print("Old weights exists")
        else:
            print("Old weights doesn't exists\nSaving new weights")
        np.save(join(save_dir,'mod1.npy'), mod1)
        print("Local model update written to local storage!")

    def build_model(x,y):
        # print(dirname(__file__))
        dir_path=join(dirname(__file__),"../assets/model_update/")
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        model_location=join(dir_path,'agg_model.h5')
        # print(model_location)
        if path.exists(model_location):
            print("Agg Model Exists...\nLoading Model...")
            model = tensorF.keras.models.load_model(model_location,compile=False)
            # print('here0')
        else:
            print("No agg model found!\nBuilding model...")
            model =Sequential()
            iShape = (len(x[0]),)
            oShape = len(y[0])
            print(iShape,oShape)
            # print(x[0])
            # print(y[0])
            model.add(Dense(units=128, activation='relu',input_shape=iShape))
            model.add(Dropout(0.5))
            model.add(Dense(64,activation="relu"))
            model.add(Dropout(0.3))
            model.add(Dense(oShape,activation='softmax'))
        # print('here_if')
        md = tensorF.keras.optimizers.Adam(learning_rate=0.01,decay=1e-6)
        # print('here1')
        model.compile(loss='categorical_crossentropy',
                            optimizer=md,
                            metrics=['accuracy'])
        # print('here2')
        model.fit(x,y,epochs=100,verbose=1)
        # print('here3')
        print("Done Training")
        return model

    def ourText(text,lm):
      newTokens = nltk.word_tokenize(text)
      newTokens = [lm.lemmatize(word) for word in newTokens]
      return newTokens
#
    def wordBag(text, vocab,lm):
      newTokens = ourText(text,lm)
      bagOfWords = [0]*len(vocab)
      for w in newTokens:
        for i,word in enumerate(vocab):
          if word ==w:
            bagOfWords[i]=1
      return np.array(bagOfWords)
#
    def PClass(text,vocab,labels,model,lm):
      bagOfWords = wordBag(text,vocab,lm)
      ourResult = model.predict(np.array([bagOfWords]))[0]
      # print(ourResult)
      newThresh=0.0
      yp = [[i,res]for i,res in enumerate(ourResult) if res>newThresh]
      yp.sort(key=lambda x:x[1],reverse=True)
      newList=[]
      for r in yp:
        newList.append(labels[r[0]])
      return newList
#
#
    def getRes(firstList,fJson):
      # print(firstList)
      tag = firstList[0]
      listOfIntents = fJson['intents']
      for i in listOfIntents:
        if i['tag']==tag:
          ourResult= random.choice(i['responses'])
          break
      return ourResult
#

    def addPatterns(firstList,userInput):
      tag=firstList[0]
      dir_name=join(dirname(__file__),'intents.json')
      with open(dir_name, 'r') as file:
        data = json.load(file)
      listOfIntents=data['intents']
      for i in listOfIntents:
        if i['tag']==tag:
          i['patterns'].append(userInput)
          with open(dir_name, 'w') as file:
            json.dump(data, file,indent=2)
      print("Pattern added")
      return

    def initialize_model(model,newWords,classes,data,lm):
        print("Initializing Model...")
        while True:
            newMessage = input("You : ")
            if(newMessage=="Program interrupted by user"):
                print('Model Retraining...')
                break
            intents = PClass(newMessage,newWords,classes,model,lm)
            # print(intents)
            ourResult = getRes(intents,data)
            print("Chatbot : ",ourResult)
            response=input("Do you find the response relevant? Press [Y/N]")
            # if("y" ==response or "Y" ==response):
            #   addPatterns(intents,newMessage)

    def train_and_init():
        x,y,newWords,classes,data,lm =  process_data()
        model = build_model(x,y)
        evaluate_model(model, x, y)
        save_local_model_update(model)
        initialize_model(model,newWords,classes,data,lm)

    train_and_init()

