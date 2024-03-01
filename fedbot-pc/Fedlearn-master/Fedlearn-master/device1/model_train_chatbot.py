
import json
import string
import random
import nltk
import numpy as np
import tensorflow as tensorF

from nltk.stem import WordNetLemmatizer
from keras.models import load_model
from colorama import Fore,Style
# from pickling import startPickling
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras import Sequential 


from os import path



batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)

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

def build_model(x, y):
    # if there is model update set its weights as model weights
    if path.exists("model_update/agg_model.h5"):
        print("Agg model exists...\nLoading model...")
        model = load_model("model_update/agg_model.h5")
    else:
        print("No agg model found!\nBuilding model...")
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
    model.fit(x,y,epochs=200,verbose=1)
    print("Done Training")

    return model

def evaluate_model(model, x_test, y_test):
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

def save_local_model_update(model):
    mod1 = model.get_weights()
    np.save('local_model/mod1', mod1)
    print("Local model update written to local storage!")

def train():
    x,y,newWords,classes,data,lm =  process_data()
    model = build_model(x,y)
    evaluate_model(model, x, y)
    save_local_model_update(model)
    initialize_model(model,newWords,classes,data,lm)


def ourText(text,lm):
  newTokens = nltk.word_tokenize(text)
  newTokens = [lm.lemmatize(word) for word in newTokens]
  return newTokens

def wordBag(text, vocab,lm):
  newTokens = ourText(text,lm)
  bagOfWords = [0]*len(vocab)
  for w in newTokens:
    for i,word in enumerate(vocab):
      if word ==w:
        bagOfWords[i]=1
  return np.array(bagOfWords)

def PClass(text,vocab,labels,model,lm):
  bagOfWords = wordBag(text,vocab,lm)
  ourResult = model.predict(np.array([bagOfWords]))[0]
  newThresh=0.2
  yp = [[i,res]for i,res in enumerate(ourResult) if res>newThresh]
  yp.sort(key=lambda x:x[1],reverse=True)
  newList=[]
  for r in yp:
    newList.append(labels[r[0]])
  return newList  


def getRes(firstList,fJson):
  tag = firstList[0]
  listOfIntents = fJson['intents']
  for i in listOfIntents:
    if i['tag']==tag:
      ourResult= random.choice(i['responses'])
      break
  return ourResult

def addPatterns(firstList,userInput):
  tag=firstList[0]
  with open('intents.json', 'r') as file:
    data = json.load(file)
  listOfIntents=data['intents']
  for i in listOfIntents:
    if i['tag']==tag:
      i['patterns'].append(userInput)
      with open('intents.json', 'w') as file:
        json.dump(data, file,indent=2)
  print(f"{Fore.GREEN}Pattern added")
  return


def initialize_model(model,newWords,classes,data,lm):
    print(f"{Fore.YELLOW}Initializing Model...")
    while True:
        newMessage = input(f"{Fore.BLUE}You : ")
        print(f"{Style.RESET_ALL}")
        if(newMessage=="q"):
            return "Model Trained"
        intents = PClass(newMessage,newWords,classes,model,lm)
        ourResult = getRes(intents,data)
        print(f"{Fore.GREEN}Chatbot : ",ourResult)
        response=input(f"{Fore.RED}Do you want to add the Pattern? Press [Y/N]")
        if(response=="y" or response=="Y" ):
          addPatterns(intents,newMessage)










