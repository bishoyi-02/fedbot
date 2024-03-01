import json
f = open('ChatGPT_Dataset.json')
data = json.load(f)

preprocessed_data=[]

for intent in data['intents']:
    # print(intent["patterns"])
    for i,question in  enumerate(intent["patterns"]):
        preprocessed_data.append({"Question": question,
                                   "Answer" : intent["responses"][i]})
 
print(preprocessed_data[:10])
f.close()