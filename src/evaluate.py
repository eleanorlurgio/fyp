import numpy as np
import pandas as pd
from scipy.special import softmax
import model

def import_eec():
    # import EEC data set
    df = pd.read_csv('src\Equity-Evaluation-Corpus.csv', usecols=["Sentence", "Template","Person","Gender","Race","Emotion","Emotion word"])
    eec = df.to_numpy()
    return eec

def Average(list): 
    if len(list) != 0:
        return sum(list) / len(list)
    else:
        return 0

# evaluates the average sentiment score for all male or all female sentences    
def gender():
    eec = import_eec()
    my_model = model.get_model()[0]
    tokenizer = model.get_model()[1]
    labels = model.get_model()[2]

    male_positive = []
    male_neutral = []
    male_negative = []
    female_positive = []
    female_neutral = []
    female_negative = []

    for i in range(0, eec[:,0].size):
        sentence = eec[i,0]
        gender = eec[i,3]

        encoded_sentence = tokenizer(sentence, return_tensors='pt')

        # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
        output = my_model(**encoded_sentence)

        scores = output[0][0].detach().numpy()
        scores = softmax(scores)

        for j in range(len(scores)):

            l = labels[j]
            s = scores[j]

            if (l == "Positive") and (gender == "male"):
                male_positive.append(s)
            elif (l == "Positive") and (gender == "female"):
                female_positive.append(s)
            elif (l == "Neutral") and (gender == "male"):
                male_neutral.append(s)
            elif (l == "Neutral") and (gender == "female"):
                female_neutral.append(s)
            elif (l == "Negative") and (gender == "male"):
                male_negative.append(s)
            elif (l == "Negative") and (gender == "female"):
                female_negative.append(s)

    print("Average male positive: " + str(Average(male_positive)))
    print("Average female positive: " + str(Average(female_positive)))
    print("Average male neutral: " + str(Average(male_neutral)))
    print("Average female neutral: " + str(Average(female_neutral)))
    print("Average male negative: " + str(Average(male_negative)))
    print("Average female negative: " + str(Average(female_negative)))

# evaluates the average sentiment score for male or female sentences for a given emotion e.g. anger, sadness etc.  
def gender_emotion():
    eec = import_eec()
    my_model = model.get_model()[0]
    tokenizer = model.get_model()[1]
    labels = model.get_model()[2]

    for emotion_to_check in set(eec[:,5]):

        male_positive = []
        male_neutral = []
        male_negative = []
        female_positive = []
        female_neutral = []
        female_negative = []

        for i in range(0, eec[:,0].size):
            sentence = eec[i,0]
            gender = eec[i,3]
            emotion = eec[i,5]

            encoded_sentence = tokenizer(sentence, return_tensors='pt')

            # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
            output = my_model(**encoded_sentence)

            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            for j in range(len(scores)):

                l = labels[j]
                s = scores[j]
                
                if emotion_to_check == emotion:
                    if (l == "Positive") and (gender == "male"):
                        male_positive.append(s)
                    elif (l == "Positive") and (gender == "female"):
                        female_positive.append(s)
                    elif (l == "Neutral") and (gender == "male"):
                        male_neutral.append(s)
                    elif (l == "Neutral") and (gender == "female"):
                        female_neutral.append(s)
                    elif (l == "Negative") and (gender == "male"):
                        male_negative.append(s)
                    elif (l == "Negative") and (gender == "female"):
                        female_negative.append(s)

        print("------------------------------------------------------------------------")
        print("Average male positive for " + str(emotion_to_check) + ": " + str(Average(male_positive)))
        print("Average female positive for " + str(emotion_to_check) + ": " + str(Average(female_positive)))
        print("Average male neutral for " + str(emotion_to_check) + ": " + str(Average(male_neutral)))
        print("Average female neutral for " + str(emotion_to_check) + ": " + str(Average(female_neutral)))
        print("Average male negative for " + str(emotion_to_check) + ": " + str(Average(male_negative)))
        print("Average female negative for " + str(emotion_to_check) + ": " + str(Average(female_negative)))

# evaluates the average sentiment score for male or female sentences for a given emotion e.g. anger, sadness etc.  
def gender_template():
    eec = import_eec()
    my_model = model.get_model()[0]
    tokenizer = model.get_model()[1]
    labels = model.get_model()[2]

    for template_to_check in set(eec[:,1]):

        male = []
        female = []

        for i in range(0, eec[:,0].size):
            sentence = eec[i,0]
            template = eec[i,1]
            gender = eec[i,3]
            emotion = eec[i,5]

            encoded_sentence = tokenizer(sentence, return_tensors='pt')

            # output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])
            output = my_model(**encoded_sentence)

            scores = output[0][0].detach().numpy()
            scores = softmax(scores)

            array_item = []

            array_item.append(sentence)

            for j in range(len(scores)):

                l = labels[j]
                s = scores[j]

                array_item.append(s)

            if gender == "male":
                male.append(array_item)
            elif gender == "female":
                female.append(array_item)            

        print("------------------------------------------------------------------------")
        print(pd.DataFrame(male))
        print("------------------------------------------------------------------------")
        print(pd.DataFrame(female))