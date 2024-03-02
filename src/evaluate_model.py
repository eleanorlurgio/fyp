import numpy as np
import pandas as pd
from scipy.special import softmax
from matplotlib import pyplot as plt
import roberta_model
import example_model

def import_eec():
    # import EEC data set
    df = pd.read_csv('src\datasets\Equity-Evaluation-Corpus.csv', usecols=["Sentence", "Template","Person","Gender","Race","Emotion","Emotion word"])
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
    my_model = roberta_model.get_saved_model()[0]
    tokenizer = roberta_model.get_saved_model()[1]
    labels = roberta_model.get_saved_model()[2]

    male_positive = []
    male_slightly_positive = []
    male_neutral = []
    male_slightly_negative = []
    male_negative = []

    female_positive = []
    female_slightly_positive = []
    female_neutral = []
    female_slightly_negative = []
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
            elif (l == "Slightly Positive") and (gender == "male"):
                male_slightly_positive.append(s)
            elif (l == "Slightly Positive") and (gender == "female"):
                female_slightly_positive.append(s)
            elif (l == "Neutral") and (gender == "male"):
                male_neutral.append(s)
            elif (l == "Neutral") and (gender == "female"):
                female_neutral.append(s)
            elif (l == "Slightly Negative") and (gender == "male"):
                male_slightly_negative.append(s)
            elif (l == "Slightly Negative") and (gender == "female"):
                female_slightly_negative.append(s)
            elif (l == "Negative") and (gender == "male"):
                male_negative.append(s)
            elif (l == "Negative") and (gender == "female"):
                female_negative.append(s)

    # print results in terminal
    print("Average male positive: " + str(Average(male_positive)))
    print("Average female positive: " + str(Average(female_positive)))
    print("------------------------------------------------------------------------")
    print("Average male slightly positive: " + str(Average(male_slightly_positive)))
    print("Average female slightly positive: " + str(Average(female_slightly_positive)))
    print("------------------------------------------------------------------------")
    print("Average male neutral: " + str(Average(male_neutral)))
    print("Average female neutral: " + str(Average(female_neutral)))
    print("------------------------------------------------------------------------")
    print("Average male slightly negative: " + str(Average(male_slightly_negative)))
    print("Average female slightly negative: " + str(Average(female_slightly_negative)))
    print("------------------------------------------------------------------------")
    print("Average male negative: " + str(Average(male_negative)))
    print("Average female negative: " + str(Average(female_negative)))

    # plot results
    x = ['Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Positive']
    y = [Average(male_negative), Average(male_slightly_negative), Average(male_neutral), Average(male_slightly_positive), Average(male_positive)]
    y_2 = [Average(female_negative), Average(female_slightly_negative), Average(female_neutral), Average(female_slightly_positive), Average(female_positive)]

    plt.plot(x, y, "bo", label="male")
    plt.plot(x, y_2, "ro", label="female")
    plt.legend(loc="upper left")
    plt.title("Average Sentiment Score per Gender")
    plt.xlabel("Sentiment rating")
    plt.ylabel("Proportion of sentences per rating")
    plt.savefig('graphs/average_by_gender.png')


# evaluates the average sentiment score for male or female sentences for a given emotion e.g. anger, sadness etc.  
def gender_emotion():
    eec = import_eec()
    my_model = roberta_model.get_saved_model()[0]
    tokenizer = roberta_model.get_saved_model()[1]
    labels = roberta_model.get_saved_model()[2]

    for emotion_to_check in set(eec[:,5]):

        male_positive = []
        male_slightly_positive = []
        male_neutral = []
        male_slightly_negative = []
        male_negative = []

        female_positive = []
        female_slightly_positive = []
        female_neutral = []
        female_slightly_negative = []
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
                    elif (l == "Slightly Positive") and (gender == "male"):
                        male_slightly_positive.append(s)
                    elif (l == "Slightly Positive") and (gender == "female"):
                        female_slightly_positive.append(s)
                    elif (l == "Neutral") and (gender == "male"):
                        male_neutral.append(s)
                    elif (l == "Neutral") and (gender == "female"):
                        female_neutral.append(s)
                    elif (l == "Slightly Negative") and (gender == "male"):
                        male_slightly_negative.append(s)
                    elif (l == "Slightly Negative") and (gender == "female"):
                        female_slightly_negative.append(s)
                    elif (l == "Negative") and (gender == "male"):
                        male_negative.append(s)
                    elif (l == "Negative") and (gender == "female"):
                        female_negative.append(s)

        print("------------------------------------------------------------------------")
        print("Average male positive for " + str(emotion_to_check) + ": " + str(Average(male_positive)))
        print("Average female positive for " + str(emotion_to_check) + ": " + str(Average(female_positive)))
        print("Average male slightly positive for " + str(emotion_to_check) + ": " + str(Average(male_slightly_positive)))
        print("Average female slightly positive for " + str(emotion_to_check) + ": " + str(Average(female_slightly_positive)))
        print("Average male neutral for " + str(emotion_to_check) + ": " + str(Average(male_neutral)))
        print("Average female neutral for " + str(emotion_to_check) + ": " + str(Average(female_neutral)))
        print("Average male slightly negative for " + str(emotion_to_check) + ": " + str(Average(male_slightly_negative)))
        print("Average female slightly negative for " + str(emotion_to_check) + ": " + str(Average(female_slightly_negative)))
        print("Average male negative for " + str(emotion_to_check) + ": " + str(Average(male_negative)))
        print("Average female negative for " + str(emotion_to_check) + ": " + str(Average(female_negative)))

        # plot results
        x = ['Negative', 'Slightly Negative', 'Neutral', 'Slightly Positive', 'Positive']
        y = [Average(male_negative), Average(male_slightly_negative), Average(male_neutral), Average(male_slightly_positive), Average(male_positive)]
        y_2 = [Average(female_negative), Average(female_slightly_negative), Average(female_neutral), Average(female_slightly_positive), Average(female_positive)]

        plt.plot(x, y, "bo", label="male")
        plt.plot(x, y_2, "ro", label="female")
        plt.legend(loc="upper left")
        plt.title("Average sentiment score per gender for " + str(emotion_to_check))
        plt.xlabel("Sentiment rating")
        plt.ylabel("Proportion of sentences per rating")
        plt.savefig('graphs/{}'.format(emotion_to_check))
        plt.clf()

# evaluates the average sentiment score for male or female sentences for a given emotion e.g. anger, sadness etc.  
def gender_evaluate():
    eec = import_eec()
    my_model = roberta_model.get_saved_model()[0]
    tokenizer = roberta_model.get_saved_model()[1]
    labels = roberta_model.get_saved_model()[2]

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
            s = scores[j] * 100

            array_item.append(s)

        if gender == "male":
            male.append(array_item)
        elif gender == "female":
            female.append(array_item)       

    df_male = pd.DataFrame(male)
    df_female = pd.DataFrame(female)

    df_male.columns = ['Sentence', 'Negative (%)', 'Neutral (%)', 'Positive (%)']
    df_female.columns = ['Sentence', 'Negative (%)', 'Neutral (%)', 'Positive (%)']

    print("---MALE---------------------------------------------------------------------")
    print(df_male)
    print("---FEMALE---------------------------------------------------------------------")
    print(df_female)

    # find difference

    df_diff = df_male.drop("Sentence", axis=1) - df_female.drop("Sentence", axis=1)

    df_diff.insert(0, "Sentence (Male)", df_male.loc[:, "Sentence"], True)
    df_diff.insert(1, "Sentence (Female)", df_female.loc[:, "Sentence"], True)

    df_diff.to_csv("src\datasets\df_diff.csv", sep=',', index=False, encoding='utf-8')

def get_gender_evaluate():
        
        df_diff = pd.read_csv('src\datasets\df_diff.csv')
        
        print(df_diff)