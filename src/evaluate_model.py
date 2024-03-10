import numpy as np
import pandas as pd
from scipy.special import softmax
from matplotlib import pyplot as plt
import roberta_sst5
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
def gender(my_model, tokenizer, labels):
    eec = import_eec()

    male_very_positive = []
    male_positive = []
    male_neutral = []
    male_negative = []
    male_very_negative = []

    female_very_positive = []
    female_positive = []
    female_neutral = []
    female_negative = []
    female_very_negative = []

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

            if (l == "very positive") and (gender == "male"):
                male_very_positive.append(s)
            elif (l == "very positive") and (gender == "female"):
                female_very_positive.append(s)
            elif (l == "positive") and (gender == "male"):
                male_positive.append(s)
            elif (l == "positive") and (gender == "female"):
                female_positive.append(s)
            elif (l == "neutral") and (gender == "male"):
                male_neutral.append(s)
            elif (l == "neutral") and (gender == "female"):
                female_neutral.append(s)
            elif (l == "negative") and (gender == "male"):
                male_negative.append(s)
            elif (l == "negative") and (gender == "female"):
                female_negative.append(s)
            elif (l == "very negative") and (gender == "male"):
                male_very_negative.append(s)
            elif (l == "very negative") and (gender == "female"):
                female_very_negative.append(s)

    # print results in terminal
    print("Average male very positive: " + str(Average(male_very_positive) * 100))
    print("Average female very positive: " + str(Average(female_very_positive) * 100))
    print("Difference: " + str((Average(male_very_positive) - Average(female_very_positive)) * 100))
    print("------------------------------------------------------------------------")
    print("Average male positive: " + str(Average(male_positive) * 100))
    print("Average female positive: " + str(Average(female_positive) * 100))
    print("Difference: " + str((Average(male_positive) - Average(female_positive)) * 100))
    print("------------------------------------------------------------------------")
    print("Average male neutral: " + str(Average(male_neutral) * 100))
    print("Average female neutral: " + str(Average(female_neutral) * 100))
    print("Difference: " + str((Average(male_neutral) - Average(female_neutral)) * 100))
    print("------------------------------------------------------------------------")
    print("Average male negative: " + str(Average(male_negative) * 100))
    print("Average female negative: " + str(Average(female_negative) * 100))
    print("Difference: " + str((Average(male_negative) - Average(female_negative)) * 100))
    print("------------------------------------------------------------------------")
    print("Average male very negative: " + str(Average(male_very_negative) * 100))
    print("Average female very negative: " + str(Average(female_very_negative) * 100))
    print("Difference: " + str((Average(male_very_negative) - Average(female_very_negative)) * 100))

    # plot results
    x = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
    y = [Average(male_very_negative), Average(male_negative), Average(male_neutral), Average(male_positive), Average(male_very_positive)]
    y_2 = [Average(female_very_negative), Average(female_negative), Average(female_neutral), Average(female_positive), Average(female_very_positive)]

    plt.plot(x, y, "bo", label="male")
    plt.plot(x, y_2, "ro", label="female")
    plt.legend(loc="upper left")
    plt.title("Average Sentiment Score per Gender")
    plt.xlabel("Sentiment rating")
    plt.ylabel("Proportion of sentences per rating")
    plt.savefig('graphs/average_by_gender.png')

    file = open("eval.txt", "a")
    file.write("Average sentiment difference\n")
    file.write("Very positive (%): " + str((Average(male_very_positive) - Average(female_very_positive)) * 100) + "\n")
    file.write("Positive (%): " + str((Average(male_positive) - Average(female_positive)) * 100) + "\n")
    file.write("Neutral (%): " + str((Average(male_neutral) - Average(female_neutral)) * 100) + "\n")
    file.write("Negative (%): " + str((Average(male_negative) - Average(female_negative)) * 100) + "\n")
    file.write("Very negative (%): " + str((Average(male_very_negative) - Average(female_very_negative)) * 100) + "\n\n")
    file.close()


# evaluates the average sentiment score for male or female sentences for a given emotion e.g. anger, sadness etc.  
def gender_emotion(my_model, tokenizer, labels):
    eec = import_eec()

    for emotion_to_check in set(eec[:,5]):

        male_very_positive = []
        male_positive = []
        male_neutral = []
        male_negative = []
        male_very_negative = []

        female_very_positive = []
        female_positive = []
        female_neutral = []
        female_negative = []
        female_very_negative = []

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
                    if (l == "very positive") and (gender == "male"):
                        male_very_positive.append(s)
                    elif (l == "very positive") and (gender == "female"):
                        female_very_positive.append(s)
                    elif (l == "positive") and (gender == "male"):
                        male_positive.append(s)
                    elif (l == "positive") and (gender == "female"):
                        female_positive.append(s)
                    elif (l == "neutral") and (gender == "male"):
                        male_neutral.append(s)
                    elif (l == "neutral") and (gender == "female"):
                        female_neutral.append(s)
                    elif (l == "negative") and (gender == "male"):
                        male_negative.append(s)
                    elif (l == "negative") and (gender == "female"):
                        female_negative.append(s)
                    elif (l == "very negative") and (gender == "male"):
                        male_very_negative.append(s)
                    elif (l == "very negative") and (gender == "female"):
                        female_very_negative.append(s)

        print("------------------------------------------------------------------------")
        print("Average male very positive for " + str(emotion_to_check) + ": " + str(Average(male_very_positive) * 100))
        print("Average female very positive for " + str(emotion_to_check) + ": " + str(Average(female_very_positive) * 100))
        print("Difference: " + str((Average(male_very_positive) - Average(female_very_positive)) * 100))
        print("Average male positive for " + str(emotion_to_check) + ": " + str(Average(male_positive) * 100))
        print("Average female positive for " + str(emotion_to_check) + ": " + str(Average(female_positive) * 100))
        print("Difference: " + str((Average(male_positive) - Average(female_positive)) * 100))
        print("Average male neutral for " + str(emotion_to_check) + ": " + str(Average(male_neutral) * 100))
        print("Average female neutral for " + str(emotion_to_check) + ": " + str(Average(female_neutral) * 100))
        print("Difference: " + str((Average(male_neutral) - Average(female_neutral)) * 100))
        print("Average male negative for " + str(emotion_to_check) + ": " + str(Average(male_negative) * 100))
        print("Average female negative for " + str(emotion_to_check) + ": " + str(Average(female_negative) * 100))
        print("Difference: " + str((Average(male_negative) - Average(female_negative)) * 100))
        print("Average male very negative for " + str(emotion_to_check) + ": " + str(Average(male_very_negative) * 100))
        print("Average female very negative for " + str(emotion_to_check) + ": " + str(Average(female_very_negative) * 100))
        print("Difference: " + str((Average(male_very_negative) - Average(female_very_negative)) * 100))

        # plot results
        x = ['very negative', 'negative', 'neutral', 'positive', 'very positive']
        y = [Average(male_very_negative), Average(male_negative), Average(male_neutral), Average(male_positive), Average(male_very_positive)]
        y_2 = [Average(female_very_negative), Average(female_negative), Average(female_neutral), Average(female_positive), Average(female_very_positive)]

        plt.plot(x, y, "bo", label="male")
        plt.plot(x, y_2, "ro", label="female")
        plt.legend(loc="upper left")
        plt.title("Average sentiment score per gender for " + str(emotion_to_check))
        plt.xlabel("Sentiment rating")
        plt.ylabel("Proportion of sentences per rating")
        plt.savefig('graphs/{}'.format(emotion_to_check))
        plt.clf()

        file = open("eval.txt", "a")
        file.write("Average sentiment difference per emotion\n")
        file.write(str(emotion_to_check) + ":\n")
        file.write("Very positive (%): " + str((Average(male_very_positive) - Average(female_very_positive)) * 100) + "\n")
        file.write("Positive (%): " + str((Average(male_positive) - Average(female_positive)) * 100) + "\n")
        file.write("Neutral (%): " + str((Average(male_neutral) - Average(female_neutral)) * 100) + "\n")
        file.write("Negative (%): " + str((Average(male_negative) - Average(female_negative)) * 100) + "\n")
        file.write("Very negative (%): " + str((Average(male_very_negative) - Average(female_very_negative)) * 100) + "\n\n")
        file.close()

# evaluates the average sentiment score for male or female sentences for a given emotion e.g. anger, sadness etc.  
def gender_evaluate(my_model, tokenizer, labels):
    eec = import_eec()

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

    df_male.columns = ['Sentence', 'very negative (%)', 'negative (%)', 'neutral (%)', 'positive (%)', 'very positive (%)']
    df_female.columns = ['Sentence', 'very negative (%)', 'negative (%)', 'neutral (%)', 'positive (%)', 'very positive (%)']

    print("---MALE---------------------------------------------------------------------")
    print(df_male)
    print("---FEMALE---------------------------------------------------------------------")
    print(df_female)

    # find difference
    df_diff = df_male.drop("Sentence", axis=1) - df_female.drop("Sentence", axis=1)

    # insert the male sentences in column 0 and the female sentences in column 1 for comparison
    df_diff.insert(0, "Sentence (Male)", df_male.loc[:, "Sentence"], True)
    df_diff.insert(1, "Sentence (Female)", df_female.loc[:, "Sentence"], True)

    # find absolute values of the difference
    df_diff_abs = df_male.drop("Sentence", axis=1) - df_female.drop("Sentence", axis=1)
    df_diff_abs = df_diff_abs.abs()



    # insert new column that is the sum of the bias values for that row
    df_diff_abs.insert(5, "Total Bias (Absolute)", df_diff_abs.sum(axis='columns', numeric_only=True), True)
    print(df_diff_abs)

    # copy total bias column from df_diff_positive to df_diff
    df_diff.loc[:, "Total Bias (Absolute)"] = df_diff_abs.loc[:, "Total Bias (Absolute)"]

    # sort sentences from most to least biased
    df_diff.sort_values("Total Bias (Absolute)", axis=0, ascending=False,inplace=True, na_position='first')

    df_diff.to_csv("src\datasets\diff_roberta.csv", sep=',', index=False, encoding='utf-8')

def get_gender_evaluate():
        
        df_diff = pd.read_csv('src\datasets\df_diff.csv')
        
        print(df_diff)

def main(model):
    my_model, tokenizer, labels = model

    file = open("eval.txt", "w")
    file.write("EEC Evaluation Results \n\n")
    file.close()

    # gender(my_model, tokenizer, labels)
    # gender_emotion(my_model, tokenizer, labels)
    gender_evaluate(my_model, tokenizer, labels)
