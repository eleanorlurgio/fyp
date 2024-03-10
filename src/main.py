import evaluate_model
import example_model
import roberta_sst5
# import gpt2_model

def main():

    # roberta_model.main()
    # gpt2_model()
    # evaluate_model.gender()
    # evaluate_model.gender_emotion()
    # evaluate_model.gender_evaluate()
    evaluate_model.main(roberta_sst5.get_saved_model())

if __name__ == "__main__":
    main()