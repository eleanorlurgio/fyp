# Gender Bias in Natural Language Processing

GitHub link: https://github.com/eleanorlurgio/fyp

This project consists of code to fine-tune 4 models for sentiment analysis and then assess them for gender bias.

## Instructions

1. Clone or download repository

2. Create python virtual environment using:

```
python -m venv venv
.\venv\Scripts\Activate.ps1
```
3. Make sure you are using the correct venv python interpreter. I used a Python 3.11.0 interpreter.

4. Install dependencies by running:

pip install -r requirements.txt

Alternative installation process - run the following lines:

```
pip install numpy
pip install pandas
pip install scipy
pip install transformers
pip install torch
pip install datasets
pip install transformers[torch]
pip install scikit-learn
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install evaluate
pip install matplotlib
pip install plotly
pip install torchtext
pip install portalocker
pip install jupyterlab
pip install sentencepiece
pip install torchsummary
```

5. Run the appropriate Python file to train and evaluate each model. For example run:

python .\src\experiment_1\model_1.py 

Summary of each file:

* experiment_1/model_1 = BERT model
* experiment_1/model_2 = RoBERTa model
* experiment_1/model_3 = DistilBERT model
* experiment_1/model_4 = XLNet model
* experiment_2/model_1 = BERT model with augmented dataset (utilising the augment.py file)
* experiment_2/model_2 = RoBERTa model with augmented dataset (utilising the augment.py file)
* experiment_2/model_3 = DistilBERT model with augmented dataset (utilising the augment.py file)
* experiment_2/model_4 = XLNet model with augmented dataset (utilising the augment.py file)


Note - each file's 'main' function looks like this:

```
def main():
    train_model()
    load_model()
```

To only load the model without training from the beginning, comment or remove the train_model() line

6. View results inside of results folder