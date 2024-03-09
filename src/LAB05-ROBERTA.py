import torch
import torchtext
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Set a fixed value for the random seed to ensure reproducible results
SEED = 1234
# Determine whether a CUDA-compatible GPU is available, and use it if so; otherwise, use the CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Apply the fixed random seed to PyTorch to ensure consistent initialization and random operations
torch.manual_seed(SEED)
# Ensure that any operations performed by cuDNN (a GPU-acceleration library used by PyTorch) are deterministic,
# which can help in reproducing results but may reduce performance
torch.backends.cudnn.deterministic = True


print("PyTorch Version: ", torch.__version__)
print("torchtext Version: ", torchtext.__version__)
print(f"Using {'GPU' if str(DEVICE) == 'cuda' else 'CPU'}.")



tokenizer = AutoTokenizer.from_pretrained("roberta-base")

len(tokenizer.vocab)

tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')

print(tokens)

# original input string
print(tokenizer(['hello world']))

# input string with tab (\t) character
print(tokenizer(['hello	world']))

# input string with newline (\n) character
print(tokenizer(['''
    hello
    world
''']))


print(tokenizer(['hello, world!']))



# Print only the 'input_ids'
print(tokenizer(['hello world 👋'])['input_ids'])

# Use f-string for formatting (Python 3.6+) to access the token corresponding to id 100
token_with_id_100 = list(tokenizer.get_vocab().keys())[list(tokenizer.get_vocab().values()).index(100)]
print(f"Token with id 100: {token_with_id_100}")

## Or, if you're using an older version of Python, use the .format() method
#print("Token with id 100: {}".format(token_with_id_100))

tokens = tokenizer.tokenize('Hello WORLD how ARE yoU?')

indexes = tokenizer.convert_tokens_to_ids(tokens)

print(indexes)


init_token = tokenizer.cls_token
eos_token = tokenizer.sep_token
pad_token = tokenizer.pad_token
unk_token = tokenizer.unk_token

print(init_token, eos_token, pad_token, unk_token)


init_token_idx = tokenizer.convert_tokens_to_ids(init_token)
eos_token_idx = tokenizer.convert_tokens_to_ids(eos_token)
pad_token_idx = tokenizer.convert_tokens_to_ids(pad_token)
unk_token_idx = tokenizer.convert_tokens_to_ids(unk_token)

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

init_token_idx = tokenizer.cls_token_id
eos_token_idx = tokenizer.sep_token_id
pad_token_idx = tokenizer.pad_token_id
unk_token_idx = tokenizer.unk_token_id

print(init_token_idx, eos_token_idx, pad_token_idx, unk_token_idx)

max_input_length = tokenizer.max_model_input_sizes['roberta-base']

print(max_input_length)

from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer

# Define a class TransformerTokenizer that inherits from torch.nn.Module
class TransformerTokenizer(torch.nn.Module):
    # The constructor takes a tokenizer object as input
    def __init__(self, tokenizer):
        super().__init__()  # Initialize the superclass (torch.nn.Module)
        self.tokenizer = tokenizer  # Store the tokenizer object for later use
    
    # Define the forward method, which will be called to tokenize input text
    def forward(self, input):
        # If the input is a list (presumably of strings), iterate over the list
        if isinstance(input, list):
            tokens = [] 
            for text in input:  # Iterate over each string in the input list
                # Tokenize the current string and append the list of tokens to the tokens list
                tokens.append(self.tokenizer.tokenize(text))
            return tokens  # Return the list of lists of tokens
        # If the input is a single string
        elif isinstance(input, str):
            return self.tokenizer.tokenize(input)
        raise ValueError(f"Type {type(input)} is not supported.")
        
# Create a vocabulary object from the tokenizer's vocabulary, setting minimum frequency to 0
# This includes all tokens in the tokenizer's vocabulary in the vocab object
tokenizer_vocab = vocab(tokenizer.vocab, min_freq=0)

import torchtext.transforms as T

text_transform = T.Sequential(
    TransformerTokenizer(tokenizer),  # Tokenize
    T.VocabTransform(tokenizer_vocab),  # Conver to vocab IDs
    T.Truncate(max_input_length - 2),  # Cut to max length to add BOS and EOS token
    T.AddToken(token=tokenizer_vocab["<s>"], begin=True),  # BOS token
    T.AddToken(token=tokenizer_vocab["</s>"], begin=False),  # EOS token
    T.ToTensor(padding_value=tokenizer_vocab["<pad>"]),  # Convert to tensor and pad
)