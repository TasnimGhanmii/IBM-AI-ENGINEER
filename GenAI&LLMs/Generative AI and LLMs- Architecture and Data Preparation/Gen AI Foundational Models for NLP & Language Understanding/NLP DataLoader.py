import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k
from typing import Iterable, List
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import torchtext
import torch


sentences = [
    "If you want to know what a man's like, take a good look at how he treats his inferiors, not his equals.",
    "Fame's a fickle friend, Harry.",
    "It is our choices, Harry, that show what we truly are, far more than our abilities.",
    "Soon we must all face the choice between what is right and what is easy.",
    "Youth can not know how age thinks and feels. But old men are guilty if they forget what it was to be young.",
    "You are awesome!"
]

# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences, tokenizer, vocab):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):

        #Run the  tokenizer on that string.
        #Returns a Python list of strings
                                #Grabs the raw string that sits at position idx in the list you gave the Dataset.
        tokens = self.tokenizer(self.sentences[idx])

        # Convert tokens to tensor indices using vocab
        #looks each token up in the Vocab object that was built earlier.
        #If the token never appeared during vocabulary construction it maps to <unk> (index 0).
        #self.vocab also truns the words into numerical form by giving them numerical coordinates, it returns the index of the word from the list it formed
        #we can access the words &/or repreent em by their coordinates 
        tensor_indices = [self.vocab[token] for token in tokens]
        return torch.tensor(tensor_indices)

# Tokenizer
tokenizer = get_tokenizer("basic_english")

# Build vocabulary
#is gonna parse the sentences and transforms them into tokens
vocab = build_vocab_from_iterator(map(tokenizer, sentences))

# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

print("Custom Dataset Length:", len(custom_dataset))
print("Sample Items:")
for i in range(6):
    sample_item = custom_dataset[i]
    print(f"Item {i + 1}: {sample_item}")

# Create an instance of your custom data set
custom_dataset = CustomDataset(sentences, tokenizer, vocab)

# Define batch size
batch_size = 2

# Create a data loader
#multi-threaded iterator that yields mini-batches.
dataloader = DataLoader(custom_dataset, batch_size=batch_size, shuffle=True)

# Iterate through the data loader
for batch in dataloader:
    print(batch)


# Create a custom collate function
def collate_fn(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, batch_first=True, padding_value=0)
    return padded_batch

# Create a data loader with the custom collate function with batch_first=True,
dataloader = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn)

# Iterate through the data loader
for batch in dataloader: 
    for row in batch:
        for idx in row:
            words = [vocab.get_itos()[idx] for idx in row]
        print(words)


# Create a custom collate function
def collate_fn_bfFALSE(batch):
    # Pad sequences within the batch to have equal lengths
    padded_batch = pad_sequence(batch, padding_value=0)
    return padded_batch




# Create a data loader with the custom collate function with batch_first=True,
dataloader_bfFALSE = DataLoader(custom_dataset, batch_size=batch_size, collate_fn=collate_fn_bfFALSE)

# Iterate through the data loader
for seq in dataloader_bfFALSE:
    for row in seq:
        #print(row)
        words = [vocab.get_itos()[idx] for idx in row]
        print(words)


# Iterate through the data loader with batch_first = TRUE
for batch in dataloader:    
    print(batch)
    print("Length of sequences in the batch:",batch.shape[1])


# Define a custom data set
class CustomDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]
    
custom_dataset=CustomDataset(sentences)


def collate_fn(batch):
    # Tokenize each sample in the batch using the specified tokenizer
    tensor_batch = []
    for sample in batch:
        tokens = tokenizer(sample)
        # Convert tokens to vocabulary indices and create a tensor for each sample
        tensor_batch.append(torch.tensor([vocab[token] for token in tokens]))

    # Pad sequences within the batch to have equal lengths using pad_sequence
    # batch_first=True ensures that the tensors have shape (batch_size, max_sequence_length)
    padded_batch = pad_sequence(tensor_batch, batch_first=True)
    
    # Return the padded batch
    return padded_batch


# Create a data loader for the custom dataset
dataloader = DataLoader(
    dataset=custom_dataset,   # Custom PyTorch Dataset containing your data
    batch_size=batch_size,     # Number of samples in each mini-batch
    shuffle=True,              # Shuffle the data at the beginning of each epoch
    collate_fn=collate_fn      # Custom collate function for processing batches
)


for batch in dataloader:
    print(batch)
    print("shape of sample",len(batch))