from transformers import pipeline
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import transformers

def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')

"Text classification with DistilBERT"
"""“DistilBERT” is a whole family of models: base, uncased, cased, German, French, squad-finetuned, sentiment-finetuned, etc.
The string you pass (distilbert-base-uncased-finetuned-sst-2-english) tells HuggingFace which specific member of that family to download and load, so you get the right weights and vocabulary for your task."""

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

text = "Congratulations! You've won a free ticket to the Bahamas. Reply WIN to claim."

# Tokenize the input text 
                         #so it returns a pytorch tensor instead of a python list cause the model requires that
inputs = tokenizer(text, return_tensors="pt")
print(inputs)

#inference
# Perform inference
#turns off PyTorch’s bookkeeping for gradients, making the forward pass faster and using less memory because I don’t need to compute or store any gradients cause this is inference, not training.
with torch.no_grad():
    outputs = model(**inputs)
#or
#model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask'])

#raw, un-normalized scores 
logits = outputs.logits

#post processing the output
# Convert logits to probabilities
probs = torch.softmax(logits, dim=-1)

# Get the predicted class
predicted_class = torch.argmax(probs, dim=-1)

# Map the predicted class to the label
labels = ["NEGATIVE", "POSITIVE"]
predicted_label = labels[predicted_class]

print(f"Predicted label: {predicted_label}")


"Text generation with GPT-2"
#loading model & tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

#input preprocess
prompt = "Once upon a time"

# Tokenize the input text
inputs = tokenizer(prompt, return_tensors="pt")
inputs

#inference
#gen text
output_ids = model.generate(
    inputs.input_ids, 
    attention_mask=inputs.attention_mask,
    pad_token_id=tokenizer.eos_token_id,
    max_length=50, 
    num_return_sequences=1
)

"""or with torch.no_grad():
    outputs = model(**inputs)"""

#post prrocess the output
# Decode the generated text to turn it into human readable form
generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)


"Hugging Face pipeline() function"

"""is Hugging Face’s “one-liner” helper: I tell it what I want to do (the task) and optionally which model to use, and it quietly downloads/loads the tokenizer + model, wraps them in the right pre- and post-processing code, and hands you a ready-to-use callable object."""
#does all the steps auto
"""transformers.pipeline
(   #The task to perform, such as "text-classification", "text-generation", "question-answering"
    task: str,
    #The model to use. This can be a string (model identifier from Hugging Face model hub), a path to a directory containing model files, or a pre-loaded model instance.
    model: Optional = None,
    #The configuration to use. This can be a string, a path to a directory, or a pre-loaded config object.
    config: Optional = None,
    #The tokenizer to use. This can be a string, a path to a directory, or a pre-loaded tokenizer instance.
    tokenizer: Optional = None,
    #The feature extractor to use for tasks that require it (image processing)
    feature_extractor: Optional = None,
    #The framework to use, either "pt" for PyTorch or "tf" for TensorFlow. If not specified, it will be inferred.
    framework: Optional = None,
    #The specific model version to use
    revision: str = 'main',
    #Whether to use the fast version of the tokenizer if available.
    use_fast: bool = True,
    #Additional keyword arguments passed to the model during initialization.
    model_kwargs: Dict[str, Any] = None,
    #Additional keyword arguments passed to the pipeline components.
    **kwargs
)"""

# Load a general text classification model
classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Classify a sample text
result = classifier("Congratulations! You've won a free ticket to the Bahamas. Reply WIN to claim.")
print(result)

classifier = pipeline("text-classification", model="papluca/xlm-roberta-base-language-detection")
result = classifier("Bonjour, comment ça va?")
print(result)

# Initialize the text generation pipeline with GPT-2
generator = pipeline("text-generation", model="gpt2")

# Generate text based on a given prompt
prompt = "Once upon a time"
result = generator(prompt, max_length=50, num_return_sequences=1, truncation=True)

# Print the generated text
print(result[0]['generated_text'])

"Text generation using T5 with pipeline()"
# Generate text based on a given prompt
prompt = "translate English to French: How are you?"
result = generator(prompt, max_length=50, num_return_sequences=1)

# Print the generated text
print(result[0]['generated_text'])
