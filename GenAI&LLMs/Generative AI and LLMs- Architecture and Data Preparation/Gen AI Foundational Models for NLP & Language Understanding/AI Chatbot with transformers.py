from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
#AutoModelForSeq2SeqLM lets you interact with your chosen language model.
#AutoTokenizer streamlines the input and presents it to the language model in the most efficient manner. 
# It achieves this by converting the text input into "tokens", which is the model's preferred way of interpreting text.



# Selecting the model.
model_name = "facebook/blenderbot-400M-distill"

# Load the model and tokenizer
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#Downloads the 400 M-parameter distilled Blenderbot weights and instantiates a PyTorch model object ready for inference (evaluation mode).
tokenizer = AutoTokenizer.from_pretrained(model_name)
#Downloads the matching vocabulary/merges/special-token files and instantiates the tokenizer object that knows how to convert strings ↔ token ids for this specific model.

# Define the chat function
def chat_with_bot():
    while True:
        # Get user input
        input_text = input("You: ")

        # Exit conditions
        if input_text.lower() in ["quit", "exit", "bye"]:
            print("Chatbot: Goodbye!")
            break

        # Tokenize input and generate response
        inputs = tokenizer.encode(input_text, return_tensors="pt")
        #Converts the raw string into a list of token-ids, then wraps it in a PyTorch tensor of shape [1, seq_len] (batch size 1). 
        # return_tensors="pt" tells the tokenizer to return PyTorch tensors instead of plain Python lists.
        outputs = model.generate(inputs, max_new_tokens=150) 
        #Feeds the token-id tensor to the model and invokes the default generation strategy (greedy/beam-search with the checkpoint’s own generation config). 
        # The model autoregressively produces new token ids until the end-of-sequence token is emitted or 150 additional tokens have been generated.
        response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        #outputs[0] extracts the first (and only) sequence from the returned batch.
        #decode turns the token ids back into a human-readable string.
        #skip_special_tokens=True removes <s>, </s>, <pad>, etc.
        #.strip() deletes leading/trailing spaces or newlines.
        
        # Display bot's response
        print("Chatbot:", response)

