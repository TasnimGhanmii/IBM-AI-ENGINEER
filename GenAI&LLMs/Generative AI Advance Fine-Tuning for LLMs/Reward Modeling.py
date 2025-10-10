
import json, torch, matplotlib.pyplot as plt, warnings
from datasets import load_dataset, DatasetDict
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, TrainingArguments
from peft import LoraConfig, TaskType
from trl import RewardTrainer

warnings.filterwarnings('ignore')
def warn(*args, **kwargs): pass
warnings.warn = warn

def save_to_json(data, file_path):
    with open(file_path, 'w') as f: json.dump(data, f, indent=4)
    print(f"Data saved to {file_path}")

def load_from_json(file_path):
    with open(file_path, 'r') as f: return json.load(f)

dataset = load_dataset("Dahoas/synthetic-instruct-gptj-pairwise")
print(dataset)

for i in range(3):
    print('prompt\n', dataset['train'][i]['prompt'], '\n')
    print('chosen\n', dataset['train'][i]['chosen'], '\n')
    print('rejected\n', dataset['train'][i]['rejected'], '\n---')

model_name_or_path = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name_or_path, use_fast=True)
model = GPT2ForSequenceClassification.from_pretrained(model_name_or_path, num_labels=1)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id
max_length = 1024

get_res = lambda ds, res: ["\n\nHuman: " + p + "\n\nAssistant: " + r for p, r in zip(ds["train"]["prompt"], ds["train"][res])]
chosen_samples = get_res(dataset, 'chosen')
rejected_samples = get_res(dataset, 'rejected')

def add_combined_columns(example):
    example['prompt_chosen'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["chosen"]
    example['prompt_rejected'] = "\n\nHuman: " + example["prompt"] + "\n\nAssistant: " + example["rejected"]
    return example

dataset['train'] = dataset['train'].map(add_combined_columns)

get_max_len = lambda samples: max([len(s) for s in samples])
print("chosen max len", get_max_len(chosen_samples))
print("rejected max len", get_max_len(rejected_samples))

find_short = lambda ds, max_len: [i for i, (c, r) in enumerate(zip(ds['prompt_chosen'], ds['prompt_rejected'])) if len(c) < max_len or len(r) < max_len]

max_length = 1024
subset_indices = find_short(dataset['train'], max_length)
dataset['train'] = dataset['train'].select(subset_indices)

def preprocess_function(examples):
    tokenized_chosen = tokenizer(examples['prompt_chosen'], truncation=True, max_length=max_length, padding="max_length")
    tokenized_rejected = tokenizer(examples['prompt_rejected'], truncation=True, max_length=max_length, padding="max_length")
    return {
        "input_ids_chosen": tokenized_chosen["input_ids"],
        "attention_mask_chosen": tokenized_chosen["attention_mask"],
        "input_ids_rejected": tokenized_rejected["input_ids"],
        "attention_mask_rejected": tokenized_rejected["attention_mask"],
    }

train_str = {'chosen': dataset['train']['prompt_chosen'], 'rejected': dataset['train']['prompt_rejected']}
dataset['train'] = dataset['train'].map(preprocess_function, batched=True, remove_columns=['prompt', 'chosen', 'rejected', 'prompt_chosen', 'prompt_rejected'])

split_dataset = dataset['train'].train_test_split(test_size=0.2)
dataset_dict = DatasetDict({'train': split_dataset['train'], 'test': split_dataset['test']})

peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0.1, target_modules=["attn.c_attn", "attn.c_proj"])

training_args = TrainingArguments(
    per_device_train_batch_size=3,
    num_train_epochs=3,
    gradient_accumulation_steps=8,
    learning_rate=1.41e-5,
    output_dir="./model_output3",
    logging_steps=10,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    save_total_limit=2,
)

trainer = RewardTrainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=dataset_dict['train'],
    eval_dataset=dataset_dict['test'],
    peft_config=peft_config,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2ForSequenceClassification.from_pretrained("./extracted_model/model_output3", num_labels=1).to(DEVICE)

log_file = "extracted_model/model_output3/checkpoint-2500/trainer_state.json"
with open(log_file, 'r') as f: logs = json.load(f)
steps, losses = [], []
for log in logs["log_history"]:
    if "loss" in log:
        steps.append(log["step"])
        losses.append(log["loss"])
plt.figure(figsize=(10, 5))
plt.plot(steps, losses, label="Training Loss")
plt.xlabel("Steps"); plt.ylabel("Loss"); plt.title("Training Loss Over Time"); plt.legend(); plt.show()

def predict_and_get_logits(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.logits.squeeze().item()

def compare_texts(text1, text2):
    logit1 = predict_and_get_logits(text1)
    logit2 = predict_and_get_logits(text2)
    selected = text1 if logit1 > logit2 else text2
    print("selected---------")
    print(selected, f"score: {max(logit1, logit2)}")
    return selected

N = 10
correct = 0
for chosen, rejected in zip(train_str['chosen'][:N], train_str['rejected'][:N]):
    selected = compare_texts(chosen, rejected)
    if selected == chosen: correct += 1
print("Accuracy:", correct / N)

K = 50
start = len(train_str['chosen']) // 2
correct = 0
for chosen, rejected in zip(train_str['chosen'][start:start + K], train_str['rejected'][start:start + K]):
    selected = compare_texts(chosen, rejected)
    if selected == chosen: correct += 1
print("Accuracy on different subset:", correct / K)