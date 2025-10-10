

import warnings, torch, evaluate, pickle, json, matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import get_peft_model, LoraConfig, TaskType
from urllib.request import urlopen
import io

warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = load_dataset("json", data_files="CodeAlpaca-20k.json", split="train")
dataset = dataset.filter(lambda e: e["input"] == '').shuffle(seed=42)
dataset_split = dataset.train_test_split(test_size=0.2, seed=42)
train_dataset, test_dataset = dataset_split['train'], dataset_split['test']
tiny_test_dataset = test_dataset.select(range(10))
tiny_train_dataset = train_dataset.select(range(10))

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m").to(device)
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m", padding_side='left')

def formatting_prompts_func(mydataset):
    return [f"### Instruction:\\n{mydataset['instruction'][i]}\\n\\n### Response:\\n{mydataset['output'][i]}</s>" for i in range(len(mydataset['instruction']))]

def formatting_prompts_func_no_response(mydataset):
    return [f"### Instruction:\\n{mydataset['instruction'][i]}\\n\\n### Response:\\n" for i in range(len(mydataset['instruction']))]

expected_outputs, instructions_with_responses, instructions = [], formatting_prompts_func(test_dataset), formatting_prompts_func_no_response(test_dataset)
for i in tqdm(range(len(instructions))):
    tiw = tokenizer(instructions_with_responses[i], return_tensors="pt", max_length=1024, truncation=True, padding=False)
    ti = tokenizer(instructions[i], return_tensors="pt")
    expected_outputs.append(tokenizer.decode(tiw['input_ids'][0][len(ti['input_ids'][0])-1:], skip_special_tokens=True))

class ListDataset(Dataset):
    def __init__(self, lst): self.original_list = lst
    def __len__(self): return len(self.original_list)
    def __getitem__(self, i): return self.original_list[i]

instructions_torch = ListDataset(instructions)

gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, batch_size=2, max_length=50, truncation=True, padding=False, return_full_text=False)

urlopened = urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/VvQRrSqS1P0_GobqtL-SKA/instruction-tuning-generated-outputs-base.pkl')
generated_outputs_base = pickle.load(io.BytesIO(urlopened.read()))

sacrebleu = evaluate.load("sacrebleu")
results_base = sacrebleu.compute(predictions=generated_outputs_base, references=expected_outputs)
print(round(results_base["score"], 1))

lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"], lora_dropout=0.1, task_type=TaskType.CAUSAL_LM)
model = get_peft_model(model, lora_config)

response_template = "### Response:\n"
collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

training_args = SFTConfig(output_dir="/tmp", num_train_epochs=10, save_strategy="epoch", fp16=True, per_device_train_batch_size=2, per_device_eval_batch_size=2, max_seq_length=1024, do_eval=True)
trainer = SFTTrainer(model, train_dataset=train_dataset, eval_dataset=test_dataset, formatting_func=formatting_prompts_func, args=training_args, packing=False, data_collator=collator)

urlopened = urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/49I70jQD0-RNRg2v-eOoxg/instruction-tuning-log-history-lora.json')
log_history_lora = json.load(io.BytesIO(urlopened.read()))
train_loss = [log["loss"] for log in log_history_lora if "loss" in log]
plt.figure(figsize=(10, 5)); plt.plot(train_loss, label='Training Loss'); plt.xlabel('Steps'); plt.ylabel('Loss'); plt.title('Training Loss'); plt.legend(); plt.show()

gen_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, batch_size=2, max_length=50, truncation=True, padding=False, return_full_text=False)

urlopened = urlopen('https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/o7uYxe15xvX4CN-6Lr10iA/instruction-tuning-generated-outputs-lora.pkl')
generated_outputs_lora = pickle.load(io.BytesIO(urlopened.read()))

results_lora = sacrebleu.compute(predictions=generated_outputs_lora, references=expected_outputs)
print(round(results_lora["score"], 1))