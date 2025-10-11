import pandas as pd
import torch, json, tarfile, pickle, os, matplotlib.pyplot as plt, warnings
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

warnings.filterwarnings('ignore')
def warn(*args, **kwargs): pass
warnings.warn = warn

def save_to_json(data, file_path):
    with open(file_path, 'w') as f: json.dump(data, f, indent=4)
def load_from_json(file_path):
    with open(file_path, 'r') as f: return json.load(f)

def pad_sequence_to_length(tensor, length, pad_token_id):
    padding_length = length - tensor.size(0)
    if padding_length > 0:
        padding = torch.full((padding_length,), pad_token_id, dtype=torch.long, device=tensor.device)
        return torch.cat((tensor, padding))
    return tensor

def pad_list_to_batch_size(tensors, batch_size, pad_token_id):
    max_length = max(t.size(0) for t in tensors)
    padded_tensors = [pad_sequence_to_length(t, max_length, pad_token_id) for t in tensors]
    while len(padded_tensors) < batch_size:
        padded_tensors.append(torch.full((max_length,), pad_token_id, dtype=torch.long, device=tensors[0].device))
    return padded_tensors[:batch_size]

config = PPOConfig(model_name="lvwerra/gpt2-imdb", learning_rate=1.41e-5)
sent_kwargs = {"top_k": None, "function_to_apply": "none", "batch_size": 2}

model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token
ref_model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)

dataset_name = "imdb"
ds = load_dataset(dataset_name, split="train")
ds = ds.rename_columns({"text": "review"})
ds = ds.filter(lambda x: len(x["review"]) > 200, batched=False)

input_min_text_length, input_max_text_length = 2, 8
input_size = LengthSampler(input_min_text_length, input_max_text_length)

def tokenize(sample):
    sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size()]
    sample["query"] = tokenizer.decode(sample["input_ids"])
    return sample

ds = ds.map(tokenize, batched=False)
ds.set_format(type="torch")

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

ppo_trainer = PPOTrainer(config, model, ref_model, tokenizer, dataset=ds, data_collator=collator)
device = ppo_trainer.accelerator.device
if ppo_trainer.accelerator.num_processes == 1:
    device = 0 if torch.cuda.is_available() else "cpu"

sentiment_pipe = pipeline("sentiment-analysis", model="lvwerra/distilbert-imdb", device=device)

batch = next(iter(ppo_trainer.dataloader))
batch = {key: batch[key][0:2] for key in batch}
response_tensors = []
query_tensors = batch["input_ids"]
get_text = lambda response: ''.join([tokenizer.decode(r.squeeze()) for r in response])

generation_kwargs = {
    "min_length": -1,
    "top_k": 0.0,
    "top_p": 1.0,
    "do_sample": True,
    "pad_token_id": 50256,
}

output_min_length = 4
output_max_length = 16
output_length_sampler = LengthSampler(output_min_length, output_max_length)

for query in query_tensors:
    gen_len = output_length_sampler()
    generation_kwargs["max_new_tokens"] = gen_len
    response = ppo_trainer.generate(query, **generation_kwargs)
    response_tensors.append(response.squeeze()[-gen_len:])

batch["response"] = [tokenizer.decode(r.squeeze()) for r in response_tensors]
texts = [q + r for q, r in zip(batch["query"], batch["response"])]
pipe_outputs = sentiment_pipe(texts, **sent_kwargs)
positive_scores = [item["score"] for output in pipe_outputs for item in output if item["label"] == "POSITIVE"]
rewards = [torch.tensor(score) for score in positive_scores]

batch_size = 128
pad_token_id = tokenizer.pad_token_id
query_tensors = pad_list_to_batch_size(query_tensors, batch_size, pad_token_id)
response_tensors = pad_list_to_batch_size(response_tensors, batch_size, pad_token_id)
rewards = rewards + [torch.tensor(0) for _ in range(batch_size - len(rewards))]



with tarfile.open("ppo-good-tar.gz", "r:gz") as tar: tar.extractall()
model_1 = AutoModelForCausalLMWithValueHead.from_pretrained("ppov3new1")
tokenizer = AutoTokenizer.from_pretrained("ppov3new1")
with open("ppo-good.pkl", 'rb') as f: all_stats = pickle.load(f)
model_1.to(device)

loss_values = [stat['ppo/loss/total'] for stat in all_stats]
reward_values = [stat['ppo/mean_scores'] for stat in all_stats]
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(loss_values, label='Total Loss', color='b')
plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('PPO Training Loss over Time'); plt.legend(); plt.grid(True)
plt.subplot(2, 1, 2)
plt.plot(reward_values, label='Mean Reward', color='g')
plt.xlabel('Epoch'); plt.ylabel('Reward'); plt.title('PPO Mean Reward over Time'); plt.legend(); plt.grid(True)
plt.tight_layout(); plt.show()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pipeline_device = 0 if device.type == "cuda" else -1
gen_kwargs = {"min_length": -1, "max_new_tokens": 20, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}

def generate_some_text(input_text, my_model):
    input_ids = tokenizer(input_text, return_tensors='pt').input_ids.to(device)
    generated_ids = my_model.generate(input_ids, **gen_kwargs)
    return tokenizer.decode(generated_ids[0], skip_special_tokens=True)

input_text = "Once upon a time in a land far"
generated_text = generate_some_text(input_text, model_1)
print(generated_text)
print(sentiment_pipe(generated_text, **sent_kwargs))

generated_text = generate_some_text(input_text, ref_model)
print(generated_text)
print(sentiment_pipe(generated_text, **sent_kwargs))

def compare_models_on_dataset(model, ref_model, dataset, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler):
    gen_kwargs = {"min_length": -1, "top_k": 0.0, "top_p": 1.0, "do_sample": True, "pad_token_id": tokenizer.eos_token_id}
    bs = 16
    game_data = {}
    dataset.set_format("pandas")
    df_batch = dataset[:].sample(bs)
    game_data["query"] = df_batch["query"].tolist()
    query_tensors = df_batch["input_ids"].tolist()
    response_tensors_ref, response_tensors = [], []
    max_pos_ref = ref_model.config.max_position_embeddings
    max_pos_model = model.config.max_position_embeddings
    for i in range(bs):
        gen_len = output_length_sampler()
        input_ids = torch.tensor(query_tensors[i]).unsqueeze(dim=0).to(device)
        total_ref = input_ids.shape[-1] + gen_len
        if total_ref > max_pos_ref:
            max_inp_ref = max_pos_ref - gen_len
            input_ids_ref = input_ids[:, -max_inp_ref:]
        else:
            input_ids_ref = input_ids
        output = ref_model.generate(input_ids_ref, max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
        response_tensors_ref.append(output)
        total_model = input_ids.shape[-1] + gen_len
        if total_model > max_pos_model:
            max_inp_model = max_pos_model - gen_len
            input_ids_model = input_ids[:, -max_inp_model:]
        else:
            input_ids_model = input_ids
        output = model.generate(input_ids_model, max_new_tokens=gen_len, **gen_kwargs).squeeze()[-gen_len:]
        response_tensors.append(output)
    game_data["response (before)"] = [tokenizer.decode(response_tensors_ref[i]) for i in range(bs)]
    game_data["response (after)"] = [tokenizer.decode(response_tensors[i]) for i in range(bs)]
    texts_before = [q + r for q, r in zip(game_data["query"], game_data["response (before)"])]
    game_data["rewards (before)"] = [output[1]["score"] for output in sentiment_pipe(texts_before, **sent_kwargs)]
    texts_after = [q + r for q, r in zip(game_data["query"], game_data["response (after)"])]
    game_data["rewards (after)"] = [output[1]["score"] for output in sentiment_pipe(texts_after, **sent_kwargs)]
    return pd.DataFrame(game_data)

df_results = compare_models_on_dataset(model_1, ref_model, ds, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler)
print(df_results)


with tarfile.open("ppo-bad-tar.gz", "r:gz") as tar: tar.extractall()
model_0 = AutoModelForCausalLMWithValueHead.from_pretrained("ppov3new_bad1")
with open("ppo-bad.pkl", 'rb') as f: all_stats = pickle.load(f)
model_0.to(device)

df_results = compare_models_on_dataset(model_0, model_1, ds, tokenizer, sentiment_pipe, sent_kwargs, device, output_length_sampler)
print(df_results)