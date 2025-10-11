# --------------------------------------------------
# DPO Fine-Tuning – stripped to the bone
# --------------------------------------------------


import multiprocessing, os, requests, tarfile, pandas as pd, matplotlib.pyplot as plt, torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed
from trl import DPOConfig, DPOTrainer

set_seed(42)

model = AutoModelForCausalLM.from_pretrained("gpt2")
model_ref = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
model.config.use_cache = False

ds = load_dataset("BarraHome/ultrafeedback_binarized")
for k in ds: ds[k] = ds[k].select(range(50))

def process(r):
    del r["prompt_id"], r["messages"], r["score_chosen"], r["score_rejected"], r["chosen-model"], r["rejected-model"]
    r["chosen"] = r["chosen"][-1]["content"]
    r["rejected"] = r["rejected"][-1]["content"]
    return r

ds = ds.map(process, num_proc=multiprocessing.cpu_count(), load_from_cache_file=False)
train_dataset = ds["train_prefs"]
eval_dataset = ds["test_prefs"]

peft_config = LoraConfig(r=4, target_modules=["c_proj", "c_attn"], task_type="CAUSAL_LM", lora_alpha=8, lora_dropout=0.1, bias="none")

training_args = DPOConfig(
    beta=0.1, output_dir="dpo", num_train_epochs=5, per_device_train_batch_size=1, per_device_eval_batch_size=1,
    remove_unused_columns=False, logging_steps=10, gradient_accumulation_steps=1, learning_rate=1e-4,
    evaluation_strategy="epoch", warmup_steps=2, fp16=False, save_steps=500, report_to='none'
)

trainer = DPOTrainer(
    model=model, ref_model=None, args=training_args, train_dataset=train_dataset,
    eval_dataset=eval_dataset, tokenizer=tokenizer, peft_config=peft_config, max_length=512
)

trainer.train()
log = pd.DataFrame(trainer.state.log_history)
plt.plot(log[log.loss.notna()].epoch, log[log.loss.notna()].loss, label="train")
plt.plot(log[log.eval_loss.notna()].epoch, log[log.eval_loss.notna()].eval_loss, label="eval")
plt.legend(); plt.show()

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/YIDeT3qihEpWChdXN_RmTg/DPO-tar.gz"
with open("DPO.tar", "wb") as f: f.write(requests.get(url).content)
with tarfile.open("DPO.tar", "r") as t: t.extractall()

dpo_model = AutoModelForCausalLM.from_pretrained("./DPO")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
gen_config = GenerationConfig(do_sample=True, top_k=1, temperature=0.1, max_new_tokens=25, pad_token_id=tokenizer.eos_token_id)

def respond(prompt, m):
    return tokenizer.decode(m.generate(**tokenizer(prompt, return_tensors="pt"), generation_config=gen_config)[0], skip_special_tokens=True)

PROMPT = "Is a higher octane gasoline better for your car?"
print("DPO :", respond(PROMPT, dpo_model))
print("GPT2:", respond(PROMPT, AutoModelForCausalLM.from_pretrained("gpt2")))