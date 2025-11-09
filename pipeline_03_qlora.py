import argparse, os, torch
from pyspark.sql import SparkSession
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

parser = argparse.ArgumentParser()
parser.add_argument("--curated", required=True)
parser.add_argument("--out", required=True)
args = parser.parse_args()
os.makedirs(args.out, exist_ok=True)

# 1) Load Parquet -> pandas -> HF Dataset
spark = SparkSession.builder.appName("MakeSFTData").getOrCreate()
pdf = spark.read.parquet(args.curated)\
        .select("reason_for_visit", "description")\
        .na.drop()\
        .limit(20000).toPandas()
spark.stop()

def fmt(r):
    return {"text": f"<s>[INSTRUCTION] {r['reason_for_visit']}\n[RESPONSE] {r['description']}</s>"}

train_data = [fmt(r) for _, r in pdf.iterrows()]
dataset = Dataset.from_list(train_data)

# 2) Base model in 4-bit
model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
base = prepare_model_for_kbit_training(base)
torch.cuda.empty_cache()

# 3) LoRA config
lconf = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "v_proj"]
)
model = get_peft_model(base, lconf)

# 4) Tokenize
def tok(ex):
    out = tokenizer(ex["text"], truncation=True, padding="max_length", max_length=512)
    out["labels"] = out["input_ids"].copy()
    return out

tokenized = dataset.map(tok, batched=True, remove_columns=["text"])

# 5) Train
train_args = TrainingArguments(
    output_dir=args.out,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
)
trainer = Trainer(model=model, args=train_args, train_dataset=tokenized)
trainer.train()

# 6) Save adapter only
model.save_pretrained(args.out)
tokenizer.save_pretrained(args.out)
print("✅ LoRA adapter saved to:", args.out)

