import torch
from datasets import load_dataset
from peft import (  # Added prepare_model_for_kbit_training
    LoraConfig,
    prepare_model_for_kbit_training,
)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

# --- 1. Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "Modotte/CodeX-2M-Thinking"
OUTPUT_DIR = "/content/drive/MyDrive/qwen2.5-0.5b-codex-qlora"
MAX_SEQ_LENGTH = 1024

# --- 2. QLoRA Configuration (4-bit Quantization) ---
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

# --- 3. Load Tokenizer and Model ---
print("Loading Tokenizer and Model in 4-bit...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# FIX 1: Explicitly force non-quantized layers (embeddings) to float16
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    torch_dtype=torch.bfloat16,  # <--- Forces embeddings and unquantized layers to fp16 instead of Qwen's default bf16
    device_map="auto",
    trust_remote_code=True,
)

# FIX 2: Prepare model for k-bit training
# This casts LayerNorms and the lm_head to float32 for stable training
# and prevents the BFloat16 gradient scaler crash.
model = prepare_model_for_kbit_training(model)

# Disable cache for training (required when using gradient checkpointing)
model.config.use_cache = False

# --- 4. Define LoRA Config ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# --- 5. Load and Format Dataset (Streaming) ---
print("Loading Streaming Dataset...")


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["input"])):
        text = f"### Input:\n{examples['input'][i]}\n\n### Output:\n{examples['output'][i]}{tokenizer.eos_token}"
        output_texts.append(text)
    return {"text": output_texts}


dataset = load_dataset(DATASET_ID, streaming=True)
train_dataset = dataset["train"].map(
    formatting_prompts_func, batched=True, remove_columns=["input", "output"]
)

# --- 6. Training Configuration (SFTConfig) ---
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    max_steps=1000,
    fp16=False,
    bf16=True,
    report_to="none",
    optim="paged_adamw_8bit",
    max_length=MAX_SEQ_LENGTH,
    dataset_text_field="text",
    # Enable Gradient Checkpointing
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# --- 7. Initialize and Run Trainer ---
print("Starting QLoRA Training...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()

# --- 8. Save Model ---
print("Training complete. Saving QLoRA adapter...")
trainer.save_model(f"{OUTPUT_DIR}-final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
print(f"QLoRA adapter saved to {OUTPUT_DIR}-final")
