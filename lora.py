import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# --- 1. Configuration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B"
DATASET_ID = "Modotte/CodeX-2M-Thinking"
# Pointing to your mounted Google Drive
OUTPUT_DIR = "/content/drive/MyDrive/qwen2.5-0.5b-codex-lora"
MAX_SEQ_LENGTH = 1024

# --- 2. Load Tokenizer and Model ---
print("Loading Tokenizer and Model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Using float16 for Colab T4 GPUs
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,  # change to bf16 if using 4060ti
    device_map="auto",
    trust_remote_code=True,
)

# --- 3. Define LoRA Config ---
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules="all-linear",
)

# --- 4. Load and Format Dataset (Streaming) ---
print("Loading Streaming Dataset...")


def formatting_prompts_func(examples):
    output_texts = []
    for i in range(len(examples["input"])):
        # Combines input and output into a single text field
        text = f"### Input:\n{examples['input'][i]}\n\n### Output:\n{examples['output'][i]}{tokenizer.eos_token}"
        output_texts.append(text)
    return {"text": output_texts}


dataset = load_dataset(DATASET_ID, streaming=True)
train_dataset = dataset["train"].map(
    formatting_prompts_func, batched=True, remove_columns=["input", "output"]
)

# --- 5. Training Configuration (SFTConfig) ---
# FIXED: 'max_seq_length' is now 'max_length' in modern trl versions
sft_config = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=200,
    max_steps=1000,  # REQUIRED for streaming datasets (size is unknown)
    fp16=False,  # Essential for Colab T4 GPUs change to False if using 4060ti
    bf16=True,  # T4 GPUs don't support bf16 natively change to True if using 4060ti
    report_to="none",
    optim="adamw_torch",
    max_length=MAX_SEQ_LENGTH,  # CHANGED FROM max_seq_length
    dataset_text_field="text",
)

# --- 6. Initialize and Run Trainer ---
print("Starting Training...")
trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    peft_config=peft_config,
    processing_class=tokenizer,
)

trainer.train()

# --- 7. Save Model ---
print("Training complete. Saving model...")
trainer.save_model(f"{OUTPUT_DIR}-final")
tokenizer.save_pretrained(f"{OUTPUT_DIR}-final")
print(f"Model saved to {OUTPUT_DIR}-final")
