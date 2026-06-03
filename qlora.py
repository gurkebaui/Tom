import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

# Load the Qwen model and tokenizer
model_name = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=False,
)

# Load the quantized model
model = AutoModelForCausalLM.from_pretrained(
    model_name, quantization_config=quantization_config, device_map="auto"
)

# QLoRA configuration
lora_config = LoraConfig(
    r=8,  # Rank of the low-rank matrices
    lora_alpha=32,  # Scaling factor for the low-rank matrices
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # Modules to apply LoRA
    lora_dropout=0.05,  # Dropout probability
    bias="none",  # Bias type
    task_type="CAUSAL_LM",  # Task type
)

# Apply QLoRA to the model
model = get_peft_model(model, lora_config)

# Load the dataset in streaming mode
dataset = load_dataset("bluuwhale/nsfwstory", streaming=True)
train_dataset = dataset["train"]


# Preprocess the dataset
def preprocess_function(examples):
    inputs = tokenizer(
        examples["text"], truncation=True, padding="max_length", max_length=512
    )
    return inputs


train_dataset = train_dataset.map(preprocess_function, batched=True)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    num_train_epochs=3,
    save_steps=100,
    logging_steps=10,
    optim="paged_adamw_8bit",
    max_steps=1000,
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
)

# Train the model
trainer.train()

# Save the model
model.save_pretrained("./qwen_qlora_model")
tokenizer.save_pretrained("./qwen_qlora_model")
