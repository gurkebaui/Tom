import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# --- 1. Konfiguration ---
MODEL_ID = "Qwen/Qwen2.5-0.5B"
OUTPUT_DIR = "./qwen25-0.5b-qlora-checkpoints"

# QLoRA 4-Bit Quantisierungs-Konfiguration
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",           # Normal Float 4
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,      # Nested Quantization
)

# --- 2. Tokenizer und Modell laden ---
print("Lade Tokenizer und Modell...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Wichtig: Padding-Token setzen, falls nicht vorhanden
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
)

# Modell für k-bit Training vorbereiten (friert Basis-Modell ein und castet LayerNorm)
model = prepare_model_for_kbit_training(model)

# --- 3. LoRA Konfiguration ---
# Die Target-Module für Qwen2 / Llama-Architekturen
peft_config = LoraConfig(
    r=16,                                  # Rank (Größe der LoRA-Matrizen)
    lora_alpha=32,                         # Alpha-Skalierung
    lora_dropout=0.05,                     # Dropout zur Vermeidung von Overfitting
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters() # Zeigt, dass nur ~1% der Parameter trainiert werden

# --- 4. Datensatz laden ---
# Wir nutzen hier einen kleinen Demo-Datensatz.
# Für eigene Daten muss das Dictionary das Feld "text" enthalten.
print("Lade Datensatz...")
dataset = load_dataset("mlabonne/guanaco-llama2-1k", split="train")

# --- 5. Training Arguments ---
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,               # Für einen schnellen Test auf 100 Schritte limitiert
    save_steps=50,
    fp16=False,
    bf16=torch.cuda.is_bf16_supported(), # Nutzt bfloat16, wenn von der GPU unterstützt
    optim="paged_adamw_8bit",    # Speicher-effizienter Optimierer
    report_to="none",            # Deaktiviert Tracking-Tools wie Weights & Biases
)

# --- 6. Trainer Setup ---
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    max_seq_length=512,          # Maximale Token-Länge pro Beispiel
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",   # Das Feld im Dataset, das den Text enthält
)

# --- 7. Training starten ---
print("Starte Training...")
trainer.train()

# --- 8. Modell speichern ---
print("Speichere LoRA-Adapter...")
final_dir = "./qwen25-0.5b-qlora-final"
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"Training abgeschlossen! Adapter gespeichert unter: {final_dir}")
