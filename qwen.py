import torch
import transformers

model_name = "Qwen/Qwen2.5-0.5B"

model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

input_text = "Once upon a time"
input_ids = tokenizer.encode(input_text, return_tensors="pt")

output = model.generate(input_ids, max_new_tokens=100)
decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_output)
