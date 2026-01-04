from datasets import load_dataset
from colorama import Fore

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, prepare_model_for_kbit_training
import torch

dataset = load_dataset("data", split='train')
print(Fore.YELLOW + str(dataset[2]) + Fore.RESET) 

def format_chat_template(batch, tokenizer):

    system_prompt =  """You are an internal HHS Leave & Benefits policy assistant.

Rules:
- Answer using ONLY the provided policy-based training patterns.
- Be concise and factual.
- If the policy excerpt does not contain the answer, respond EXACTLY with:
  Not specified in the provided excerpt.
- Do not add disclaimers, do not mention external laws, and do not invent details.
"""

    samples = []

    # Access the inputs from the batch
    questions = batch["question"]
    answers = batch["answer"]

    for i in range(len(questions)):
        row_json = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": questions[i]},
            {"role": "assistant", "content": answers[i]}
        ]

        # Apply chat template and append the result to the list
        tokenizer.chat_template = "{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
        text = tokenizer.apply_chat_template(row_json, tokenize=False)
        samples.append(text)

    # Return a dictionary with lists as expected for batched processing
    return {
        "instruction": questions,
        "response": answers,
        "text": samples  # The processed chat template text for each row
    }

# add parameter token
base_model = "meta-llama/Llama-3.2-1B"
tokenizer = AutoTokenizer.from_pretrained(
        base_model, 
        trust_remote_code=True,
)

train_dataset = dataset.map(lambda x: format_chat_template(x, tokenizer), num_proc=8, batched=True, batch_size=10)
print(Fore.LIGHTMAGENTA_EX + str(train_dataset[0]) + Fore.RESET) 


quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

# add parameter token
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="cuda:0",
    quantization_config=quant_config,
    cache_dir="./workspace",
)

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

peft_config = LoraConfig(
    r=64,
    lora_alpha=128,
    lora_dropout=0.05,
    target_modules="all-linear",
    task_type="CAUSAL_LM",
)

trainer = SFTTrainer(
    model,
    train_dataset=train_dataset,
    args=SFTConfig(output_dir="meta-llama/Llama-3.2-1B-SFT", num_train_epochs=50),
    peft_config=peft_config,
)

trainer.train()

trainer.save_model('checkpoints')
trainer.model.save_pretrained("final_model")