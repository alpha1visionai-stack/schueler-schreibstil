import torch
from unsloth import FastLanguageModel

print("=" * 60)
print("Test-Inferenz mit deinem NEUEN Schülerstil-Modell")
print("=" * 60)

model_path = "schueler_model_merged"
max_seq_length = 2048

print(f"Lade zusammengefügtes Modell aus '{model_path}'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

FastLanguageModel.for_inference(model)

messages = [
    {"role": "system", "content": "Du bist ein Oberstufenschüler (Persona: Der Pragmatiker) und antwortest auf Fragen im Schülerschreibstil."},
    {"role": "user", "content": "Warum ist es sinnvoll, im Deutschunterricht Erörterungen zu schreiben?"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

print("\nGeneriere Antwort im trainierten Schülerstil...")
outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 256,
    use_cache = True,
    temperature = 0.7,
    min_p = 0.1,
)

response = tokenizer.batch_decode(outputs)
print("\n" + "=" * 60)
print("ANTWORT DES TRAINIERTEN SCHÜLERSTIL-MODELLS:")
print("=" * 60)
# Extrahiere die Assistenten-Antwort
print(response[0].split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip() if "<|im_start|>assistant" in response[0] else response[0])
