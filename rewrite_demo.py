import torch
from unsloth import FastLanguageModel

print("=" * 60)
print("Style Transfer / Umformulierungstest (KI-Text -> Schülerstil)")
print("=" * 60)

model_path = "schueler_model_merged"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Typischer steifer / generischer KI-Text (z. B. aus ChatGPT):
ai_text = """In der heutigen digitalisierten Ära gewinnt die Implementierung von künstlicher Intelligenz im pädagogischen Kontext zunehmend an Relevanz. Es lässt sich konstatieren, dass die Nutzbarmachung von Sprachmodellen zwar Synergieeffekte im Lernprozess freisetzen kann, jedoch keinesfalls die kritische Reflexionsfähigkeit des Lernenden zu substituieren vermag."""

messages = [
    {"role": "system", "content": "Du bist ein Oberstufenschüler. Formuliere den angegebenen Text so um, dass er klingt, als hätte ihn ein Oberstufenschüler selbst verfasst. Behalte den Kernaussage bei, aber nutze natürlichen Schülerschreibstil."},
    {"role": "user", "content": f"Formuliere diesen Text im Schülerstil um:\n\n{ai_text}"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 256,
    use_cache = True,
    temperature = 0.7,
    min_p = 0.1,
)

response = tokenizer.batch_decode(outputs)[0]
ans = response.split("[/INST]")[-1].replace("</s>", "").strip() if "[/INST]" in response else response

print("\nORIGINAL KI-TEXT:")
print("-" * 60)
print(ai_text)
print("\nUMFORMULIERT IM SCHÜLERSTIL:")
print("-" * 60)
print(ans)
