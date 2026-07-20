import torch
from unsloth import FastLanguageModel

print("=" * 60)
print("Führe Merge von Basismodell + Schülerstil-Adapter durch...")
print("=" * 60)

max_seq_length = 2048
dtype = None
load_in_4bit = True

base_model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
adapter_path = "lora_model_schueler"

print(f"\n1. Lade Basismodell und trainierten Adapter aus '{adapter_path}'...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = adapter_path,
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

print("\n2. Speichere als EINZELNES konsolidiertes 4-bit Modell (Hugging Face Format)...")
merged_dir = "schueler_model_merged"
model.save_pretrained_merged(merged_dir, tokenizer, save_method = "merged_4bit_forced")
print(f"[ERFOLG] Einzelnes Modell gespeichert in: '{merged_dir}'")

print("\n3. Exportiere als konsolidierte GGUF-Datei für LM Studio / Ollama...")
gguf_dir = "schueler_model_gguf"
try:
    model.save_pretrained_merged(gguf_dir, tokenizer, save_method = "gguf", quantization_method = "q4_k_m")
    print(f"[ERFOLG] GGUF-Datei gespeichert in: '{gguf_dir}'")
except Exception as e:
    print(f"[HINWEIS] GGUF Export Hinweis: {e}")

print("\n============================================================")
print("ALLES FERTIG! Dein neues Schülerstil-Modell ist vollständig einsatzbereit.")
print("============================================================")
