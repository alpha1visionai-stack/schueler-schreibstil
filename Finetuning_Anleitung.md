# Anleitung zum Fine-Tuning eines Schülerstil-LLMs auf einer 12GB GPU

Diese Anleitung beschreibt Schritt für Schritt, wie du ein vortrainiertes Large Language Model (z. B. **Llama 3.1 8B Instruct** oder **Llama 3.2 3B Instruct**) mit dem generierten Datensatz auf einer NVIDIA-Grafikkarte mit 12 GB VRAM (z. B. RTX 3060, RTX 4070) feintunst.

Um das Modell auf einer 12GB GPU trainieren zu können, verwenden wir **QLoRA (Quantized Low-Rank Adaptation)** in Kombination mit Speicheroptimierungen wie Gradient Checkpointing.

Es werden zwei alternative Wege beschrieben:
1. **Weg A: Unsloth (Empfohlen)** – Extrem schnell und speichereffizient (bis zu 2x schneller, benötigt weniger VRAM).
2. **Weg B: Standard Hugging Face TRL + PEFT** – Der Standard-Stack mit `transformers` und `bitsandbytes`.

---

## Vorbereitung der Umgebung

Erstelle eine virtuelle Python-Umgebung und installiere die benötigten Bibliotheken.

### Für Weg A (Unsloth - Linux oder Windows WSL empfohlen)
Unsloth bietet vorkompilierte und optimierte Triton-Kernel an.
```bash
pip install --no-cache-dir "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-cache-dir trl peft accelerate bitsandbytes
```

### Für Weg B (Standard Hugging Face)
Für Standard-Training unter nativem Windows oder Linux:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install transformers trl peft accelerate bitsandbytes datasets
```

---

## Das Python-Trainingsskript

Hier ist das vollständige Skript für das Training. Erstelle eine Datei namens `train.py`.

### Option 1: `train.py` mit Unsloth (Sehr speichereffizient)

```python
import torch
from unsloth import FastLanguageModel
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments

# 1. Konfiguration
max_seq_length = 2048 # Passt für typische Schülertexte gut
dtype = None # None für Autodetektion (Float16 für Tesla T4/V100, Bfloat16 für Ampere/Ada Lovelace)
load_in_4bit = True # 4-bit Quantisierung erzwingen für 12GB GPU

# 2. Modell laden
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3.1-8b-Instruct-bnb-4bit", # Vorquantisiertes Llama 3.1 8B
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

# 3. LoRA-Adapter hinzufügen
model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Rank (größer = mehr Kapazität, aber mehr VRAM)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_alpha = 16,
    lora_dropout = 0, # Unsloth optimiert für 0
    bias = "none",
    use_gradient_checkpointing = "unsloth", # Spart enorm VRAM
    random_state = 3407,
    use_rslora = False,
    loftq_config = None,
)

# 4. Datensatz laden und formatieren
# Da unser Datensatz im standardisierten Chat-Format (messages) vorliegt,
# kann Hugging Face / TRL dies direkt verarbeiten.
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

# 5. Trainer konfigurieren
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = dataset,
    dataset_text_field = "messages", # Für ChatML / Standard-Konversationsformat
    max_seq_length = max_seq_length,
    dataset_num_proc = 2,
    packing = False, # Kann das Lernen bei kurzen Texten beschleunigen
    args = TrainingArguments(
        per_device_train_batch_size = 2, # Kleine Batch-Size wegen VRAM
        gradient_accumulation_steps = 4, # Simuliert Batch-Size von 8 (2 * 4)
        warmup_steps = 5,
        max_steps = 60, # Anzahl der Trainingsschritte (an Datensatzgröße anpassen)
        learning_rate = 2e-4, # Typische LoRA-Learning-Rate
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit", # 8-bit Adam spart VRAM
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
    ),
)

# 6. Training starten
trainer_stats = trainer.train()

# 7. Modell speichern (LoRA Adapter)
model.save_pretrained("lora_model_schueler")
tokenizer.save_pretrained("lora_model_schueler")

# Optional: Als 16bit oder 4bit Modell für schnellere Inferenz mergen
# model.save_pretrained_merged("merged_model", tokenizer, save_method = "merged_16bit")
```

---

### Option 2: `train.py` mit Standard Hugging Face (Falls Unsloth nicht genutzt werden kann)

```python
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Quantisierungs-Konfiguration für 12GB VRAM
bnb_config = BitsAndBytesConfig(
    load_in_4_bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    bnb_4bit_use_double_quant=True
)

# 2. Modell laden
model_id = "meta-llama/Llama-3.2-3B-Instruct" # Alternative: "meta-llama/Meta-Llama-3.1-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# Modell für k-bit Training vorbereiten
model = prepare_model_for_kbit_training(model)

# 3. LoRA Konfiguration
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, peft_config)

# Gradient Checkpointing aktivieren (essentiell für 12GB GPU!)
model.gradient_checkpointing_enable()

# 4. Datensatz laden
dataset = load_dataset("json", data_files="train_data.jsonl", split="train")

# 5. Trainer initialisieren
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=1, # Batch-Size 1 zur Sicherheit auf 12GB
    gradient_accumulation_steps=8, # Simuliert effektive Batch-Size von 8
    learning_rate=2e-4,
    logging_steps=10,
    max_steps=100,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    optim="paged_adamw_8bit", # Paged Adamw lagert Zustände bei Bedarf aus
    remove_unused_columns=False,
    gradient_checkpointing=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="messages",
    max_seq_length=1024,
    tokenizer=tokenizer,
    args=training_args,
)

# 6. Training ausführen
trainer.train()

# 7. Speichern
model.save_pretrained("./schueler_style_model")
tokenizer.save_pretrained("./schueler_style_model")
```

---

## Wichtige Tuning-Tipps für den Schreibstil

Da das Ziel des Fine-Tunings die **Stil-Adaption** und nicht die Aneignung von neuem Sachwissen ist, solltest du folgende Hyperparameter beachten:

1. **Epochen / Schritte gering halten**: Trainiere das Modell nicht zu lang. Meist reichen **1 bis 2 Epochen** aus (bzw. ca. 50-100 Schritte bei einer effektiven Batch-Size von 8-16). Zu langes Training führt zu Overfitting und das Modell verliert seine inhaltliche Flexibilität.
2. **Niedrige Learning Rate**: Nutze `2e-4` oder sogar `1e-4` für LoRA. Zu hohe Werte zerstören das vortrainierte Sprachverständnis des Modells.
3. **Modell-Wahl (Instruct vs. Base)**: Da Schüler auf Fragen antworten, solltest du unbedingt die **Instruct**-Variante (z. B. `Llama-3.1-8B-Instruct`) als Basis verwenden. Ein reines Base-Modell versteht die Aufgabenstellungen (Prompts) nicht als Chat-Interaktion, sondern versucht diese nur fortzusetzen.

---

## Testen des trainierten Modells (Inferenz)

Nach dem Training kannst du den gelernten LoRA-Adapter zusammen mit dem Basismodell wie folgt laden, um Texte im Schülerstil zu generieren:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "meta-llama/Llama-3.2-3B-Instruct" # Muss mit dem Trainingsmodell übereinstimmen
peft_model_id = "./schueler_style_model"           # Pfad zu deinen abgespeicherten Gewichten

# 1. Tokenizer und Basismodell laden (in 4-bit für schnellen Test)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)

# 2. LoRA Adapter laden und mergen
model = PeftModel.from_pretrained(model, peft_model_id)

# 3. Prompt definieren im passenden Chat-Format
messages = [
    {"role": "system", "content": "Du antwortest im Stil eines Oberstufenschülers (Persona: Der Pragmatiker)."},
    {"role": "user", "content": "Erkläre kurz den Unterschied zwischen Mitose und Meiose."}
]

inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True).to("cuda")

# 4. Text generieren
with torch.no_grad():
    outputs = model.generate(
        inputs,
        max_new_tokens=300,
        temperature=0.7,
        do_sample=True
    )

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```
