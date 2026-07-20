# Fine-Tuning Schüler-Schreibstil (Unsloth Studio & QLoRA)

Dieses Repository enthält den vollständigen Datensatz, die Aufbereitungs-Skripte sowie die Trainings- und Inferenz-Pipelines für das Fine-Tuning eines LLMs auf den Schreibstil eines Oberstufenschülers (Persona: *Der Pragmatiker*).

---

## 📌 Projektübersicht

* **Datensatz:** 576 kuratierte, hochgradig argumentative und stilistisch konsistente Konversationen im **ChatML / OpenAI Messages** Format (`train_data.jsonl`).
* **Basismodell:** `unsloth/mistral-7b-instruct-v0.3-bnb-4bit`
* **Training-Technologie:** **Unsloth** (QLoRA 4-bit, Gradient Checkpointing, Bfloat16, Flash-Patching).
* **Hardware-Anforderung:** NVIDIA GPU mit 12 GB VRAM (erfolgreich auf NVIDIA GeForce RTX 5070 Ti Laptop GPU trainiert).

---

## 📁 Repository-Struktur

```text
├── train_data.jsonl          # Vollständiger Trainingsdatensatz (576 Beispiele)
├── train.py                  # QLoRA Fine-Tuning Skript mit automatischer Modell-Konsolidierung
├── export_merged.py          # Skript zum Zusammenführen des LoRA-Adapters in ein einzelnes Modell
├── test_inference.py         # Inferenz-Skript zum Testen des trainierten Schülerstil-Modells
├── validate_unsloth_format.py# Format-Validierer für ChatML-Kompatibilität
├── clean_dataset.py          # Datensatz-Bereinigung und Veredelung
├── generate_dataset.py       # Datensatz-Generierungsskript
├── PROJECT_STATUS.md         # Projekt-Dokumentation & Status
└── Finetuning_Anleitung.md   # Ausführliche Anleitung für Unsloth & Hugging Face TRL
```

---

## 🚀 Schnelleinstieg

### 1. Abhängigkeiten installieren
```bash
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
pip install trl peft accelerate bitsandbytes datasets
```

### 2. Fine-Tuning starten
```bash
python train.py
```

### 3. Modell testen (Inferenz)
```bash
python test_inference.py
```

---

## 📄 Lizenz
MIT License
