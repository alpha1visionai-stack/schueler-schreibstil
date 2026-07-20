import torch
import sys
import os

def check_gpu():
    print("=" * 60)
    print("Prüfe GPU-Verfügbarkeit...")
    print("=" * 60)
    cuda_ok = torch.cuda.is_available()
    print(f"CUDA verfügbar: {cuda_ok}")
    if cuda_ok:
        print(f"Grafikkarte: {torch.cuda.get_device_name(0)}")
        print(f"VRAM Gesamt: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    else:
        print("[FEHLER] Keine CUDA GPU erkannt! Bitte im Geräte-Manager die NVIDIA RTX 5070 Ti reaktivieren.")
        sys.exit(1)

def run_fine_tuning():
    check_gpu()
    
    print("\nLade Unsloth & Transformers...")
    from unsloth import FastLanguageModel
    from datasets import load_dataset
    from trl import SFTTrainer, SFTConfig

    max_seq_length = 2048
    dtype = None # Auto-detect (bfloat16 / float16)
    load_in_4bit = True

    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit"
    print(f"\n1. Lade Basismodell: {model_name}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    print("\n2. Konfiguriere QLoRA Adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    print("\n3. Lade Trainingsdaten (train_data.jsonl)...")
    dataset_file = "train_data.jsonl"
    if not os.path.exists(dataset_file):
        raise FileNotFoundError(f"Datei '{dataset_file}' nicht gefunden!")
        
    dataset = load_dataset("json", data_files=dataset_file, split="train")
    print(f"Anzahl Beispiele: {len(dataset)}")

    print("\n3b. Formatiere ChatML-Nachrichten für Unsloth Trainer...")
    def formatting_prompts_func(examples):
        convos = examples["messages"]
        texts = [tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=False) for convo in convos]
        return { "text" : texts }

    dataset = dataset.map(formatting_prompts_func, batched = True)

    print("\n4. Initialisiere SFTTrainer...")
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = SFTConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 5,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            save_strategy = "no",
        ),
    )

    print("\n5. Starte Fine-Tuning...")
    trainer_stats = trainer.train()

    print("\n6. Speichere LoRA-Adapter...")
    output_path = "lora_model_schueler"
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("\n7. Erstelle EIN EINZELNES neues Modell (Merge von Basis-Modell + Schreibstil-Adapter)...")
    merged_path = "schueler_model_merged"
    model.save_pretrained_merged(merged_path, tokenizer, save_method = "merged_16bit")
    print(f"[ERFOLG] Einzelnes zusammengefügtes Modell gespeichert in: '{merged_path}'")

    print("\n8. Exportiere als einzelne GGUF-Datei (für LM Studio / Ollama)...")
    gguf_path = "schueler_model_gguf"
    try:
        model.save_pretrained_merged(gguf_path, tokenizer, save_method = "gguf", quantization_method = "q4_k_m")
        print(f"[ERFOLG] Konsolidierte GGUF-Modelldatei gespeichert in: '{gguf_path}'")
    except Exception as e:
        print(f"[HINWEIS] GGUF-Export übersprungen oder fehlgeschlagen: {e}")

    print("\n============================================================")
    print("ALL WORK DONE! Dein neues, eigenständiges Schülerstil-Modell ist einsatzbereit.")
    print("============================================================")

if __name__ == "__main__":
    run_fine_tuning()
