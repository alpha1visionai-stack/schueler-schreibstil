import torch
import json
from unsloth import FastLanguageModel

test_questions = [
    "Warum muss man in der Schule eigentlich immer noch Gedichtanalysen schreiben?",
    "Sollten KI-Tools wie ChatGPT für Hausarbeiten in der Schule erlaubt sein?",
    "Wie bereitest du dich am besten auf eine schwere Klausur vor?",
    "Lohnt sich ein Studium nach dem Abitur überhaupt noch oder sollte man lieber direkt arbeiten?"
]

results = []

def generate_answers(model_path, model_label):
    print(f"\n============================================================")
    print(f"Lade Modell: {model_label} ({model_path})...")
    print(f"============================================================")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )
    FastLanguageModel.for_inference(model)
    
    answers = []
    for q in test_questions:
        messages = [
            {"role": "system", "content": "Du bist ein Oberstufenschüler (Persona: Der Pragmatiker) und antwortest prägnant im Schülerschreibstil."},
            {"role": "user", "content": q}
        ]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize = True,
            add_generation_prompt = True,
            return_tensors = "pt",
        ).to("cuda")
        
        outputs = model.generate(
            input_ids = inputs,
            max_new_tokens = 200,
            use_cache = True,
            temperature = 0.7,
            min_p = 0.1,
        )
        decoded = tokenizer.batch_decode(outputs)[0]
        # Clean response string
        if "[/INST]" in decoded:
            ans = decoded.split("[/INST]")[-1].replace("</s>", "").replace("</s>", "").strip()
        elif "<|im_start|>assistant" in decoded:
            ans = decoded.split("<|im_start|>assistant")[-1].replace("<|im_end|>", "").strip()
        else:
            ans = decoded.strip()
        answers.append(ans)
        
    # Free VRAM
    del model
    del tokenizer
    torch.cuda.empty_cache()
    return answers

# 1. Original Modell
orig_answers = generate_answers(
    r"C:\Users\walte\.cache\huggingface\hub\models--unsloth--mistral-7b-instruct-v0.3-bnb-4bit\snapshots\d5f623888f1415cf89b5c208d09cb620694618ee",
    "Original Modell (Mistral 7B Instruct)"
)

# 2. Fine-Tuned Modell
ft_answers = generate_answers(
    "schueler_model_merged",
    "Fine-Tuned Schülerstil Modell"
)

# Build comparison output
comparison_data = []
for i, q in enumerate(test_questions):
    comparison_data.append({
        "question": q,
        "original": orig_answers[i],
        "finetuned": ft_answers[i]
    })

with open("comparison_results.json", "w", encoding="utf-8") as f:
    json.dump(comparison_data, f, ensure_ascii=False, indent=2)

print("\n[ERFOLG] Vergleichstests erfolgreich abgeschlossen! Ergebnisse in comparison_results.json gespeichert.")
