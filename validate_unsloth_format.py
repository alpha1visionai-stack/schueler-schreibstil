import json

def test_unsloth_format(file_path="train_data.jsonl"):
    print("Pruefe Unsloth-Kompatibilitaet fuer:", file_path)
    
    # 1. Test: Einlesen der JSONL-Datei
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    print(f"[OK] JSONL-Datei erfolgreich gelesen ({len(lines)} Datensaetze).")
    
    # 2. Test: Prüfe Struktur aller Einträge nach Unsloth / Hugging Face Kriterien
    for idx, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            example = json.loads(line)
        except Exception as e:
            raise ValueError(f"Eintrag {idx}: Ungueltiges JSON! {e}")
            
        if "messages" not in example:
            raise ValueError(f"Eintrag {idx}: Hat kein 'messages'-Feld!")
            
        messages = example["messages"]
        if not isinstance(messages, list) or len(messages) != 3:
            raise ValueError(f"Eintrag {idx}: 'messages' muss genau 3 Nachrichten enthalten (system, user, assistant)!")
            
        roles = [m.get("role") for m in messages]
        if roles != ["system", "user", "assistant"]:
            raise ValueError(f"Eintrag {idx}: Rollen-Reihenfolge abweichend: {roles}. Erwartet: ['system', 'user', 'assistant'].")
            
        for m in messages:
            if "content" not in m or not isinstance(m["content"], str) or not m["content"].strip():
                raise ValueError(f"Eintrag {idx}: Nachricht fuer Rolle '{m.get('role')}' hat keinen gueltigen Text-Inhalt!")
                
    print(f"[OK] Alle {len(lines)} Eintraege entsprechen exakt dem von Unsloth geforderten ChatML / OpenAI 'messages' Format!")
    
    # 3. Demo der Konvertierung in Unsloth / Llama-3 Template-Format
    sample = json.loads(lines[0])["messages"]
    print("\nBeispiel-Formatierung (wie Unsloth es mit to_sharegpt / get_chat_template fuer Llama-3 konvertiert):")
    print("----------------------------------------------------------------------")
    print(f"<|start_header_id|>system<|end_header_id|>\n\n{sample[0]['content']}<|eot_id|>")
    print(f"<|start_header_id|>user<|end_header_id|>\n\n{sample[1]['content']}<|eot_id|>")
    print(f"<|start_header_id|>assistant<|end_header_id|>\n\n{sample[2]['content'][:150]}...<|eot_id|>")
    print("----------------------------------------------------------------------")
    print("\nFazit: Der Datensatz ist zu 100% kompatibel mit Unsloth!")

if __name__ == "__main__":
    test_unsloth_format()
