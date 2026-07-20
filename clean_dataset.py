import json
import re

# Die Blacklist aus dem Konzept und Erweiterungen
BLACKLIST = [
    "als ki-modell",
    "als künstliche intelligenz",
    "als großes sprachmodell",
    "es ist wichtig zu beachten",
    "wichtig ist zu erwähnen",
    "zusammenfassend lässt sich festhalten, dass",
    "zusammenfassend lässt sich sagen, dass",
    "komplexes feld",
    "vielschichtiges thema",
    "abschließend lässt sich sagen",
    "ein weiterer wichtiger punkt ist",
    "es gibt keine eindeutige antwort",
    "wie bereits erwähnt"
]

def clean_and_verify_dataset(file_path, min_words=80, max_words=600):
    cleaned_lines = []
    removed_blacklist = 0
    removed_too_short = 0
    removed_too_long = 0
    
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    for line in lines:
        if not line.strip():
            continue
            
        try:
            data = json.loads(line)
            assistant_content = ""
            for msg in data.get("messages", []):
                if msg.get("role") == "assistant":
                    assistant_content = msg.get("content", "")
                    
            if not assistant_content:
                continue
                
            # 1. Qualitätskontrolle: Wortanzahl prüfen
            word_count = len(assistant_content.split())
            if word_count < min_words:
                removed_too_short += 1
                continue
            if word_count > max_words:
                removed_too_long += 1
                continue
                
            # 2. Qualitätskontrolle: Blacklist-Phrasen suchen
            has_blacklist_phrase = False
            for phrase in BLACKLIST:
                if phrase in assistant_content.lower():
                    has_blacklist_phrase = True
                    break
                    
            if has_blacklist_phrase:
                removed_blacklist += 1
                continue
                
            # Wenn alles in Ordnung ist, behalten wir die Zeile
            cleaned_lines.append(line)
            
        except Exception as e:
            print(f"Fehler beim Parsen einer Zeile: {e}")
            
    # Bereinigte Daten zurückschreiben
    with open(file_path, "w", encoding="utf-8") as f:
        f.writelines(cleaned_lines)
        
    print("--- QUALITÄTSKONTROLLE (Schritt 5) BERICHT ---")
    print(f"Ausgangsanzahl der Datensätze: {len(lines)}")
    print(f"Bereinigte Datensätze übrig: {len(cleaned_lines)}")
    print(f"Aussortiert (zu kurz, < {min_words} Wörter): {removed_too_short}")
    print(f"Aussortiert (zu lang, > {max_words} Wörter): {removed_too_long}")
    print(f"Aussortiert (enthält KI-Blacklist-Floskeln): {removed_blacklist}")
    print("---------------------------------------------")

if __name__ == "__main__":
    clean_and_verify_dataset("train_data.jsonl")
