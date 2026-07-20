import os
import sys
import json
import argparse
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

# Definition der Personas
PERSONAS = {
    "ueberflieger": {
        "name": "Der Überflieger (Note 1)",
        "description": "Sehr eloquent, strukturiert, nutzt Fachbegriffe meist korrekt. Zeigt aber trotzdem noch eine jugendliche/schulische Argumentationsstruktur (keine rein akademische Wissenschaftssprache). Schreibstil ist sauber, fehlerfrei und stilsicher, jedoch nicht gekünstelt."
    },
    "bemueht": {
        "name": "Der Bemühte (Note 2-3)",
        "description": "Nutzt viele gelernte Phrasen und Konnektoren ('Im Hinblick auf', 'Des Weiteren', 'Zusammenfassend lässt sich konstatieren'). Neigt zu Schachtelsätzen, bei denen am Ende die Grammatik leicht verrutscht. Verbeißt sich manchmal in Details und versucht krampfhaft, intellektuell zu klingen."
    },
    "pragmatiker": {
        "name": "Der Pragmatiker (Note 3-4)",
        "description": "Schreibt eher kurze, einfache Sätze. Sprachlich sehr simpel, wiederholt oft dieselben Satzanfänge ('Zuerst...', 'Dann...', 'Danach...'). Nutzt informelle Füllwörter ('voll', 'quasi', 'halt', 'irgendwie'). Inhaltlich passabel, aber oft oberflächlich."
    }
}

# Aufgabenmatrix nach Fächern
TASKS = {
    "Deutsch": [
        "Analysiere kurz das Verhalten von Franz Moor in Akt 1, Szene 1 aus Schillers 'Die Räuber'.",
        "Interpretiere kurz das Motiv der Natur und Sehnsucht in einem typischen Gedicht der Romantik (z. B. von Eichendorff).",
        "Schreibe eine kurze Stellungnahme zur Frage: Sollte im Deutschunterricht vermehrt moderne Jugendliteratur statt klassischer Lektüre gelesen werden?",
        "Schreibe eine kurze Erörterung zum Thema: Hat die gedruckte Tageszeitung im Zeitalter digitaler Medien noch eine Zukunft?"
    ],
    "Geschichte": [
        "Analysiere kurz die Argumentation und Intention in Philipp Scheidemanns Ausrufung der Republik am 9. November 1918.",
        "Bewerte die sozialen Folgen der Industrialisierung für die Arbeiterschaft im Deutschland des 19. Jahrhunderts.",
        "Erkläre kurz, warum das Attentat von Sarajevo am 28. Juni 1914 als 'Funke am Pulverfass' bezeichnet wird.",
        "Erörtere die Ursachen und Folgen der Weltwirtschaftskrise von 1929 für die Weimarer Republik."
    ],
    "Biologie": [
        "Erkläre das Prinzip der Informationsübertragung an einer chemischen Synapse.",
        "Beschreibe die lichtabhängige Reaktion der Fotosynthese und deren Bedeutung für die Pflanze.",
        "Erkläre die Entstehung eines Aktionspotenzials am Axon (Depolarisation und Repolarisation).",
        "Beschreibe den Unterschied zwischen Mitose und Meiose und deren biologische Funktion."
    ],
    "Physik": [
        "Erkläre das Phänomen der Interferenz am Doppelspaltexperiment mit Licht.",
        "Erkläre den Unterschied zwischen einer Transversalwelle und einer Longitudinalwelle anhand von Beispielen.",
        "Beschreibe den Aufbau und die Funktionsweise eines Transformators.",
        "Erkläre das Prinzip des photoelektrischen Effekts und warum er die Lichtquantenhypothese stützt."
    ],
    "PoWi": [
        "Sollte das Wahlrecht bei Bundestagswahlen auf 16 Jahre gesenkt werden? Nenne je zwei Pro- und Contra-Argumente.",
        "Erörtere die Vor- und Nachteile eines allgemeinen Tempolimits von 130 km/h auf deutschen Autobahnen.",
        "Ist eine CO2-Steuer eine gerechte Maßnahme zum Klimaschutz? Nimm kurz Stellung.",
        "Diskutiere die Frage, ob soziale Netzwerke wie TikTok eine Gefahr für die politische Willensbildung von Jugendlichen darstellen."
    ]
}

# Unzulässige Phrasen, die den typischen "KI-Assistenten-Stil" verraten
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
    "natürlich stehe ich dir",
    "gerne helfe ich",
    "es gibt keine eindeutige antwort",
    "wie bereits erwähnt"
]

def clean_text(text):
    """
    Entfernt einfache kosmetische KI-Floskeln am Anfang oder Ende des Textes.
    """
    text = text.strip()
    
    # Entferne typische Einleitungssätze von Chat-Modellen
    lines = text.split("\n")
    if lines and (lines[0].lower().startswith("hier ist") or lines[0].lower().startswith("gerne") or "schüler" in lines[0].lower() and len(lines[0]) < 100):
        lines = lines[1:]
    
    # Wieder zusammenfügen
    text = "\n".join(lines).strip()
    
    # Bereinige Anführungszeichen am Anfang/Ende, falls das Modell den Text eingepackt hat
    if text.startswith('"') and text.endswith('"'):
        text = text[1:-1].strip()
    elif text.startswith('„') and text.endswith('“'):
        text = text[1:-1].strip()
        
    return text

def contains_blacklist_phrases(text):
    """
    Überprüft, ob der Text verbotene KI-Phrasen enthält.
    """
    text_lower = text.lower()
    found_phrases = []
    for phrase in BLACKLIST:
        if phrase in text_lower:
            found_phrases.append(phrase)
    return found_phrases

def generate_sample(client, model, subject, task, persona_key, persona_info, style_sample=None, max_retries=3):
    """
    Generiert ein einzelnes Trainingsbeispiel unter Berücksichtigung von Persona und Filterregeln.
    """
    system_prompt = f"""Du bist ein Datengenerator für Schülertexte der gymnasialen Oberstufe (Klassen 11-13) in Deutschland.
Deine Aufgabe ist es, eine authentische Schülerantwort auf eine gegebene Hausaufgabe oder Klausuraufgabe zu verfassen.

Berücksichtige dabei die zugewiesene Persona:
{persona_info['name']}: {persona_info['description']}

WICHTIGE STILREGELN:
1. Schreibe absolut AUTHENTISCH. Vermeide den typischen, perfekt geschliffenen, distanzierten KI-Assistenten-Stil.
2. Der Text muss sprachlich und strukturell genau zur Persona passen.
3. Verwende KEINE KI-Floskeln wie "Es ist wichtig zu beachten", "Zusammenfassend lässt sich sagen" oder "Als KI-Modell". Schreibe direkt als der Schüler.
4. Die Länge der Antwort sollte circa 150 bis 400 Wörter betragen (eine typische kurze Klausuraufgabe).
5. Antworte direkt mit dem Schülertext. Keine Metakommentare davor oder danach!"""

    if style_sample:
        system_prompt += f"\n\nSTIL-INSPIRATION (Wikipedia-Diskussionsstil):\nLass dich bei der Argumentation und dem Tonfall von diesem echten Wikipedia-Diskussionsbeispiel inspirieren. Verwende ähnliche Satzstrukturen und argumentative Übergänge, um den Text so klingen zu lassen, als würde der Schüler intensiv und bemüht debattieren:\n\"{style_sample}\""

    user_prompt = f"Fach: {subject}\nAufgabe: {task}"

    cleaned_content = None
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.8,
                max_tokens=800
            )
            
            raw_content = response.choices[0].message.content
            cleaned_content = clean_text(raw_content)
            
            blocked = contains_blacklist_phrases(cleaned_content)
            if not blocked:
                return cleaned_content
            else:
                # Phrasen protokollieren, aber das Log schlank halten
                pass
        except Exception as e:
            # Fehler protokollieren
            pass
            
    if cleaned_content is None:
        raise RuntimeError("Generierung fehlgeschlagen aufgrund anhaltender API-Fehler. Bitte überprüfe den API-Key.")
        
    return cleaned_content

def main():
    parser = argparse.ArgumentParser(description="Generiert einen Synthetischen Schülerstil-Datensatz.")
    parser.add_argument("--output", type=str, default="train_data_generated.jsonl", help="Pfad zur Ausgabe-JSONL-Datei")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Das zu verwendende Erzeuger-Modell")
    parser.add_argument("--num-samples", type=int, default=15, help="Anzahl der zu generierenden Beispiele")
    parser.add_argument("--workers", type=int, default=15, help="Anzahl der parallel arbeitenden Threads")
    parser.add_argument("--api-key", type=str, default=None, help="OpenAI API Key")
    parser.add_argument("--base-url", type=str, default=None, help="Custom Base-URL")
    
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key and not args.base_url:
        print("Fehler: Bitte gib einen API-Key via --api-key an oder setze die Umgebungsvariable OPENAI_API_KEY.")
        sys.exit(1)

    client = OpenAI(
        api_key=api_key or "no-key-needed",
        base_url=args.base_url
    )

    print(f"Starte parallele Generierung von {args.num_samples} Beispielen mit {args.workers} Threads...")
    print(f"Erzeugermodell: {args.model}")
    if args.base_url:
        print(f"API Base-URL: {args.base_url}")
    print(f"Speicherort: {args.output}\n")

    # Wikipedia-Stile laden falls vorhanden
    style_samples = {}
    if os.path.exists("wikipedia_style_samples.json"):
        try:
            with open("wikipedia_style_samples.json", "r", encoding="utf-8") as sf:
                style_samples = json.load(sf)
            print("Wikipedia-Stilvorlagen erfolgreich geladen.")
        except Exception as e:
            print(f"Warnung beim Laden der Stilvorlagen: {e}")

    subject_keys = list(TASKS.keys())
    persona_keys = list(PERSONAS.keys())

    # Generierungsaufgaben vorbereiten
    tasks_to_run = []
    for _ in range(args.num_samples):
        sub = random.choice(subject_keys)
        task_str = random.choice(TASKS[sub])
        pers_key = random.choice(persona_keys)
        tasks_to_run.append((sub, task_str, pers_key))

    # Lock für threadsicheres Schreiben in die Ausgabedatei und Zählen
    write_lock = threading.Lock()
    completed_count = 0
    error_count = 0

    with open(args.output, "w", encoding="utf-8") as f:
        def worker(item):
            nonlocal completed_count, error_count
            sub, task_str, pers_key = item
            persona_info = PERSONAS[pers_key]
            
            # Zufälliges Stil-Sample der passenden Persona wählen
            style_sample = None
            if pers_key in style_samples:
                style_sample = random.choice(style_samples[pers_key])
            
            try:
                student_response = generate_sample(client, args.model, sub, task_str, pers_key, persona_info, style_sample=style_sample)
                
                chat_format = {
                    "messages": [
                        {"role": "system", "content": f"Du antwortest im Stil eines Oberstufenschülers (Persona: {persona_info['name']})."},
                        {"role": "user", "content": task_str},
                        {"role": "assistant", "content": student_response}
                    ]
                }
                
                with write_lock:
                    f.write(json.dumps(chat_format, ensure_ascii=False) + "\n")
                    f.flush()
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == args.num_samples:
                        print(f"Fortschritt: {completed_count}/{args.num_samples} Beispiele erfolgreich generiert.")
            except Exception as e:
                with write_lock:
                    error_count += 1
                    print(f"  [Fehler bei einem Aufruf]: {e}")

        # Thread-Pool starten
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            executor.map(worker, tasks_to_run)

    print(f"\nGenerierung abgeschlossen!")
    print(f"Erfolgreich: {completed_count}")
    print(f"Fehlgeschlagen: {error_count}")
    print(f"Gespeichert in: {args.output}")

if __name__ == "__main__":
    main()
