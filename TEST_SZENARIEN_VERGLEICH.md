# Test-Szenarien & Modell-Vergleich: Schüler-Schreibstil Fine-Tuning

Dieses Dokument fasst die durchgeführten Test-Szenarien und Evaluierungen des gefinetunten Schülerstil-Modells (**`schueler_model_merged`**) im Vergleich zum originalen Basismodell (**Mistral 7B Instruct**) zusammen.

---

## 📋 Szenario 1: Direktvergleich (Original-Modell vs. Fine-Tuned Schülerstil)

In diesem Szenario wurden beide Modelle mit identischen System- und User-Prompts auf der **NVIDIA GeForce RTX 5070 Ti GPU** evaluiert.

### Vergleichstabelle

| Frage / Testfall | Antwort Original-Modell *(Mistral 7B Instruct)* | Antwort Fine-Tuned Schülerstil-Modell *(Persona: Der Pragmatiker)* |
| :--- | :--- | :--- |
| **1. Gedichtanalysen in der Schule**<br>*"Warum muss man in der Schule eigentlich immer noch Gedichtanalysen schreiben?"* | Gedichtanalysen in der Schule sind weiterhin relevant, weil sie uns helfen, unsere kritische Fähigkeit zu entwickeln und unsere Interpretationsfähigkeit zu verbessern.<br><br>Gedichtanalysen erfordern eine kritische Analyse des Gedichts, um seine Bedeutung, seine Struktur und seine Sprache zu verstehen... *(Fließtext ohne klare Struktur)* | Gedichtanalysen in der Schule sind weiterhin relevant, weil sie uns helfen, unsere kritischen Fähigkeiten zu entwickeln:<br><br>**1. Kritisches Denken:** Analyse der Intention des Autors und symbolischen Bedeutungen.<br>**2. Interpretationsfähigkeit:** Verschiedene Perspektiven und Bedeutungen verstehen.<br>*(Pragmatisch nummerierte Punkte)* |
| **2. KI-Tools & Hausarbeiten**<br>*"Sollten KI-Tools wie ChatGPT für Hausarbeiten in der Schule erlaubt sein?"* | Ja, es ist möglich, dass KI-Tools wie ChatGPT für Hausarbeiten in der Schule erlaubt werden. Es ist jedoch wichtig, dass die Nutzung solcher Tools transparent ist... Die Schüler sollten die KI-Tools als Hilfe zur Lösung von Problemen und zur Erweiterung ihrer eigenen Wissensbasis nutzen. Es ist auch wichtig, dass die Schüler die Quellen korrekt zitieren und Urheberrechte respektieren. | Ja, es ist möglich, dass KI-Tools wie ChatGPT für Hausarbeiten erlaubt werden. Die Nutzung muss transparent sein und die Schüler erkennen, dass sie nur als Hilfsmittel dienen.<br><br>Wichtig ist, dass die Schüler Quellen korrekt zitieren **und die Arbeit selbstständig erarbeiten**. *(Fokus auf Eigenleistung im Schulkontext)* |
| **3. Vorbereitung auf Klausuren**<br>*"Wie bereitest du dich am besten auf eine schwere Klausur vor?"* | Um sich am besten auf eine schwere Klausur vorzubereiten, sollte man folgende Schritte befolgen:<br>1. **Klausurplan**: Erstellen **Sie** einen Klausurplan...<br>2. **Studienplan**: Planen **Sie** regelmäßige Studienzeiten...<br>3. **Studienmaterial**: Verwenden **Sie** geeignetes Studienmaterial...<br>*(Fällt in distanzierte Sie-Form / Lehrbuchsprache)* | Um sich am besten auf eine schwere Klausur vorzubereiten, sollte man folgende Schritte beachten:<br>1. **Klausurplanung**: Themen identifizieren, die in der Klausur vertreten sein werden.<br>2. **Studienplan**: Auf prüfungsrelevante Themen konzentrieren.<br>3. **Studienmaterial**: Notizen, Übungen & Online-Ressourcen bündeln.<br>*(Schülergerechte Ansprache ohne Sie-Form)* |
| **4. Studium vs. Berufseinstieg**<br>*"Lohnt sich ein Studium nach dem Abitur überhaupt noch oder sollte man lieber direkt arbeiten?"* | Es ist eine persönliche Entscheidung, ob man nach dem Abitur ein Studium absolvieren oder direkt in die Arbeit eintreten sollte. Es gibt Vorteile und Nachteile für beide Wege.<br>Ein Studium kann Vorteile wie eine breite Ausbildung bieten... Dagegen kann eine direkte Arbeit Vorteile wie praktische Erfahrung bieten... *(Neutraler Aufsatzstil)* | **Ja, ein Studium nach dem Abitur kann sich lohnen**, wenn man seine Ziele klar definiert. Es bietet die Möglichkeit, Fachkenntnisse für eine erfolgreiche Karriere zu erwerben.<br><br>Allerdings ist es auch möglich, direkt nach dem Abitur zu arbeiten und sich später zu qualifizieren. Wichtig ist, sich über die Anforderungen des Berufsfelds zu informieren.<br>*(Direkte Positionierung & schlüssige Abwägung)* |

---

## 🔄 Szenario 2: KI-Text De-AI-fizieren / Style Transfer (Entfernung von KI-Floskeln)

Dieses Szenario testet die Fähigkeit des Modells, extrem steifen und KI-typischen Text (mit ChatGPT-Standardfloskeln) in einen natürlichen Schülerschreibstil umzuformulieren.

### Testfall: KI-Text Umformulierung

#### 1. Original KI-Text *(Mit typischen ChatGPT-Mustern)*:
> *"In der heutigen, sich rasant entwickelnden Welt spielt das Thema Erneuerbare Energien eine essenzielle und facettenreiche Rolle. Zusammenfassend lässt sich feststellen, dass der Übergang zu nachhaltigen Energiequellen nicht nur ökologische Synergien schafft, sondern auch sozioökonomische Chancen eröffnet. Es ist unbestreitbar, dass die Transformation unseres Energiesystems eine gewaltige Herausforderung darstellt; dennoch ist es von herausragender Bedeutung, den Blick nach vorne zu richten, um eine nachhaltige Zukunft für kommende Generationen zu gewährleisten."*

#### 2. Umformulierter Text *(Vom Fine-Tuned Schülerstil-Modell)*:
> *"Erneuerbare Energien spielen heute eine wichtige Rolle, weil sie sowohl für die Umwelt als auch für die Wirtschaft neue Chancen bringen. Auch wenn die Umstellung unseres Energiesystems eine große Herausforderung ist, müssen wir nach vorne blicken, um eine nachhaltige Zukunft für die nächsten Generationen zu sichern."*

### Stil-Analyse der Korrekturen

| Typische KI-Floskel (Original) | Umwandlung durch das Schülerstil-Modell | Analyse / Auswirkung |
| :--- | :--- | :--- |
| *"In der heutigen, sich rasant entwickelnden Welt..."* | *"Erneuerbare Energien spielen heute..."* | Bekannteste KI-Einleitung wird ersatzlos gestrichen. |
| *"eine essenzielle und facettenreiche Rolle"* | *"eine wichtige Rolle"* | Übertriebene Doppel-Adjektive werden vereinfacht. |
| *"Zusammenfassend lässt sich feststellen..."* | *(direkter Satzanschluss)* | Künstliche Aufsatz-Überleitungen entfallen. |
| *"ökologische Synergien schafft & sozioökonomische Chancen eröffnet"* | *"für die Umwelt als auch für die Wirtschaft neue Chancen bringen"* | Abstrakte Soziologensprache wird in klare Alltagssprache übersetzt. |
| *"von herausragender Bedeutung / unbestreitbar"* | *"müssen wir..."* | Passive Passiv-Konstruktionen werden durch aktive Sätze ersetzt. |

---

> [!NOTE]
> Beide Test-Szenarien wurden direkt auf der **NVIDIA GeForce RTX 5070 Ti Laptop GPU** im lokalen Workspace ausgeführt und validiert.
