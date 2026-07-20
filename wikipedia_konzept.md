# Konzept: Integration von Wikipedia-Diskussionsstilen für Schülerstil-Fine-Tuning

Dieses Konzept beschreibt, wie wir den typischen Diskussionsstil der deutschsprachigen Wikipedia nutzen können, um den synthetischen Schreibstil-Datensatz für Schüler der gymnasialen Oberstufe qualitativ aufzuregen.

---

## 1. Problemstellung und Motivation
Der bisherige Datensatz simuliert Schüler-Personas gut, neigt jedoch manchmal dazu, in einen standardisierten, glatten "KI-Erklärstil" zurückzufallen.
Um dem Modell das typische **besserwisserische, argumentative und debattierfreudige** Verhalten beizubringen (insbesondere für den PoWi- und Deutsch-Unterricht), brauchen wir echte menschliche Vorlagen für Streitkultur, Wortklauberei und detailverliebte Argumentation.

Wikipedia-Diskussionen (z. B. auf den Seiten `Wikipedia:Auskunft`, `Wikipedia:Meinungsbilder` oder in den Fachredaktionen) bieten hierfür einen idealen Textkorpus:
- **Formulierungsmuster:** "Zu bedenken ist...", "Hier werden Ursache und Wirkung vertauscht...", "Das greift zu kurz...".
- **Tonfall:** Formell, bemüht objektiv, oft leicht passiv-aggressiv, extrem detailorientiert.
- **Struktur:** These -> Gegenthese -> Belegschlacht.

---

## 2. Architekturentwurf: Synthetische Veredelung (Few-Shot)

Die rohe Übernahme von Wikipedia-Diskussionen scheitert in der Praxis an der unsauberen Formatierung (Wiki-Syntax, unvollständige Sätze, Meta-Diskussionen). Die zielführendste Lösung ist daher die **synthetische Veredelung (Ansatz B)**:

```
+-------------------------------------------------------+
|  1. Extraktion typischer Argumentationsmuster        |
|  aus Wikipedia-Foren (Auskunft, Meinungsbilder)       |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  2. Integration der Muster als "Stilvorlagen" in      |
|  das Python-Generierungsskript (generate_dataset.py)  |
+---------------------------+---------------------------+
                            |
                            v
+-------------------------------------------------------+
|  3. Das Erzeuger-LLM nutzt diese Vorlagen, um eine    |
|  Schülerantwort im gewünschten Stil zu schreiben      |
+-------------------------------------------------------+
```

---

## 3. Konkrete Implementierungsschritte

### Schritt A: Die Stil-Datenbank (`wikipedia_style_samples.json`)
Wir erstellen eine JSON-Datei, die typische argumentative Satzanfänge und Textblöcke aus echten Wikipedia-Diskussionen enthält.
Beispiele:
- *Besserwisserisch:* "Es greift zu kurz, hier lediglich von X zu sprechen, da der historische Kontext von Y völlig ausgeblendet wird."
- *Streitbar:* "Dieser Argumentation muss vehement widersprochen werden. Ein Blick auf die Daten zeigt..."
- *Bemüht-sachlich:* "Im Hinblick auf die obigen Ausführungen ist festzustellen, dass zwar X zutrifft, des Weiteren aber Z vernachlässigt wurde."

### Schritt B: Erweiterung des System-Prompts in `generate_dataset.py`
Das Generierungsskript wird so modifiziert, dass es beim Erzeugen einer Antwort (insbesondere für die Personas "Der Bemühte" und "Der Überflieger") dem Erzeuger-LLM diese Stilvorlagen als Inspirationsquelle übergibt.
Der Prompt wird ergänzt um:
> *"Lass dich bei der Argumentation und dem Tonfall von folgenden echten Diskussionsmustern inspirieren. Verwende ähnliche Satzstrukturen und argumentative Übergänge, um den Text so klingen zu lassen, als würde der Schüler bemüht debattieren: {wikipedia_style_sample}"*

### Schritt C: Testlauf und Filterung
Über das Generierungsskript lassen wir Testläufe durchführen. Die Qualitätskontrolle filtert weiterhin ungeeignete Ausreißer aus.

---

## 4. Vorteile dieser Lösung
1. **Kein Datenmüll:** Die generierten Texte bleiben im reinen Aufgaben-Antwort-Format (keine Signaturen, keine Wiki-Syntax).
2. **Kombinierbarkeit:** Der Wikipedia-Stil wird harmonisch mit den Schüler-Personas gekreuzt (z.B. der *Bemühte*, der sich in Wikipedia-artigen Details verheddert, oder der *Überflieger*, der elegant wie ein erfahrener Wikipedia-Autor argumentiert).
3. **Effizienz:** Keine Notwendigkeit, Gigabytes an Wikipedia-Dumps herunterzuladen und lokal zu parsen.
