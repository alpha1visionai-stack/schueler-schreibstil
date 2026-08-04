import os
import time
import shutil

p = r"schueler_gguf\unsloth.Q4_K_M.gguf"
print("Warte auf Fertigstellung der GGUF-Datei...")

while not os.path.exists(p):
    time.sleep(3)

print(f"GGUF-Datei gefunden ({os.path.getsize(p) / (1024**3):.2f} GB)! Kopiere nach LM Studio...")
target_dir = r"C:\Users\walte\.lmstudio\models\schueler-style\schueler-mistral-7b-gguf"
os.makedirs(target_dir, exist_ok=True)
target_path = os.path.join(target_dir, "schueler-mistral-7b.Q4_K_M.gguf")

shutil.copyfile(p, target_path)
print(f"[ERFOLG] GGUF-Datei kopiert nach: {target_path}")
