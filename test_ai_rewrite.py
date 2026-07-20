import torch
from unsloth import FastLanguageModel

print("=" * 60)
print("Test: Extrem KI-lastigen Text im Schülerstil umschreiben")
print("=" * 60)

model_path = "schueler_model_merged"
max_seq_length = 2048

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = model_path,
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)
FastLanguageModel.for_inference(model)

# Typischer extrem KI-lastiger ChatGPT-Text mit allen bekannten KI-Floskeln:
extreme_ai_text = """In der heutigen, sich rasant entwickelnden Welt spielt das Thema Erneuerbare Energien eine essenzielle und facettenreiche Rolle. Zusammenfassend lässt sich feststellen, dass der Übergang zu nachhaltigen Energiequellen nicht nur ökologische Synergien schafft, sondern auch sozioökonomische Chancen eröffnet. Es ist unbestreitbar, dass die Transformation unseres Energiesystems eine gewaltige Herausforderung darstellt; dennoch ist es von herausragender Bedeutung, den Blick nach vorne zu richten, um eine nachhaltige Zukunft für kommende Generationen zu gewährleisten."""

messages = [
    {"role": "system", "content": "Du bist ein pragmatischer Oberstufenschüler. Formuliere den folgenden Text so um, dass alle gestelzten KI-Floskeln entfernt werden und der Text wie ein echter Schüleraufsatz klingt."},
    {"role": "user", "content": f"Schreibe diesen extrem KI-lastigen Text im natürlichen Schülerschreibstil um:\n\n{extreme_ai_text}"}
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_tensors = "pt",
).to("cuda")

outputs = model.generate(
    input_ids = inputs,
    max_new_tokens = 300,
    use_cache = True,
    temperature = 0.7,
    min_p = 0.1,
)

response = tokenizer.batch_decode(outputs)[0]
ans = response.split("[/INST]")[-1].replace("</s>", "").strip() if "[/INST]" in response else response

print("\nEXTREM KI-LASTIGER ORIGINALTEXT:")
print("-" * 60)
print(extreme_ai_text)
print("\nUMGEWANDELT IM SCHÜLERSTIL:")
print("-" * 60)
print(ans)
