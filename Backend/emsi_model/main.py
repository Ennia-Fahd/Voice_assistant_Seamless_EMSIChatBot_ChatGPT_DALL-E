import fitz
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, \
    TrainingArguments

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Étape 1: Extraction du texte du PDF
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""

    for page_number in range(doc.page_count):
        page = doc[page_number]
        text += page.get_text()

    return text

# Spécifiez le chemin vers votre fichier PDF
pdf_path = "dataset.pdf"

# Extraire le texte du PDF
dataset_text = extract_text_from_pdf(pdf_path)

# Étape 2: Fine-tuning du modèle GPT-2
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Sauvegarder le texte extrait dans un fichier pour le fine-tuning
output_text_path = "dataset.txt"
with open(output_text_path, "w", encoding="utf-8") as output_file:
    output_file.write(dataset_text)

# Charger et prétraiter les données pour le fine-tuning
train_data = TextDataset(
    tokenizer=tokenizer,
    file_path=output_text_path,
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# Définir les paramètres d'entraînement
training_args = TrainingArguments(
    output_dir="./gpt2-fine-tuned",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
)

# Créer l'entraîneur et effectuer le fine-tuning
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_data,
)

trainer.train()

# Étape 3: Génération de réponses en fonction des requêtes
def generate_response(prompt, model, tokenizer, max_length=100):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=max_length, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text



# Endpoint for generating responses
@app.get("/EMSI_Model")
async def generate_response(sentence: str):
    # Use the fine-tuned model and tokenizer to generate a response
    input_ids = tokenizer.encode(sentence, return_tensors="pt")
    output = model.generate(input_ids, max_length=100, num_beams=5, no_repeat_ngram_size=2, top_k=50, top_p=0.95)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"original_text": sentence, "generated_response": generated_text}



