from transformers.models.auto.modeling_auto import  AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers import Trainer
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
import os
load_dotenv()

# access_token = os.getenv("Hugging_Face_TOKEN")
df = pd.read_csv(r"mentorea_english_general.csv")
label_to_id = {
    'LABEL_0': 0, # neutral
    'LABEL_1': 1, # frustrated
    'LABEL_2': 2, # positive
    'LABEL_3': 3, # negative
    'LABEL_4': 4  # grateful
}
id_to_label = {v: k for k, v in label_to_id.items()}
df['label'] = df['label'].map(label_to_id)

dataset = Dataset.from_pandas(df[['text', 'label']])
dataset = dataset.train_test_split(test_size=200)

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_to_id),
    label2id=label_to_id,
    id2label=id_to_label
)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = dataset.map(tokenize_function, batched=True)
training_args = TrainingArguments(
    output_dir="mentorea_english_checkpoints",
    eval_strategy="epoch",                
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,                        
    weight_decay=0.01, 
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)
trainer.train()

stage1_path = "./mentorea_english_stage1"
trainer.save_model(stage1_path)
tokenizer.save_pretrained(stage1_path)

# ===> Second FineTunning

df2 = pd.read_csv(r"mentorea_4k_synthetic_feedback.csv")
label_mapping = {"neutral": 0, "frustrated": 1, 
                'positive':2, 'negative':3, 'grateful':4}
id_to_label2 = {v: k for k, v in label_mapping.items()}
df2['label'] = df2['label'].map(label_mapping)

dataset2 = Dataset.from_pandas(df2[['text', 'label']])
dataset2 = dataset2.train_test_split(test_size=0.2)

tokenized_datasets2 = dataset2.map(tokenize_function, batched=True)

model_stage2 = AutoModelForSequenceClassification.from_pretrained(
    stage1_path, 
    num_labels=len(label_to_id),
    label2id=label_to_id,
    id2label=id_to_label
)
training_args_stage2 = TrainingArguments(
    output_dir="mentorea_english_final_checkpoints",
    eval_strategy="epoch",                
    learning_rate=1e-5, 
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=2,                
    weight_decay=0.01,  
)
trainer_stage2 = Trainer(
    model=model_stage2,
    args=training_args_stage2,
    train_dataset=tokenized_datasets2["train"],
    eval_dataset=tokenized_datasets2["test"],
)

trainer_stage2.train()

final_path = "./mentorea_english_model_final"
trainer_stage2.save_model(final_path)
tokenizer.save_pretrained(final_path)