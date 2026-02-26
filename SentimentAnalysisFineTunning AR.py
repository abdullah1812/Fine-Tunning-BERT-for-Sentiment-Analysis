from transformers.models.auto.modeling_auto import  AutoModelForSequenceClassification
from transformers.models.auto.tokenization_auto import AutoTokenizer
from transformers.training_args import TrainingArguments
from transformers import Trainer
from dotenv import load_dotenv
from datasets import Dataset
import pandas as pd
import os
load_dotenv()

access_token = os.getenv("Hugging_Face_TOKEN")
model_name = "UBC-NLP/MARBERT"

df = pd.read_json(r"Mentoria_Arabic data.json")
label_mapping = {"neutral": 0, "frustrated": 1, 
                'positive':2, 'negative':3, 'grateful':4}

df['label'] = df['sentiment'].map(label_mapping)
dataset = Dataset.from_pandas(df[['text', 'label']])
dataset1 = dataset.train_test_split(test_size=0.2)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(label_mapping))

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length",
                    truncation=True, max_length=128)

tokenized_datasets = dataset1.map(tokenize_function, batched=True)
dataset2 = tokenized_datasets["test"].train_test_split(test_size=0.5) # For val, test Splitting

training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=dataset2["train"],
)

trainer.train()
save_directory = "./mentorea_ar_sentiment_model"
trainer.save_model(save_directory)
tokenizer.save_pretrained(save_directory)

test_metrics = trainer.evaluate(dataset2["test"])
print("Test Scores:", test_metrics)