# Fine-Tunning-BERT-for-Sentiment-Analysis
FineTunning Bert Base Models For Sentiment Analysis for Mentorea Platform

## 📝 Project Overview
This project aims to build highly accurate Artificial Intelligence models for Sentiment Analysis in both Arabic and English.
It leverages Deep Learning techniques and the fine-tuning of state-of-the-art pre-trained language models to achieve exceptional accuracy that closely mirrors real-world, everyday language.

## 🛠️ Technologies & Environment
* **Programming Language:** Python 3.13.9
* **Core Library:** Transformers 5.2.0 (Hugging Face)

---

## 🌍 Arabic Model
To analyze sentiment in Arabic text, or Mixing Arabic with Englishm a robust Arabic-specific model was fine-tuned on a high-quality, custom dataset:

* **Base Model:** `UBC-NLP/MARBERT`
* **Dataset:** A custom dataset of **25,000 samples** was carefully generated and curated using the **Gemini**. This data was highly engineered to accurately represent real-world usage and everyday colloquial language.
* **Results:** The model demonstrated exceptional performance and remarkable stability during evaluation:
  * **Validation Loss:** `0.000036`
  * **Validation Accuracy:** `0.998`
  * **Test Accuracy:** `0.995`

---

## 🌐 English Model
For the English language, an innovative Two-Stage Fine-tuning approach was adopted to ensure maximum contextual understanding:

* **Base Model:** `distilbert-base-uncased`
* **Stage 1 (General Fine-tuning):** The model was initially trained on a **Large-scale General Sentiment Dataset**, 38000 sample, an experimental baseline to test its ability to capture broad sentiments.
* **Stage 2 (Domain-Specific Fine-tuning):** A secondary fine-tuning process was executed using **4,000 samples** specifically designed to be extremely close to highly realistic and precise scenarios.
* **Results:** Following the second stage, the model achieved outstanding results, matching the exceptional performance of the Arabic model in understanding and analyzing complex sentiments.

