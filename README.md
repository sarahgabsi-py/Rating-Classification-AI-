# Rating Classification with DistilBERT – NLP Project

# 📌 Project Objective
This project aims to automatically classify the rating (1–5 stars) of text reviews for data science books.
The model used is DistilBERT, a pre-trained Transformer optimized for efficiency and performance.

# The complete workflow includes:
- Dataset analysis
- Preprocessing and balancing
- Tokenization
- Tensor creation
- Model fine-tuning
- Evaluation and error analysis
- Inference using the Hugging Face Pipeline

# 📊 Dataset
Name: Amazon Reviews Science
Size: 20,647 reviews
Format: text + rating (1–5)
Source: local CSV file

- Main columns:
comment → review text
stars → numerical rating
book_url → link to the book

- Exploratory Analysis:
Strong imbalance toward 5-star reviews
Average review length: 87 words
Outliers: from 1 word to 4,868 words

# 📈 Included plots (in /images folder):
- Rating distribution
- Review length distribution
- Class balance

# 🧹 Preprocessing Main operations:
- Convert ratings from 1–5 to 0–4
- Remove empty reviews or reviews with only one word
- Analyze text length
- Undersample the training set to balance classes

# Stratified split:
- 80% train
- 10% validation
- 10% test

# 🔧 Data Processing
Tokenization:
- Model: distilbert-base-uncased
- Parameters: max_length=256, padding="max_length",truncation=True

Tensor creation: Data converted into Hugging Face Dataset objects with columns:
- input_ids
- attention_mask
- label

# 🧠 Model
- Model used: DistilBertForSequenceClassification
- num_labels=5
- Optimized for text classification
- Trained with Hugging Face Trainer

Training parameters:
- Epochs: 3
- Batch size: 8
- Metrics: accuracy
- Checkpoints saved and evaluated at each epoch

# 📈 Results
- Validation accuracy
- Test accuracy
- Best performance 
- Difficulty 

# Inference with Hugging Face Pipeline Implemented a pipeline to:
- Classify new reviews
- Visualize probabilities using softmax
- Interpret the final prediction

# 🛠️ Technologies Used
- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- Pandas
- Matplotlib / Seaborn
