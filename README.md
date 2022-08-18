# Supervised-NLP-models
Various supervised NLP models are applied on the CrowdTangle (Facebook) dataset for tasks like, sentiment analysis, hate speech detection etc. We use multilingual and bilingual models for benchmarking purposes, and compare their performance. 

The goal is to automatically filter out the toxic contents and identify harmful actors within the CrowdTangle platform.


## Benchmark results


 Pre-trained model | Language | Training dataset | Method | Batch size | Learning Rate | Epochs | Macro F1 | Link
 --- |---| ---|---|---|---|---|---|---
Hindi-BERT | Hindi | UMSAB | Transformers Trainer object | 8 | 2e-5 | 3 | 40.9% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Hindi-BERT-fine-tuning-for-sentiment-analysis-task-using-UMSAB-dataset.ipynb)
Twitter-XLM-roBERTa-base | Hindi | UMSAB | Transformers Trainer object | 32 | 2e-5 | 15 | 47.7% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/T-XLM-RoBERTa-base-fine-tuning-for-sentiment-analysis-task-using-UMSAB-dataset.ipynb)
Hindi-BERT | Hindi | Review dataset | Keras | 6 | 1.2e-4 | 3 | 79% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Hindi-BERT-fine-tuning-with-keras-using-review-dataset.ipynb)
 

  

