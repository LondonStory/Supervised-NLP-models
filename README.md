# Supervised-NLP-models
Various supervised NLP models are applied on the CrowdTangle (Facebook) dataset for tasks like, sentiment analysis, hate speech detection etc. We use multilingual and bilingual models for benchmarking purposes, and compare their performance. 

The goal is to automatically filter out the toxic contents and identify harmful actors within the CrowdTangle platform.


## Benchmark results


 Pre-trained model | Language | Training dataset | Method | Batch size | Learning Rate | Epochs | Macro F1 
 --- |---| ---|---|---|---|---|---
 Twitter-XLM-roBERTa-base | Hindi | UMSAB | Transformers Trainer object | 32 | 2e-5 | 15 | 47.7%  

  

