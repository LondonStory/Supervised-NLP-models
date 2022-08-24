# Supervised-NLP-models

## Description
In this repository various pre-trained language models are fine tuned on various labelled datasets. Then these models are applied on the CrowdTangle (Facebook) dataset to perform forward prediction on various downstream NLP tasks such as, sentiment analysis, hate speech detection etc. We use multilingual and bilingual models for benchmarking purposes, and compare their performance below. 

## Aim
The goal is to automatically filter out the toxic contents and identify harmful actors within the CrowdTangle platform.


## Benchmark results from fine-tuning the pre-trained models/ Leaderboard

 Pre-trained model | Language | Training dataset | Method | Batch size | Learning Rate | Epochs | Macro F1 | Link to the code
 --- |---| ---|---|---|---|---|---|---
Hindi-BERT | Hindi | [UMSAB](https://github.com/cardiffnlp/xlm-t/tree/main/data/sentiment/hindi) | Transformers Trainer object | 8 | 2e-5 | 3 | 40.9% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Hindi-BERT-fine-tuning-for-sentiment-analysis-task-using-UMSAB-dataset.ipynb)
Hindi-BERT | Hindi | [Review dataset](https://github.com/LondonStory/Supervised-NLP-models/tree/main/datasets/review-dataset) | Keras | 6 | 1.2e-4 | 3 | 79% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Hindi-BERT-fine-tuning-with-keras-using-review-dataset.ipynb)
Twitter-XLM-roBERTa-base | Hindi | [UMSAB](https://github.com/cardiffnlp/xlm-t/tree/main/data/sentiment/hindi) | Transformers Trainer object | 32 | 2e-5 | 15 | 47.7% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/T-XLM-RoBERTa-base-fine-tuning-for-sentiment-analysis-task-using-UMSAB-dataset.ipynb)
Twitter-XLM-roBERTa-base | Hindi | [Review dataset](https://github.com/LondonStory/Supervised-NLP-models/tree/main/datasets/review-dataset) | Native PyTorch | 16 | 2e-5 | 2 | 89% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/T-XLM-RoBERTa-base-finetuning-with-pytorch.ipynb)


**Remarks on optimised hyperparameter choices in native PyTorch training:**

* We have used only two epochs in training `T-XLM-RoBERTa-base` because afterwards the validation loss starts increasing again. `Epoch 2` is found to be the point of inflection, after which the difference between the training and validation loss starts increasing rapidly (i.e., starts overfitting). 
* The model is seen to train poorly with lower batch size than `16` (experiemented with 10 & 8). 
* Optimum learning rate for `T-XLM-roBERTa-base` model is found to be `2e-5`. Experimented with lower values (i.e., `1.2e-4`), but the training becomes inefficient. 
* Maximum sequence of word piece tokens (greater than which the text snippets are truncated), is chosen to be `312`. This is good enough for experiementing with social media text snippets. If the length is kept at `512`, the performance drops. Also it runs the risk of having cuda memory issues.

## Notebooks on forward inference prediction tasks on CrowdTangle dataset

- [Multilingual sentiment analysis using `XLM-RoBERTa` model](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Multilingual_Sentiment_Analysis_using_XLM_RoBERTa.ipynb)

- [Toxicity inference on CrowdTangle data by using Google's `Perspective API`](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Toxicity-inference-on-CrowdTangle-data-with-Perspective-API.ipynb)
  

