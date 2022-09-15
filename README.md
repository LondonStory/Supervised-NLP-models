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
Twitter-XLM-roBERTa-base | Hindi | [Review dataset](https://github.com/sid573/Hindi_Sentiment_Analysis) | Native PyTorch | 16 | 2e-5 | 2 | 89% | [Notebook](https://github.com/LondonStory/Supervised-NLP-models/blob/main/T-XLM-RoBERTa-base-finetuning-with-pytorch.ipynb)


**Remarks on optimised hyperparameter choices in native PyTorch training:**

* We have used only two epochs in training `T-XLM-RoBERTa-base` because afterwards the validation loss starts increasing again. `Epoch 2` is found to be the point of inflection, after which the difference between the training and validation loss starts increasing rapidly (i.e., starts overfitting). 
* The model is seen to train poorly with lower batch size than `16` (experiemented with batch sizes `10` and `8`). 
* Optimum learning rate for `T-XLM-roBERTa-base` model is found to be `2e-5`. Experimented with lower values (i.e., `1.2e-4`), but the training becomes inefficient. 
* Maximum sequence of word piece tokens (greater than which the text snippets are truncated), is chosen to be `312`. This is good enough for experiementing with social media text snippets. If the length is kept at `512`, the performance drops. Also it runs the risk of having cuda memory issues.

**Elements of the confusion matrix**


- `True positives (TP)` are positive outcomes that the model predicts correctly
- `True Negatives (TN)` are negative outcomes that the model predicted correctly
- `False Positives (FP)` or `Type I error` are positive outcomes that the model has predicted incorrectly
- `False Negatives (FN)` or `Type II error` are negative outcomes that the model predicted incorrectly

**Metrics of model evaluation**

- `Accuracy` = number of correct predictions / total predictions
- `Precision` = TP / (TP + FP)
- `Recall` = TP / (TP + FN)
- `F1-score` = 2 x (Precision x Recall)/ (Precision + Recall)

In other words, precision is the percentage of outcomes that are positive. And recall is the percentage of actual positives that were correctly identified. And F1-score is the harmonic mean of precision and recall scores. 

**Case 1.** Having high precision but low recall means that although the model is good at predicting the positive class, it only detects a small proportion of the total number of positive outcomes. Therefore the model is under-predicting.

**Case 2.** Having low precision but high recall means that although the model correctly predicted most of the positive cases, it also predicted a lot of negatives to be positive too. Therefore, the model is over-predicting.


## Notebooks on forward inference prediction tasks on CrowdTangle dataset

- [Multilingual sentiment analysis using `XLM-RoBERTa` model](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Multilingual_Sentiment_Analysis_using_XLM_RoBERTa.ipynb)

- [Toxicity inference on CrowdTangle data by using Google's `Perspective API`](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Toxicity-inference-on-CrowdTangle-data-with-Perspective-API.ipynb)
  
- [Sentiment prediction task (forward inference) on CrowdTangle's dataset by using the `fine-tuned Hindi language XLM-roBERTa` model (with 89% F1 score)](https://github.com/LondonStory/Supervised-NLP-models/blob/main/Forward-prediction-of-Hindi-sentiment-analysis.ipynb). For the results look [HERE](https://github.com/LondonStory/Supervised-NLP-models/tree/main/results)

## Translation of CrowdTangle dataset (Hindi to English)

- [Translating CrowdTangle Hindi entries to English for analysing the quality of forward prediction task]()

