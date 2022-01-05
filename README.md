# Spam Email Classifier Models

This repository contains two notebooks that create and deploy a machine learning model that can be used to classify email as _spam_ or _ham_.

_Ham_ is the term used to define emails that aren't _spam_.

The dataset used for training is the <a href='http://www2.aueb.gr/users/ion/data/enron-spam/'>enron spam email dataset</a>.

Determining whether a given email is _spam_ or not can be considered a _classification_ problem. The model to be trained should be able to generalize based on what it learned from the dataset and predict whether a new given email is _spam_ or not. This is also a _Natural Language Processing_ problem, as we are dealing with text data, and we want our model to be able to analize it in order to make predictions.

Based on the format of the dataset, different text pre-processing techniques will be used to transform the raw email data into a cleaner, and simplified version of the text in each email, that can be later used for training.

On top of that, not all machine learning algorithms can deal with text data. Whenever needed, these text data will be converted to vectors or real numbers, this is done using different text vectorization techniques (also known as word embeddings).

The resulting model will provide a probability value of a given email to be either _ham_ or _spam_.

## About the notebooks
### *spam_detection_local*

The first notebook was written with the intent to train a classifier model locally, without using a SageMaker Notebook Instance or SageMaker Studio. Once a model is trained, the model is serialized locally. At this point, a session is established to an AWS Account where the model artifact is uploaded to S3, and then deployed to a SageMaker Endpoint.

This notebook downloads the dataset, then performs exploratory data analysis to understand the data structure and distribution; It then performs feature engineering, which includes text pre-processing and vectorization. For training it uses a *sci-kit learn* pipeline where all the preprocessing and vectorization happens as previous steps before training the algorithms.

Using pipelines, it allows us to quickly test different variations of normalizaton and vectorization tecniques, as well as different classification algorithms:

* Text Normalization options: Simple, Stemming, Lemmatization.
* Vectorization Options: Simple, Bag of Words, and TF-IDF (on top of bag of words).
* Classifiers: Naive Bayes, Support Vector Machine (SVM), K Nearest Neighbors, Random Forest  

### *spam_detection_aws*

The second notebook builds upon the first one, this time, instead of working lcoally, the notebook was written with the intent of working in a SageMaker environment (either Notebook or SageMaker Studio). It defines a SageMaker data pipeline, that performs the same steps as the first notebook: loading the dataset, text preprocessing, model training and model deployment. The big difference here is that this time, all the pipeline steps are executing using separate compute instances. This allows us to scale up all the steps in the workflow, as opposed to running them using a single local computer.

This notebook uses one of the out of box algorithms provided by SageMaker for text processing: BlazingText.

## Model evaluation

The important metrics to look for are:

* **Accuracy** - Ratio of correctly predicted observations against total # of observations. How many predictions (ham and spam) did the model get right?
* **Precision** - Ratio of correctly predicted positive observations against total predicted positive observations. Of all emails classified as ham/spam, how many were actually ham/spam?
* **Recall** - Also known as sensitivity. Ratio of correctly predicted positive observations against all observations that were actually positive. Of all the emails that are actually ham/spam, how many were corretly labeled as ham/spam?

For this case, the cost of incorrectly classifying a valid ("ham") email as spam is higher than incorrectly classifying a spam email as "ham".

Therefore, in addition to **accuracy**, we need to play close attention to the following:
 1. **recall metric for ham predictions**, which will tell us the percentage of emails that are correctly classsified as ham.
 2. **precision metric for spam predictions**, which will tell us the percentage of emails classified as spam, that are actually spam.