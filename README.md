# Natural Language Processing (NLP) Project

## Overview

This project focuses on developing a Natural Language Processing (NLP) model using Transformers, with a specific emphasis on leveraging various versions of BERT for text classification across diverse datasets. The project incorporates Particle Swarm Optimization (PSO) for hyperparameter optimization and employs multiple datasets to enhance model performance within time and hardware constraints.

## Table of Contents

1. [Introduction](#introduction)
2. [Data](#data)
3. [Preprocessing Data](#preprocessing-data)
4. [Methodology](#methodology)
5. [Results](#results)
6. [Conclusions](#conclusions)
7. [References](#references)

## Introduction

The project aims to provide hands-on experience in developing an NLP model using Transformers. The objective is to broaden the applicability of pre-trained models to classify text data from multiple sources. To enhance model performance, Particle Swarm Optimization (PSO) is incorporated as an optimization technique to search for optimal hyperparameters. Due to time and hardware constraints, multiple datasets are utilized to evaluate the effectiveness of hyperparameter optimization strategies.

### Data

For training and evaluating the pre-trained model, two distinct datasets from different websites are used:

- **Dataset-1:** Named "dair-ai/emotion," this dataset encompasses approximately 20,000 records for emotion detection, categorized into 6 classes.
- **Dataset-2:** Named "CNN News Articles from 2011 to 2022," this dataset comprises approximately 4,000 records categorized into 6 classes.

### Preprocessing Data

Prior to model training, preprocessing is performed on the datasets, including removing irrelevant columns, cleaning the data, and converting it into NumPy arrays (X and y) for compatibility with the model. Each dataset is then partitioned into three subsets: the Training set, the Validation set, and the Test set, using an 80%, 10%, and 10% split of the total data, respectively. The "Auto Tokenizer" provided by the BERT model is utilized to process the text data, converting each record in the dataset into input layers suitable for the model's input format.

## Methodology

Two main approaches are proposed for fine-tuning the pre-trained model:

1. **The Pre-trained Model + A Classification Layer:** This approach employs a BERT model with an untrained 6-class SoftMax layer to train this new layer using various datasets.
2. **The fine-tuning of the model using PSO:** Particle Swarm Optimization (PSO) is used for fine-tuning the model by searching the possible domain of hyperparameters and tracking the improvement in the results using a predefined cost function. PSO does not require gradient information, making it well-suited for optimization problems where derivatives are not readily available or are expensive to compute.

## Results

The outcomes of validation and testing across various datasets using pre-trained models are presented:

1. **The results of the PSO-based hyperparameter tuning model:** PSO method is utilized to determine the optimal hyperparameters for the model. The resulting hyperparameters are then used for fine-tuning the pre-trained model, which incorporates a 6-class SoftMax classifier.
2. **The results of the model with optimal parameters:** The optimal parameters are employed to train the BERT and Electra models alongside a 6-class SoftMax layer, resulting in improved accuracy, F1 score, and loss values across different databases.

## Conclusions

The project provides valuable insights into the development and optimization of NLP models using state-of-the-art techniques. Further research and experimentation, particularly with larger datasets, could lead to enhanced performance and broader applications of these models in real-world scenarios.

## References

1. [dair-ai, Hugging Face](https://huggingface.co/datasets/dair-ai/emotion)
2. [CNN News Articles from 2011 to 2022, Kaggle](https://www.kaggle.com/datasets/hadasu92/cnn-articles-after-basic-cleaning)
