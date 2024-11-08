# Fake News Detection Case Study

This project is based on the "Intro to NLP for AI" course from 365 Data Science. It follows a guided case study on building a machine learning model to categorize news articles as either factual or fake using Natural Language Processing (NLP) techniques. The project leverages various Python libraries for text processing, feature extraction, and model evaluation.
Table of Contents

    Project Overview
    Objectives
    Dataset
    Libraries Used
    Approach
    Model Evaluation
    Results
    Conclusion
    References

Project Overview

The goal of this project is to classify news articles into "Factual News" or "Fake News" using NLP techniques and machine learning algorithms. This project serves as a practical application of NLP concepts covered in the 365 Data Science course.
Objectives

    Understand the basics of NLP and text classification.
    Implement text preprocessing techniques such as tokenization, stopword removal, and vectorization.
    Train machine learning models to classify news articles.
    Evaluate model performance using metrics like accuracy, precision, recall, and F1-score.

Dataset

The dataset used in this project consists of labeled news articles categorized into two classes:

    Factual News
    Fake News

The dataset was provided as part of the 365 Data Science course.
Libraries Used

This project makes use of the following Python libraries:

    pandas for data manipulation
    numpy for numerical computations
    nltk for text preprocessing (tokenization, stopword removal)
    sklearn for machine learning models and evaluation
    vaderSentiment for sentiment analysis (optional)
    transformers for advanced NLP models (optional)

Approach

    Data Preprocessing:
        Load and inspect the dataset.
        Perform text cleaning and preprocessing, including:
            Lowercasing
            Tokenization
            Removing stopwords
            Vectorizing the text using CountVectorizer

    Model Training:
        Split the data into training and testing sets.
        Train multiple models, including:
            Logistic Regression
            Stochastic Gradient Descent (SGD) Classifier

    Model Evaluation:
        Evaluate the models using accuracy, precision, recall, and F1-score.
        Generate classification reports for a detailed breakdown of performance.

Model Evaluation

The following models were evaluated:

    Logistic Regression:
        Accuracy: ~92%
        Precision, Recall, F1-score:

    Factual News       - Precision: 0.88, Recall: 0.97, F1-score: 0.92
    Fake News          - Precision: 0.96, Recall: 0.86, F1-score: 0.91

SGD Classifier:

    Accuracy: ~88%
    Precision, Recall, F1-score:

        Factual News       - Precision: 0.85, Recall: 0.94, F1-score: 0.89
        Fake News          - Precision: 0.92, Recall: 0.83, F1-score: 0.87

Results

    The Logistic Regression model performed better in terms of overall accuracy and F1-score compared to the SGD Classifier.
    The model can effectively classify news articles, achieving an accuracy of over 90%.

Conclusion

This project demonstrates the effectiveness of NLP techniques combined with machine learning algorithms in categorizing text data. It highlights the importance of feature extraction, text preprocessing, and model evaluation in building robust NLP models.
References

    365 Data Science course on "Intro to NLP for AI"
    Scikit-learn documentation: https://scikit-learn.org
    NLTK documentation: https://www.nltk.org
