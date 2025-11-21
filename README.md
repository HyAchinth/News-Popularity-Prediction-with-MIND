# Predicting News Popularity with Supervised Machine Learning

## Overview

This project aims to **predict the popularity of news articles**, defined by the **clicks-to-impressions ratio**, using a combination of classification and regression models. We engineered features from user behavior and article metadata, integrated real-time interest signals via Google Trends, and developed a two-step machine learning pipeline for high-impact prediction.

---

## Problem Statement

**Can we accurately forecast the success of a newly published news article based on available user and article features?**

**Stakeholders:**  
- News platforms and content publishers

**Business Value:**  
- Enables strategic content selection and promotion  
- Informs personalized recommendation systems  
- Helps maximize user engagement and revenue

---

## Dataset

**Source:** [Microsoft News Dataset (MIND)](https://msnews.github.io/)

- **Behaviors Dataset**  
  Click histories and impression logs used to compute the target variable (click %).

- **News Dataset**  
  Article metadata including title, category, and abstract.

---

## Tech Stack

- Python (NumPy, Pandas, Scikit-learn)
- XGBoost, CatBoost, SVR
- Google Trends API
- ChatGPT (Prompt Engineering for NLP)
- Colab (Training + Evaluation)

---

## Key Features & Engineering

- **Parallel processing** reduced dataset size from 5M rows to 55K, achieving a **60x speedup**.
- **Prompt engineering with ChatGPT** to summarize articles and extract keywords.
- **Google Trends API integration** to enrich features with real-time interest signals.
- Log transformation of the skewed target variable.
- Customized cost function in classification to emphasize revenue impact.

---

## Model Architecture

### Step 1: Classification (Clicks vs. No Clicks)
- Top models: **Logistic Regression (balanced)**, **CatBoost**
- Ensemble: **Voting Classifier**
- Achieved a **5x performance improvement over null model**

### Step 2: Regression (Clicks-to-Impressions Ratio)
- Top models: **Support Vector Regressor (SVR)**, **CatBoost Regressor**
- Ensemble: **Voting Regressor**
- **Reduced RMSE by 10%** after hyperparameter tuning

---

## Challenges Faced

- Highly imbalanced and skewed dataset
- Sparse non-zero target values
- Limitations of weak learners in regression
- Latency in model training and evaluation
- Issues with stacking/voting ensemble behavior

---

## Key Insights

- **Prompt engineering + Google Trends** enhanced feature richness significantly.
- Classification before regression helped improve predictive stability.
- Custom cost function aligned model performance with stakeholder goals (profitability).
- Real-world applications can expand to **digital ads, YouTube video predictions**, and other media platforms.

---

## Presentation Highlights

- Delivered findings to an audience of **50+ peers**
- Received recognition for **innovative integration of ensembling, prompt engineering**, and **custom ML workflows**






