# 🚢 Titanic Survival Prediction – Data Cleaning & Preprocessing

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-yellow)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-orange)
![Status](https://img.shields.io/badge/Project-Completed-brightgreen)

![Titanic Banner](titanic.png)
---

## 📌 Project Overview

This project focuses on transforming raw Titanic dataset data into a machine-learning-ready format through systematic preprocessing techniques.

The objective was to clean, structure, and optimize the dataset to improve downstream model performance.

---

## 📂 Dataset Information

- Source: Kaggle Titanic Dataset  
- Rows: 891  
- Target Variable: `Survived`  
- Problem Type: Binary Classification  

---

## 🔁 Preprocessing Workflow

Raw Data  
↓  
Missing Value Handling  
↓  
Feature Encoding  
↓  
Outlier Removal  
↓  
Feature Scaling  
↓  
Cleaned Dataset  

---

## 🛠 Data Cleaning Steps

### 1️⃣ Missing Value Treatment
- Age → Median Imputation  
- Embarked → Mode Imputation  
- Cabin → Dropped  

### 2️⃣ Feature Selection
Removed:
- PassengerId  
- Name  
- Ticket  

### 3️⃣ Encoding
Applied One-Hot Encoding for:
- Sex  
- Embarked  

### 4️⃣ Outlier Removal
Used IQR method to eliminate extreme values.

### 5️⃣ Feature Scaling
Applied StandardScaler to normalize numerical distributions.

---

## 📊 Visual Analysis

### Survival Distribution
![Survival](images/survival_distribution.png)

### Age Distribution
![Age](images/age_distribution.png)

### Fare Distribution
![Fare](images/fare_distribution.png)

---

## 🔄 Dataset Transformation

| Stage | Rows | Columns |
|-------|------|---------|
| Raw Data | 891 | 12 |
| Cleaned Data | 712 | 9 |

---

## 🔍 Key Insights

- Female passengers had significantly higher survival rates.
- Higher fare passengers had better survival probability.
- Majority of passengers were between ages 20–40.

---

## 🤖 Model Preview

Logistic Regression Accuracy: **~80%+**

---

## 📁 Project Structure


