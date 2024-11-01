---
title: "Employee Attrition Prediction Analysis"
author: "Your Name"
date: "`r Sys.Date()`"
output: 
  html_document:
    toc: true
    toc_depth: 2
---

# Executive Summary
This analysis aims to predict employee attrition by leveraging machine learning models on various employee features, such as job satisfaction, monthly income, and overtime status. The results are used to propose actionable strategies for retention, particularly targeting high-risk employee groups.

# Introduction
Employee attrition is a key challenge for companies aiming to retain talent and minimize costs associated with turnover. This project leverages multiple machine learning models, including K-Nearest Neighbors (KNN) and Gradient Boosting Machines (GBM), to classify employees at risk of leaving. The analysis compares model performance, assesses classification accuracy and sensitivity, and makes strategic recommendations for retention.

## Project Objectives
1. Develop a predictive model to identify employees likely to leave.
2. Analyze model sensitivity and accuracy for different classifiers.
3. Propose actionable recommendations based on findings.

# Data Preparation and Exploratory Analysis
```{r setup, include=FALSE}
# Load libraries
library(caret)
library(ggplot2)
library(DMwR) # For SMOTE if used
library(gbm)
library(randomForest)

# Set seed for reproducibility
set.seed(123)

# Load dataset
data <- read.csv("your_data.csv")

# Convert categorical variables
data$Attrition <- as.factor(data$Attrition)
str(data)

