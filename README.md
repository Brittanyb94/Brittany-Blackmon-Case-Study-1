---
title: Employee Attrition Prediction Analysis for Frito Lay
author: "Your Name"
Video Presentation: https://www.youtube.com/watch?v=aMsHQjj4yms
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

# Quick summary statistics
summary(data)

# Visualize key factors
ggplot(data, aes(x = MonthlyIncome, fill = Attrition)) +
  geom_histogram(position = "dodge") +
  labs(title = "Monthly Income Distribution by Attrition Status")

# Train-test split
trainIndex <- createDataPartition(data$Attrition, p = 0.8, list = FALSE)
train_data <- data[trainIndex, ]
test_data <- data[-trainIndex, ]

# KNN with k = 11
knn_model <- train(Attrition ~ JobSatisfaction + MonthlyIncome + YearsAtCompany + OverTime + EnvironmentSatisfaction,
                   data = train_data,
                   method = "knn",
                   tuneGrid = expand.grid(k = 11),
                   trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
                   metric = "Sensitivity")

# Prediction and sensitivity calculation
knn_pred <- predict(knn_model, test_data)
confusionMatrix(knn_pred, test_data$Attrition, positive = "Yes")


# GBM model with sensitivity metric
gbm_model <- train(
  Attrition ~ JobSatisfaction + MonthlyIncome + YearsAtCompany + OverTime + EnvironmentSatisfaction,
  data = train_data,
  method = "gbm",
  trControl = trainControl(method = "cv", number = 5, classProbs = TRUE, summaryFunction = twoClassSummary),
  metric = "Sensitivity",
  verbose = FALSE
)

# Prediction and sensitivity calculation
gbm_pred <- predict(gbm_model, test_data)
confusionMatrix(gbm_pred, test_data$Attrition, positive = "Yes")

