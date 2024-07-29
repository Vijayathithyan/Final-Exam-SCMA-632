# Load necessary libraries
library(tidyverse)  # For data manipulation
library(caret)      # For machine learning utilities
library(rpart)      # For decision tree model
library(rpart.plot) # For plotting decision trees
library(pROC)       # For ROC curves and AUC calculation

setwd("D:\\MDA\\Course\\Boot Camp\\SCMA 632\\Final Exam")

# Load the dataset
bank_data <- read.csv("bank-additional-full.csv", sep = ";")

# Check the first few rows of the dataset
head(bank_data)

# Check for missing values
sum(is.na(bank_data))

# Convert categorical variables to factors
bank_data <- bank_data %>% mutate_if(is.character, as.factor)

# Inspect the data structure
str(bank_data)

# Set seed for reproducibility
set.seed(123)

# Split the data into training and testing sets
train_index <- createDataPartition(bank_data$y, p = 0.75, list = FALSE)
train_data <- bank_data[train_index, ]
test_data <- bank_data[-train_index, ]

# Fit the logistic regression model
logistic_model <- glm(y ~ ., data = train_data, family = binomial)

# Predict on the test set using Logistic Regression
logistic_pred <- predict(logistic_model, test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, "yes", "no")

# Evaluate Logistic Regression Model

# Confusion Matrix
confusion_matrix_logistic <- confusionMatrix(as.factor(logistic_pred_class), test_data$y)
print(confusion_matrix_logistic)

# Calculate metrics
accuracy_logistic <- confusion_matrix_logistic$overall["Accuracy"]
precision_logistic <- confusion_matrix_logistic$byClass["Pos Pred Value"]
recall_logistic <- confusion_matrix_logistic$byClass["Sensitivity"]
f1_score_logistic <- 2 * ((precision_logistic * recall_logistic) / (precision_logistic + recall_logistic))

# AUC-ROC
roc_logistic <- roc(test_data$y, logistic_pred)
auc_logistic <- auc(roc_logistic)

# Print metrics for Logistic Regression
cat("Logistic Regression Metrics:\n")
cat("Accuracy:", accuracy_logistic, "\n")
cat("Precision:", precision_logistic, "\n")
cat("Recall:", recall_logistic, "\n")
cat("F1 Score:", f1_score_logistic, "\n")
cat("AUC:", auc_logistic, "\n\n")

# Fit the decision tree model
tree_model <- rpart(y ~ ., data = train_data, method = "class")

# Predict on the test set using Decision Tree
tree_pred <- predict(tree_model, test_data, type = "class")

# Evaluate Decision Tree Model

# Confusion Matrix
confusion_matrix_tree <- confusionMatrix(tree_pred, test_data$y)
print(confusion_matrix_tree)

# Calculate metrics
accuracy_tree <- confusion_matrix_tree$overall["Accuracy"]
precision_tree <- confusion_matrix_tree$byClass["Pos Pred Value"]
recall_tree <- confusion_matrix_tree$byClass["Sensitivity"]
f1_score_tree <- 2 * ((precision_tree * recall_tree) / (precision_tree + recall_tree))

# AUC-ROC for Decision Tree
tree_pred_prob <- predict(tree_model, test_data, type = "prob")[, 2]
roc_tree <- roc(test_data$y, tree_pred_prob)
auc_tree <- auc(roc_tree)

# Print metrics for Decision Tree
cat("Decision Tree Metrics:\n")
cat("Accuracy:", accuracy_tree, "\n")
cat("Precision:", precision_tree, "\n")
cat("Recall:", recall_tree, "\n")
cat("F1 Score:", f1_score_tree, "\n")
cat("AUC:", auc_tree, "\n\n")

# Visualization

# Confusion Matrices
print(confusion_matrix_logistic)
print(confusion_matrix_tree)

# Plot AUC-ROC Curves
plot(roc_logistic, col = "blue", main = "AUC-ROC Curves", legacy.axes = TRUE)
plot(roc_tree, col = "red", add = TRUE)
legend("bottomright", legend = c("Logistic Regression", "Decision Tree"), col = c("blue", "red"), lwd = 2)

# Plot the decision tree
rpart.plot(tree_model, main = "Decision Tree Structure")

# Interpretation of Results

# Logistic Regression Coefficients
cat("Logistic Regression Coefficients:\n")
print(summary(logistic_model))
cat("Odds Ratios:\n")
print(exp(coef(logistic_model)))

# Decision Tree Structure
cat("Decision Tree Structure:\n")
print(tree_model)
cat("Variable Importance:\n")
print(varImp(tree_model))

# Interpretations - 
Logistic Regression Model

Model Fitting and Prediction:
  
  A logistic regression model was fitted using the training data.
Predictions were made on the test set, and the results were converted to binary outcomes based on a threshold of 0.5.

Model Evaluation:
  
  The confusion matrix for the logistic regression model indicated an accuracy of 0.9086.
The model exhibited a sensitivity of 0.9730 and a specificity of 0.4017.
Precision, recall, and F1 score were calculated as 0.9276, 0.9729671, and 0.9497356, respectively.
The AUC (Area Under the ROC Curve) was 0.9336.

Logistic Regression Coefficients:
  
  The summary of the logistic regression model provided the coefficients, standard errors, z-values, and p-values.

Key findings included significant positive coefficients for variables like "jobretired," "educationuniversity.degree," "poutcomesuccess," and "duration," indicating these factors positively influence the likelihood of a successful outcome.
Negative coefficients were observed for "jobblue-collar," "defaultunknown," "contacttelephone," and "emp.var.rate," suggesting these variables decrease the likelihood of a successful outcome.

Decision Tree Model

Model Fitting and Prediction:
  
  A decision tree model was fitted using the training data.
Predictions were made on the test set.

Model Evaluation:
  
  The confusion matrix for the decision tree model indicated an accuracy of 0.9108.
The model exhibited a sensitivity of 0.9629 and a specificity of 0.5009.
Precision, recall, and F1 score were calculated as 0.9383, 0.9628981, and 0.9504159, respectively.
The AUC (Area Under the ROC Curve) was 0.8485.

Comparison of Models

Both models achieved similar accuracy, with the decision tree model slightly outperforming the logistic regression model (0.9108 vs. 0.9086).
The logistic regression model had a higher AUC (0.9336 vs. 0.8485), indicating better overall performance in distinguishing between classes.
Both models showed high sensitivity, with the logistic regression model having a slightly higher sensitivity (0.9730 vs. 0.9629).
The decision tree model had a higher specificity (0.5009 vs. 0.4017).

Visualizations

AUC-ROC curves were plotted for both models, visually comparing their performance.
The structure of the decision tree was also visualized to understand the decision rules derived by the model.

Conclusion

The logistic regression model and decision tree model both performed well, with the logistic regression model demonstrating slightly better discrimination ability as indicated by a higher AUC.
The decision tree model provided better interpretability and slightly higher accuracy.
Depending on the specific application needs (e.g., ease of interpretation vs. model performance), either model could be considered appropriate for predicting the target variable in this dataset.

