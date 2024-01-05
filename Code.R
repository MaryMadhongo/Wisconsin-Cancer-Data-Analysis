# Load libraries
library(caret)
library(dplyr)
library(tidyr)
library(ggplot2)
library(randomForest)
library(e1071)
library(nnet)
library(pROC)
library(PRROC)
library(mice)
# Load the Wisconsin Breast Cancer data set
load_data <- function(file_path) {
  data <- read.csv(file_path, header = TRUE, stringsAsFactors = TRUE)
  return(data)
}
data <- load_data("wisconsin.csv")
# Function for basic data exploration
explore_data <- function(data) {
  str(data)
  head(data)
  summary(data)
  sapply(data[sapply(data, is.numeric)], sd, na.rm=TRUE)
}
explore_data(data)
# Function for data pre-processing
preprocess_data <- function(data) {
  # Create a copy of  data frame
  imputed_data <- data
  # Calculate the median of the Bare.nuclei variable excluding NA values
  median_value <- median(imputed_data$Bare.nuclei, na.rm = TRUE)
  # Replace NA values in Bare.nuclei with the median
  imputed_data$Bare.nuclei <- ifelse(is.na(imputed_data$Bare.nuclei),
                                     median_value, imputed_data$Bare.nuclei)
  return(imputed_data)
}
imputed_data <- preprocess_data(data)
summary(imputed_data$Bare.nuclei)
# Function to identify outliers in all numeric columns
identify_outliers <- function(imputed_data){
  outlier_list <- list()
  for (col in names(imputed_data)){
    if(is.numeric(imputed_data[[col]])) {
      # Calculate the IQR of the column
      IQR <- IQR(imputed_data[[col]], na.rm = TRUE)
      # Calculate the lower and upper bound
      lower_bound <- quantile(imputed_data[[col]], 0.25, na.rm = TRUE) - 1.5 * IQR
      upper_bound <- quantile(imputed_data[[col]], 0.75, na.rm = TRUE) + 1.5 * IQR
      # Identify the outliers
      outliers <- imputed_data[[col]][data[[col]] < lower_bound | data[[col]] > upper_bound]
      outlier_list[[col]] <- outliers
    }
  }
  return(outlier_list)
}
# Call the function
outliers <- identify_outliers(imputed_data)
# Create an empty list to store the outliers for each feature
outliers <- list()
# Loop over each feature in the imputed data
for (feature in colnames(imputed_data)) {
  # Check if the feature is numeric
  if (is.numeric(imputed_data[[feature]])) {
    # Calculate the IQR for the feature
    Q1 <- quantile(imputed_data[[feature]], 0.25, na.rm = TRUE)
    Q3 <- quantile(imputed_data[[feature]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    # Identify outliers
    outliers[[feature]] <- imputed_data[[feature]] < (Q1 - 1.5 * IQR) | imputed_data[[feature]] > (Q3 + 1.5 * IQR)
  }
}
# Print the number of outliers for each feature
sapply(outliers, sum)
# Function to normalize numerical features using min-max scaling
normalize_data <- function(imputed_data) {
  par(mar = c(1, 1, 1, 1))
  # Convert back to data frame
  imputed_data <- as.data.frame(imputed_data)  
  for (col in names(imputed_data)) {
    if (is.numeric(imputed_data[[col]])) {
      imputed_data[[col]][is.na(imputed_data[[col]])] <- mean(imputed_data[[col]],
                                                              na.rm = TRUE)
      imputed_data[[col]] <- (imputed_data[[col]] - min(imputed_data[[col]], 
                                                        na.rm = TRUE)) / (max(imputed_data[[col]], na.rm = TRUE) -
                                                                            min(imputed_data[[col]], na.rm = TRUE))
    }
  }
  
  return(imputed_data)
}
imputed_data <- normalize_data(imputed_data)
# Function to reshape data to long format
reshape_data <- function(imputed_data) {
  data_long <- tidyr::pivot_longer(imputed_data, -Class, names_to = "Variable",
                                   values_to = "Value")
  return(data_long)
}
data_long <- reshape_data(imputed_data)
# Function to create boxplot
create_boxplot <- function(data_long) {
  par(mar = c(1, 1, 1, 1))
  ggplot(data_long, aes(x = Class, y = Value, fill = Class)) +
    geom_boxplot() +
    facet_wrap(~ Variable, scales = "free") +
    theme_minimal() +
    labs(x = "Class", y = "Value", title = "Boxplots for Each Variable by Class",
         fill = "Class") +
    theme(legend.position = "none")
}
create_boxplot(data_long)
# Function to create histograms for each variable
create_histograms <- function(data_long) {
  par(mar = c(1, 1, 1, 1))
  ggplot(data_long, aes(x = Value)) +
    geom_histogram(binwidth = 0.1, fill = 'blue',color = 'black') +
    facet_wrap(~ Variable, scales = "free") +
    ggtitle("Histograms of Variables")
}
create_histograms(data_long)
# Function to create scatter plots for first 3 variables
create_scatter_plots <- function(imputed_data) {
  par(mar = c(1, 1, 1, 1))
  pairs(imputed_data[, 1:3])
}
create_scatter_plots(imputed_data)
# Function to create bar plot
create_bar_plot <- function(imputed_data) {
  par(mar = c(1, 1, 1, 1))
  ggplot(imputed_data, aes(x = Class)) +
    geom_bar(fill = 'blue',color = 'black') +
    ggtitle("Bar Plot of Class Variable")
}

create_bar_plot(imputed_data)
# Function to create correlation matrix
create_cor_matrix <- function(imputed_data) {
  cor_matrix <- cor(imputed_data[,1:9])  
  print(cor_matrix)
}
create_cor_matrix(imputed_data)
# Function to perform t-tests for each variable by Class
perform_t_tests <- function(imputed_data) {
  variables <- names(imputed_data)[1:9]
  for (var in variables) {
    print(t.test(as.formula(paste(var, "~ Class")), data=imputed_data))
  }
}
perform_t_tests(imputed_data)
# Function to create the data partition and split the data into training and testing sets
split_data <- function(imputed_data) {
  set.seed(600)
  trainIndex <- createDataPartition(imputed_data$Class, p = 0.7, list = FALSE)
  training_set <- imputed_data[trainIndex, ]
  testing_set <- imputed_data[-trainIndex, ]
  
  return(list(training_set, testing_set))
}
list_sets <- split_data(imputed_data)
training_set <- list_sets[[1]]
testing_set <- list_sets[[2]]
nrow(training_set)
nrow(testing_set)
# Function to apply log transformation
transform_features <- function(dataset) {
  transformed_dataset <- log1p(dataset[, -ncol(dataset)]) 
  transformed_dataset$Class <- dataset$Class 
  return(transformed_dataset)
}
# Apply transformation to the training and testing sets
training_set <- transform_features(training_set)
testing_set <- transform_features(testing_set)
# Apply transformation to the training set
transformed_training_set <- transform_features(training_set)
# Select only numeric columns from the dataset
numeric_columns <- sapply(transformed_training_set, is.numeric)
# Calculate correlation matrix for the numeric columns
correlation_matrix <- cor(transformed_training_set[, numeric_columns])
# Print the correlation matrix
print(correlation_matrix)
# Function for feature selection using recursive feature elimination (RFE)
select_features <- function(training_set) {
  ctrl <- rfeControl(functions=rfFuncs, method="cv", number=10)
  results <- rfe(training_set[, -ncol(training_set)], training_set$Class,
                 sizes=c(1:ncol(training_set)-1), rfeControl=ctrl)
  optimal_features <- predictors(results)
  print(optimal_features)
  training_set <- training_set[, c(optimal_features, "Class")]
  return(list(training_set, optimal_features))
}
list_sets <- select_features(training_set)
training_set <- list_sets[[1]]
optimal_features <- list_sets[[2]]
# Apply the selected features to the testing set
testing_set <- testing_set[, c(optimal_features, "Class")]
# Function to train a model and make predictions
train_model <- function(model, training_set, testing_set) {
  fitControl <- trainControl(method = "cv", number = 10)
  
  # Define the tuning grid based on the model
  if (model == "rf") {
    tuneGrid <- expand.grid(mtry = c(1:(ncol(training_set)-1)))
  } else if (model == "svmRadial") {
    tuneGrid <- expand.grid(sigma = c(0.05, 0.1), C = c(1, 2))
  } else if (model == "nnet") {
    tuneGrid <- expand.grid(size = c(1:5), decay = c(0.1, 0.5, 1))
  } else {
    tuneGrid <- NULL
  }
  fit <- train(Class ~ ., data = training_set, method = model,
               trControl = fitControl, tuneGrid = tuneGrid)
  # Get predictions
  if (model == "svmRadial") {
    predictions <- predict(fit, newdata = testing_set, type = "raw")
  } else {
    predictions <- predict(fit, newdata = testing_set, type = "prob")[,2]
  }
  # Print the length of predictions
  print(length(predictions))
  return(list(fit, predictions))
}
# Train the logistic regression model
list_logistic <- train_model("glm", training_set, testing_set)
logistic_fit <- list_logistic[[1]]
logistic_predictions <- list_logistic[[2]]
# Train the random forest model
list_rf <- train_model("rf", training_set, testing_set)
rf_fit <- list_rf[[1]]
rf_predictions <- list_rf[[2]]
# Train the SVM model
list_svm <- train_model("svmRadial", training_set, testing_set)
svm_fit <- list_svm[[1]]
svm_predictions <- list_svm[[2]]
# Train the neural network model
list_nnet <- train_model("nnet", training_set, testing_set)
nnet_fit <- list_nnet[[1]]
nnet_predictions <- list_nnet[[2]]
# Convert factor predictions to numeric
svm_predictions_numeric <- as.numeric(svm_predictions == "Positive")
# Create a data frame with a column for each set of predictions
df <- data.frame(logistic_predictions = logistic_predictions, 
                 rf_predictions = rf_predictions, 
                 svm_predictions = svm_predictions_numeric,  # use the numeric predictions here
                 nnet_predictions = nnet_predictions)
# Reshape the data frame to a long format
df_long <- reshape2::melt(df)
# Create the plot
ggplot(df_long, aes(x=value, fill=variable)) +
  geom_histogram(position="identity", alpha=0.5, bins=30) +
  theme_minimal() +
  labs(x="Predicted Value", y="Count", fill="Model", 
       title="Histogram of Predictions from Each Model")
# Function to compute the ROC curve for each model
compute_roc <- function(target, predictions) {
  par(mar = c(1, 1, 1, 1))
  roc_obj <- roc(target, predictions)
  return(roc_obj)
}
# Compute the ROC object for each model
roc_obj_logistic <- roc(response = testing_set$Class, predictor = predict(logistic_fit, newdata = testing_set, type = "prob")[,2])
roc_obj_rf <- roc(response = testing_set$Class, predictor = predict(rf_fit, newdata = testing_set, type = "prob")[,2])
roc_obj_svm <- roc(response = testing_set$Class, 
                   predictor = as.numeric(predict(svm_fit, newdata = testing_set, type = "raw")))
roc_obj_nnet <- roc(response = testing_set$Class, predictor = predict(nnet_fit, newdata = testing_set, type = "prob")[,2])

# Plot the ROC curve 
par(mar = c(5,5,1,5))
plot(1, 1, xlim = c(0, 1), ylim = c(0, 1), type = "n",
     xlab = "False Positive Rate", ylab = "True Positive Rate", main = "ROC Curves")
# Add the ROC curves for the models
lines(roc_obj_logistic, col = "blue", lwd = 2)
lines(roc_obj_rf, col = "red", lwd = 2)
lines(roc_obj_svm, col = "green", lwd = 2)
lines(roc_obj_nnet, col = "purple", lwd = 2)
# Add a legend
legend("bottomleft", legend = c("Logistic", "RF", "SVM", "NN"), col = c("blue", "red", "green", "purple"), lwd = 2, cex = 0.8)
xlab("False Positive Rate")
ylab("True Positive Rate")
title(main = "ROC Curves")
grid()
# Function to compute precision, recall, and F1 score
compute_metrics <- function(actual, predictions) {
  # Ensure predictions are numeric
  predictions <- as.numeric(as.character(predictions))
  # Generate labels based on the threshold
  predicted_labels <- ifelse(predictions > 0.5, "malignant", "benign")
  # Convert to factor for the metric functions
  predicted_labels <- as.factor(predicted_labels)
  precision <- posPredValue(actual, predicted_labels, positive="malignant")
  recall <- sensitivity(actual, predicted_labels, positive="malignant")
  F1 <- 2 * (precision * recall) / (precision + recall)
  return(list(precision, recall, F1))
}
# Function to convert factor levels to numeric
convert_levels <- function(predictions) {
  levels(predictions) <- c("benign" = 0, "malignant" = 1)
  return(as.numeric(as.character(predictions)))
}
# Apply function to each model's predictions
logistic_predictions <- convert_levels(logistic_predictions)
rf_predictions <- convert_levels(rf_predictions)
svm_predictions <- convert_levels(svm_predictions)
nnet_predictions <- convert_levels(nnet_predictions)
# Compute metrics for each model
metrics_logistic <- compute_metrics(as.factor(testing_set$Class), logistic_predictions)
metrics_rf <- compute_metrics(as.factor(testing_set$Class), rf_predictions)
metrics_svm <- compute_metrics(as.factor(testing_set$Class), svm_predictions)
metrics_nnet <- compute_metrics(as.factor(testing_set$Class), nnet_predictions)
# Convert the predicted probabilities to class labels
logistic_predictions <- ifelse(logistic_predictions > 0.5, "malignant", "benign")
rf_predictions <- ifelse(rf_predictions > 0.5, "malignant", "benign")
svm_predictions <- ifelse(svm_predictions > 0.5, "malignant", "benign")
nnet_predictions <- ifelse(nnet_predictions > 0.5, "malignant", "benign")
# Convert the predicted class to factor with appropriate levels
logistic_predictions <- factor(logistic_predictions, levels = levels(testing_set$Class))
rf_predictions <- factor(rf_predictions, levels = levels(testing_set$Class))
svm_predictions <- factor(svm_predictions, levels = levels(testing_set$Class))
nnet_predictions <- factor(nnet_predictions, levels = levels(testing_set$Class))
# Compute the confusion matrix for each model
logistic_confusion <- confusionMatrix(logistic_predictions, testing_set$Class, positive = "malignant")
rf_confusion <- confusionMatrix(rf_predictions, testing_set$Class, positive = "malignant")
svm_confusion <- confusionMatrix(svm_predictions, testing_set$Class, positive = "malignant")
nnet_confusion <- confusionMatrix(nnet_predictions, testing_set$Class, positive = "malignant")
# Print the metrics and confusion matrix for each model
print_metrics <- function(model_name, metrics, confusion) {
  print(paste(model_name, "- Precision: ", metrics[[1]], " Recall: ", metrics[[2]], " F1 Score: ", metrics[[3]]))
  print(paste("Confusion Matrix for", model_name, ":"))
  print(confusion)
}

print_metrics("Logistic Regression", metrics_logistic, logistic_confusion)
print_metrics("Random Forest", metrics_rf, rf_confusion)
print_metrics("SVM", metrics_svm, svm_confusion)
print_metrics("Neural Network", metrics_nnet, nnet_confusion)
# Function to calculate and print feature importance or coefficients
print_model_metrics <- function(models) {
  for (model_name in names(models)) {
    model <- models[[model_name]]
    if (class(model$finalModel)[1] == "randomForest") {
      # Calculate feature importance for Random Forest model
      importance <- importance(model$finalModel)
      print(paste("Feature importance for", model_name, ":"))
      print(importance)
    } else if (class(model$finalModel)[1] == "glm") {
      # Print the coefficients of the Logistic Regression model
      coefficients <- coef(model$finalModel)
      print(paste("Coefficients for", model_name, ":"))
      print(coefficients)
    } else {
      print(paste("Feature importance for", model_name, "cannot be directly calculated."))
    }
  }
}
# Create a list of models
models <- list(
  logistic_fit = logistic_fit,
  rf_fit = rf_fit,
  svm_fit = svm_fit,
  nnet_fit = nnet_fit
)
# Call the function with your list of models
print_model_metrics(models)

