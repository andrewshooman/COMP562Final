library(nnet)
library(caret)
library(ggplot2)
library(ROCR)
library(pROC)
library(Hmisc)
library(naniar)
library(dplyr)
library(mice)
library(tidyverse)
library(ggpubr)

# Read data
data = read.csv("C:/Users/ameer/Documents/Kaggle/stroke-prediction/archive/healthcare-dataset-stroke-data.csv")
#data$smoking_status[data$smoking_status == "Unknown"] = NA
data$bmi[data$bmi == "N/A"] = NA
#data$missing_bmi = as.factor(is.na(data$bmi))
data$gender[data$gender == "Other"] = "Female"
data$gender = as.factor(data$gender)
data$age = as.numeric(data$age)
data$hypertension = as.factor(data$hypertension)
data$heart_disease = as.factor(data$heart_disease)
data$ever_married = as.factor(data$ever_married)
data$work_type = as.factor(data$work_type)
data$Residence_type = as.factor(data$Residence_type)
data$bmi = as.numeric(data$bmi)
data$smoking_status = as.factor(data$smoking_status)
data$stroke = as.factor(data$stroke)
levels(data$stroke) = c("no", "yes")

# Impute BMI using linear regression
BMIFit = glm(bmi ~ gender + age + hypertension + heart_disease + ever_married + work_type + Residence_type + avg_glucose_level + smoking_status + stroke, data = data)
BMIPredictions = predict(BMIFit, newdata = data)
s = is.na(data$bmi)
data$bmi[s] = BMIPredictions[s]

# Split data into train and test sets.
smp_size = floor(0.8 * nrow(data))
set.seed(1234)
#train_ind <- sample(seq_len(nrow(data)), size = smp_size)
train_ind = createDataPartition(data$stroke, p = 0.8, list = FALSE)
training = data[train_ind, ]
test = data[-train_ind, ]

# Calculates the threshold that maximizes the Kappa statistic
bestThreshold = function(probs, trueLabels) {
  t = 0
  maxKappa = -1
  for (i in 0:100) {
    pred = as.factor(ifelse(probs > i/100, "yes", "no"))
    cmtx = confusionMatrix(pred, trueLabels, positive = "yes")
    kappa = cmtx$overall[["Kappa"]]
    if (kappa > maxKappa) {
      maxKappa = kappa
      t = i/100
    }
  }
  return(t)
}

# Fit models on data using caret

# 5-fold cross validation repeated 3 times for choosing hyperparameters
fitControl <- trainControl(method = "repeatedcv", number = 5, repeats = 3, classProbs = TRUE)

# Logistc Regression
logReg = train(
  form = stroke ~ . - id,
  data = training,
  trControl = fitControl,
  method = "glm",
  family = "binomial",
  metric = "Kappa"
)
logRegT = bestThreshold(predict(logReg, newdata = training, type = "prob")$"yes", training$stroke)

logRegProb = predict(logReg, newdata = test, type = "prob")
logRegPred = as.factor(ifelse(logRegProb$"yes" > logRegT, "yes", "no"))
#logRegPred = predict(logReg, newdata = test)
logRegCmtx = confusionMatrix(logRegPred, test$stroke, positive = "yes")
logRegRoc = roc(response = test$stroke, predictor = logRegProb$"yes")

# Linear Kernel SVM
svm = train(
  form = stroke ~ . - id,
  data = training,
  trControl = fitControl,
  method = "svmLinear",
  metric = "Kappa"
)
svmT = bestThreshold(predict(svm, newdata = training, type = "prob")$"yes", training$stroke)

svmProb = predict(svm, newdata = test, type = "prob")
svmPred = as.factor(ifelse(svmProb$"yes" > svmT, "yes", "no"))
#svmPred = predict(svm, newdata = test)
svmCmtx = confusionMatrix(svmPred, test$stroke, positive = "yes")
svmRoc = roc(response = test$stroke, predictor = svmProb$"yes")

# k-nearest neighbors
knn = train(
  form = stroke ~ . - id,
  data = training,
  trControl = fitControl,
  method = "knn",
  metric = "Kappa"
)
knnT = bestThreshold(predict(knn, newdata = training, type = "prob")$"yes", training$stroke)

knnProb = predict(knn, newdata = test, type = "prob")
knnPred = as.factor(ifelse(knnProb$"yes" > knnT, "yes", "no"))
#knnPred = predict(knn, newdata = test)
knnCmtx = confusionMatrix(knnPred, test$stroke, positive = "yes")
knnRoc = roc(response = test$stroke, predictor = knnProb$"yes")

# Random Forest
rf = train(
  form = stroke ~ . - id,
  data = training,
  trControl = fitControl,
  method = "rf",
  metric = "Kappa"
)
rfT = bestThreshold(predict(rf, newdata = training, type = "prob")$"yes", training$stroke)

rfProb = predict(rf, newdata = test, type = "prob")
rfPred = as.factor(ifelse(rfProb$"yes" > rfT, "yes", "no"))
#rfPred = predict(rf, newdata = test)
rfCmtx = confusionMatrix(rfPred, test$stroke, positive = "yes")
rfRoc = roc(response = test$stroke, predictor = rfProb$"yes")

# Stochastic Gradient Boosting
gbm = train(
  form = stroke ~ . - id,
  data = training,
  trControl = fitControl,
  method = "gbm",
  metric = "Kappa",
  verbose = FALSE
)
gbmT = bestThreshold(predict(gbm, newdata = training, type = "prob")$"yes", training$stroke)

gbmProb = predict(gbm, newdata = test, type = "prob")
gbmPred = as.factor(ifelse(gbmProb$"yes" > gbmT, "yes", "no"))
#gbmPred = predict(gbm, newdata = test)
gbmCmtx = confusionMatrix(gbmPred, test$stroke, positive = "yes")
gbmRoc = roc(response = test$stroke, predictor = gbmProb$"yes")
