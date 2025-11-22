library(caret)
library(dplyr)
library(e1071)
library(randomForest)
library(nnet)

# ---------------------- Config ----------------------
DATA_FILE <- "gesture_features.csv"
SEED <- 123

# ---------------------- Load & Prep ----------------------
if (!file.exists(DATA_FILE)) stop("Run extract_features.R first!")

df <- read.csv(DATA_FILE)

# 1. Filter out bad rows (e.g. empty graphs with 0 nodes/edges if any)
# Keep only rows where duration > 0
df <- df %>% filter(duration > 0)

# 2. Encode Target
df$gesture <- as.factor(df$gesture)

# 3. Select Feature Columns (Drop metadata)
# We exclude 'participant' and 'interval' to prevent overfitting to specific users
features <- df %>% select(-participant, -interval, -gesture)

# 4. Normalize Features (Center & Scale)
# Important for SVM and MLP
preproc <- preProcess(features, method = c("center", "scale"))
features_norm <- predict(preproc, features)

# Combine back
data_model <- cbind(gesture = df$gesture, features_norm)

# ---------------------- Training Setup ----------------------
set.seed(SEED)

# 10-Fold Cross Validation
ctrl <- trainControl(
  method = "cv", 
  number = 10, 
  savePredictions = "final",
  classProbs = TRUE  # Required for some metrics
)

msg <- function(...) cat(sprintf(...), "\n")

results <- list()

# ---------------------- 1. Logistic Regression ----------------------
msg("Training Multinomial Logistic Regression...")
# 'multinom' from nnet package handles multi-class logistic regression
fit_log <- train(
  gesture ~ ., data = data_model,
  method = "multinom",
  trControl = ctrl,
  trace = FALSE
)
results$Logistic <- fit_log

# ---------------------- 2. SVM (Radial Basis) ----------------------
msg("Training SVM (Radial)...")
fit_svm <- train(
  gesture ~ ., data = data_model,
  method = "svmRadial",
  trControl = ctrl,
  tuneLength = 5
)
results$SVM <- fit_svm

# ---------------------- 3. Random Forest ----------------------
msg("Training Random Forest...")
fit_rf <- train(
  gesture ~ ., data = data_model,
  method = "rf",
  trControl = ctrl,
  tuneLength = 3,
  ntree = 100
)
results$RF <- fit_rf

# ---------------------- 4. MLP (Neural Net) ----------------------
msg("Training MLP...")
# simple single-hidden-layer MLP from nnet
# size = number of hidden units, decay = regularization
grid_mlp <- expand.grid(size = c(5, 10, 20), decay = c(0.1, 0.01))

fit_mlp <- train(
  gesture ~ ., data = data_model,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = grid_mlp,
  trace = FALSE,
  linout = FALSE # FALSE for classification
)
results$MLP <- fit_mlp

# ---------------------- Evaluation ----------------------
msg("\n=== Model Comparison ===")

# Compare Accuracy and Kappa
resamps <- resamples(results)
print(summary(resamps))

# Detailed per-model report
for (name in names(results)) {
  model <- results[[name]]
  
  msg("\n--- %s ---", name)
  
  # Best Tune Accuracy
  best_acc <- max(model$results$Accuracy)
  msg("Best CV Accuracy: %.2f%%", best_acc * 100)
  
  # Confusion Matrix (on the CV predictions)
  preds <- model$pred
  # Filter for the best tuning parameter
  # (caret stores all predictions during tuning, we only want the best ones)
  if (!is.null(model$bestTune)) {
    # merging logic to find rows matching best params
    # simplified: just use predict on training set for a quick matrix 
    # (or extract properly from 'preds' - let's use predict() for clarity on full set)
    final_preds <- predict(model, data_model)
    cm <- confusionMatrix(final_preds, data_model$gesture)
    
    print(cm$table)
    msg("Macro F1: %.4f", mean(cm$byClass[, "F1"], na.rm=TRUE))
  }
}

# Feature Importance (Random Forest)
msg("\n--- Random Forest Feature Importance ---")
varImp_rf <- varImp(fit_rf)
print(varImp_rf)