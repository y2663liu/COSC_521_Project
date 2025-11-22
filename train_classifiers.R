library(caret)
library(dplyr)
library(e1071)
library(randomForest)
library(nnet)

# ---------------------- Config ----------------------
DATA_FILE <- "gesture_features_rewritten.csv"
SEED <- 123
DO_FEATURE_SELECTION <- TRUE 

# ---------------------- Load & Prep ----------------------
if (!file.exists(DATA_FILE)) stop("Run extract_features.R first!")

df <- read.csv(DATA_FILE)
df <- df %>% filter(duration > 0)
df$gesture <- as.factor(df$gesture)

# --- Normalization & NZV Fix ---
feature_cols <- setdiff(names(df), c("participant", "interval", "gesture"))
preproc <- preProcess(df[, feature_cols], method = c("center", "scale", "nzv")) 
df_norm <- predict(preproc, df)
feature_cols <- setdiff(names(df_norm), c("participant", "interval", "gesture"))

# ---------------------- Feature Selection (RFE) ----------------------
final_features <- feature_cols 

if (DO_FEATURE_SELECTION) {
  cat("\n=== Running Recursive Feature Elimination (RFE) ===\n")
  set.seed(SEED)
  
  # Using 5-fold CV for selection to save time/data
  rfe_ctrl <- rfeControl(functions = rfFuncs, method = "cv", number = 5)
  
  # Test smaller subsets suitable for 1000 samples
  subsets <- c(5, 8, 12, length(feature_cols))
  subsets <- subsets[subsets <= length(feature_cols)]
  
  rfe_results <- rfe(
    x = df_norm[, feature_cols], 
    y = df_norm$gesture,
    sizes = subsets,
    rfeControl = rfe_ctrl
  )
  
  print(rfe_results)
  final_features <- predictors(rfe_results)
  cat(sprintf("\nSelected %d features: %s\n", length(final_features), paste(final_features, collapse=", ")))
}

# ---------------------- Final Training Data ----------------------
data_model <- df_norm[, c("gesture", final_features)]

# ---------------------- Training Setup ----------------------
set.seed(SEED)
# 10-Fold CV is standard and safe for N=1000
ctrl <- trainControl(method = "cv", number = 10, savePredictions = "final", classProbs = TRUE)

results <- list()
msg <- function(...) cat(sprintf(...), "\n")

msg("\n=== Training Models on Selected Features ===")

# 1. Logistic Regression (Safe for small data)
msg("Training Logistic Regression...")
results$Logistic <- train(gesture ~ ., data = data_model, method = "multinom", trControl = ctrl, trace = FALSE)

# 2. SVM Radial (Safe for small data)
msg("Training SVM (Radial)...")
results$SVM <- train(gesture ~ ., data = data_model, method = "svmRadial", trControl = ctrl, tuneLength = 5)

# 3. Random Forest (Robust against overfitting)
msg("Training Random Forest...")
results$RF <- train(gesture ~ ., data = data_model, method = "rf", trControl = ctrl, tuneLength = 3, ntree = 100)

# 4. MLP (Adjusted for N=1051)
msg("Training MLP...")
# ADJUSTMENT: Smaller sizes (3, 5, 8) and higher decay (0.1, 0.5)
# This forces the model to be simpler.
grid_mlp <- expand.grid(size = c(3, 5, 8), decay = c(0.1, 0.5))

results$MLP <- train(
  gesture ~ ., data = data_model,
  method = "nnet",
  trControl = ctrl,
  tuneGrid = grid_mlp,
  trace = FALSE,
  linout = FALSE,
  maxit = 200 # Limit iterations to prevent memorization
)

# ---------------------- Evaluation ----------------------
msg("\n=== Final Model Comparison ===")
resamps <- resamples(results)
print(summary(resamps))

for (name in names(results)) {
  model <- results[[name]]
  msg("\n--- %s ---", name)
  
  # Best Tune Accuracy
  best_acc <- max(model$results$Accuracy)
  msg("Best CV Accuracy: %.2f%%", best_acc * 100)
  
  final_preds <- predict(model, data_model)
  cm <- confusionMatrix(final_preds, data_model$gesture)
  print(cm$table)
  
  f1 <- cm$byClass[, "F1"]
  f1[is.na(f1)] <- 0 
  msg("Macro F1: %.4f", mean(f1))
}