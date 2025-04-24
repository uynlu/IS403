library(dplyr)
library(readr)
library(zoo)
library(caret)
library(randomForest)
library(e1071)
library(glmnet)
library(jsonlite)
library(kernlab)
library(elasticnet)

# Hàm xử lý dữ liệu
load_and_prepare_data <- function(path) {
    df <- read_csv(path)
    
    # Kiểm tra cột Close
    if (!is.numeric(df$Close)) {
        stop("Cột 'Close' chứa giá trị không phải số")
    }
    
    df$Time <- as.POSIXct(df$Time)
    df <- df %>% 
        mutate(
            Close_lag1 = lag(Close, 1),
            MA5 = rollmean(Close, k = 5, fill = NA, align = "right"),
            Return_1day = (Close / lag(Close)) - 1
        ) %>% 
        na.omit()
    
    X <- df %>% select(-Close, -Time)
    y <- df$Close
    
    # Kiểm tra dữ liệu X và y
    if (!all(sapply(X, is.numeric))) {
        stop("X chứa cột không phải số")
    }
    if (!is.numeric(y)) {
        stop("y không phải số")
    }
    
    # Kiểm tra và xử lý NA
    if (any(is.na(X)) || any(is.na(y))) {
        cat("Warning: Removing remaining NA values\n")
        complete_cases <- complete.cases(X, y)
        X <- X[complete_cases, ]
        y <- y[complete_cases]
    }
    
    return(list(X = X, y = y))
}

# Hàm chuẩn hóa dữ liệu
scale_data <- function(X_train, X_test) {
    preproc <- preProcess(X_train, method = c("center", "scale", "zv", "nzv"))
    return(list(
        X_train_scaled = predict(preproc, X_train),
        X_test_scaled = predict(preproc, X_test)
    ))
}

# Định nghĩa cấu hình mô hình
models <- list(
    random_forest = list(
        model = "rf",
        use_scaled = FALSE,
        params = expand.grid(.mtry = c(2, 3))
    ),
    svm = list(
        model = "svmRadial",
        use_scaled = TRUE,
        params = expand.grid(C = c(0.1, 1, 10), sigma = 0.1)
    ),
    linear_regression = list(
        model = "lm",
        use_scaled = TRUE,
        params = NULL
    ),
    ridge = list(
        model = "glmnet",
        use_scaled = TRUE,
        params = expand.grid(alpha = 0, lambda = c(0.1, 1, 10))
    )
)

# Hàm huấn luyện mô hình
train_models <- function(X_train, X_train_scaled, y_train) {
    trained_models <- list()
    
    for (name in names(models)) {
        model_conf <- models[[name]]
        X_tr <- if (model_conf$use_scaled) X_train_scaled else X_train
        
        # Cross-validation
        trctrl <- trainControl(
            method = "timeslice",
            initialWindow = floor(0.7 * nrow(X_tr)),
            horizon = 5,
            fixedWindow = TRUE,
            allowParallel = TRUE
        )
        
        tryCatch({
            model_fit <- train(
                x = X_tr, 
                y = y_train,
                method = model_conf$model,
                tuneGrid = model_conf$params,
                trControl = trctrl,
                metric = "RMSE"
            )
          
            trained_models[[name]] <- model_fit
            cat("Model fitted successfully for", name, "\n")
        }, error = function(e) {
            cat("Error fitting model", name, ":", e$message, "\n")
            trained_models[[name]] <- NULL
        })
    }
    
    return(trained_models)
}

# Hàm đánh giá
error <- function(pred, true) {
    if (any(is.na(pred)) || any(is.na(true)) || any(is.infinite(pred)) || any(is.infinite(true))) {
        cat("Warning: NA or Inf values in predictions or true values\n")
        return(list(mse = NA, rmse = NA, mae = NA, mape = NA))
    }
    
    mse <- mean((pred - true)^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    mae <- mean(abs(pred - true), na.rm = TRUE)
    mape <- mean(abs((pred - true) / true), na.rm = TRUE) * 100

    return(list(mse = mse, rmse = rmse, mae = mae, mape = mape))
}

# Hàm đánh giá mô hình
evaluate_models <- function(trained_models, X_test, X_test_scaled, y_test) {
    results <- list()
    
    for (name in names(trained_models)) {
        model_conf <- models[[name]]
        X_t <- if (model_conf$use_scaled) X_test_scaled else X_test
        
        start_time <- Sys.time()
        pred <- predict(trained_models[[name]], X_t)
        
        # Tính toán lỗi
        errors <- error(pred, y_test)
        
        results[[name]] <- list(
            predictions = pred,
            errors = errors,
            time = as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        )
    }
    
    return(results)
}

# Hàm chạy pipeline chính
run_pipeline <- function(train_path, validation_path, test_path) {
    # Load data
    train_data <- load_and_prepare_data(train_path)
    validation_data <- load_and_prepare_data(validation_path)
    test_data <- load_and_prepare_data(test_path)
    
    # Gộp tập train và tập validation
    combined_X <- rbind(train_data$X, validation_data$X)
    combined_y <- c(train_data$y, validation_data$y)
    
    # Chuẩn hóa dữ liệu
    scaled_data <- scale_data(combined_X, test_data$X)
    
    # Huấn luyện mô hình
    trained_models <- train_models(
        combined_X,
        scaled_data$X_train_scaled,
        combined_y
    )
    
    # Đánh giá kết quả mô hình trên tập test
    evaluation_results <- evaluate_models(
        trained_models,
        test_data$X,
        scaled_data$X_test_scaled,
        test_data$y
    )
    
    return(evaluation_results)
}

# Hàm in kết quả
print_results <- function(results, stock_name) {
    cat("\n", paste0(stock_name, " Results:"), "\n")
    for (model in names(results)) {
        cat("\nModel:", model, "\n")
        errors <- results[[model]]$errors
        if (is.list(errors) && all(sapply(errors, is.numeric))) {
            errors_vec <- unlist(errors)
            print(round(errors_vec, 4))
        } else {
            cat("Không thể in lỗi do dữ liệu không phải số hoặc chứa NA/Inf\n")
            print(errors)  # In nguyên bản để kiểm tra
        }
    }
}

# Chạy pipeline
## Vietcombank
vcb_results <- run_pipeline(
    "./dataset/official_dataset/vcb/vcb_train.csv",
    "./dataset/official_dataset/vcb/vcb_validation.csv",
    "./dataset/official_dataset/vcb/vcb_test.csv"
)
print_results(vcb_results, "VCB")  

## BIDV
bid_results <- run_pipeline(
    "./dataset/official_dataset/bid/bid_train.csv",
    "./dataset/official_dataset/bid/bid_validation.csv",
    "./dataset/official_dataset/bid/bid_test.csv"
)
print_results(bid_results, "BID")  

## Techcombank
tcb_results <- run_pipeline(
    "./dataset/official_dataset/tcb/tcb_train.csv",
    "./dataset/official_dataset/tcb/tcb_validation.csv",
    "./dataset/official_dataset/tcb/tcb_test.csv"
)
print_results(tcb_results, "TCB")  

warnings()
