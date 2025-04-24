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

# Định nghĩa các tham số chung
N_STEPS <- 4

# Hàm xử lý dữ liệu
load_and_prepare_data <- function(path) {
    df <- read_csv(path)
    df$Time <- as.POSIXct(df$Time)
    df <- df %>%
        mutate(
            Close_lag1 = lag(Close, 1),
            MA5 = rollmean(Close, k = 5, fill = NA, align = "right"),
            Return_1day = (Close / lag(Close)) - 1
        ) %>%
        na.omit()
    
    X <- df %>% select(-Close, -Time)
    y <- df %>% select(Close)
    
    # Kiểm tra và xử lý NA
    if (any(is.na(X)) || any(is.na(y))) {
        cat("Warning: Removing remaining NA values\n")
        complete_cases <- complete.cases(X, y)
        X <- X[complete_cases, ]
        y <- y[complete_cases, ]
    }
    
    return(list(X = X, y = y))
}

# Hàm tạo dữ liệu multi-step
prepare_multi_step_data <- function(X, y, n_steps = N_STEPS) {
    y_multi <- data.frame(matrix(NA, nrow = nrow(y), ncol = n_steps))
    colnames(y_multi) <- paste0("step_", 1:n_steps)
    
    for (i in 1:n_steps) {
        y_multi[[i]] <- dplyr::lead(y$Close, i - 1)
    }
    
    valid_rows <- complete.cases(y_multi)
    return(list(X = X[valid_rows, ], y = y_multi[valid_rows, ]))
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
train_models <- function(X_train, X_train_scaled, y_train, n_steps) {
    trained_models <- list()
    
    for (name in names(models)) {
        model_conf <- models[[name]]
        X_tr <- if (model_conf$use_scaled) X_train_scaled else X_train
        
        step_models <- list()
      
        for (step in 1:n_steps) {
            y_step <- y_train[[step]]
          
            # Kiểm tra dữ liệu
            if (any(is.na(y_step))) {
                cat("Warning: NA values in y_step for", name, "step", step, "\n")
                y_step[is.na(y_step)] <- mean(y_step, na.rm = TRUE)
            }
            
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
                    y = y_step,
                    method = model_conf$model,
                    tuneGrid = model_conf$params,
                    trControl = trctrl,
                    metric = "RMSE"
                )
                
                step_models[[step]] <- model_fit
                cat("Model fitted successfully for", name, "at step", step, "\n")
            }, error = function(e) {
                cat("Error fitting model", name, "at step", step, ":", e$message, "\n")
                step_models[[step]] <- NULL
            })
        }

      trained_models[[name]] <- step_models
    }
    
    return(trained_models)
}

# Hàm đánh giá
error <- function(pred, true) {
    if (any(is.na(pred)) || any(is.na(true))) {
        cat("Warning: NA values in predictions or true values\n")
        return(list(mse = NA, rmse = NA, mae = NA, mape = NA))
    }
    
    mse <- mean((pred - true)^2, na.rm = TRUE)
    rmse <- sqrt(mse)
    mae <- mean(abs(pred - true), na.rm = TRUE)
    mape <- mean(abs((pred - true) / true), na.rm = TRUE) * 100
    
    return(list(mse = mse, rmse = rmse, mae = mae, mape = mape))
}

# Hàm đánh giá mô hình
evaluate_models <- function(trained_models, X_test, X_test_scaled, y_test, n_steps) {
    results <- list()
    
    for (name in names(trained_models)) {
        model_conf <- models[[name]]
        X_t <- if (model_conf$use_scaled) X_test_scaled else X_test
        
        step_models <- trained_models[[name]]
        predictions <- matrix(NA, nrow = nrow(X_t), ncol = n_steps)
        start_time <- Sys.time()
        
        for (step in 1:n_steps) {
            if (!is.null(step_models[[step]])) {
              pred <- predict(step_models[[step]], X_t)
              predictions[, step] <- ifelse(is.na(pred), mean(y_test[[step]], na.rm = TRUE), pred)
            }
        }
        
        errors <- list(
            MSE = mean((predictions - as.matrix(y_test))^2),
            RMSE = sqrt(mean((predictions - as.matrix(y_test))^2)),
            MAE = mean(abs(predictions - as.matrix(y_test))),
            MAPE = mean(abs((predictions - as.matrix(y_test))/as.matrix(y_test)), na.rm = TRUE) * 100
        )
        
        results[[name]] <- list(
            predictions = predictions,
            errors = errors,
            time = as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        )
    }
    
    return(results)
}

# Hàm chạy pipeline chính
run_pipeline <- function(train_path, val_path, test_path) {
    # Load data
    train_data <- load_and_prepare_data(train_path)
    val_data <- load_and_prepare_data(val_path)
    test_data <- load_and_prepare_data(test_path)
    
    # Gộp tập train và tập validation
    combined_X <- rbind(train_data$X, val_data$X)
    combined_y <- rbind(train_data$y, val_data$y)
    
    # Tạo dữ liệu cho multi-step
    train_prep <- prepare_multi_step_data(combined_X, combined_y)
    test_prep <- prepare_multi_step_data(test_data$X, test_data$y)
    
    # Chuẩn hóa dữ liệu
    scaled_data <- scale_data(train_prep$X, test_prep$X)
    
    # Huấn luyện mô hình
    trained_models <- train_models(
        train_prep$X,
        scaled_data$X_train_scaled,
        train_prep$y,
        N_STEPS
    )
    
    # Đánh giá kết quả mô hình trên tập test
    evaluation_results <- evaluate_models(
        trained_models,
        test_prep$X,
        scaled_data$X_test_scaled,
        test_prep$y,
        N_STEPS
    )
    
    return(evaluation_results)
}

# Hàm in kết quả
print_results <- function(results, stock_name) {
    cat("\n", paste0(stock_name, " Results:"), "\n")
    for (model in names(results)) {
        cat("\nModel:", model, "\n")
        print(round(results[[model]]$errors, 4))
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
