library(tseries)
library(forecast)


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


run_pipeline <- function(train_path, validation_path, test_path) {
    train <- read_csv(train_path)
    validation <- read_csv(validation_path)
    test <- read_csv(test_path)

    full_train <- rbind(train_data, validation_data)

    sarimax <- auto.arima(
        full_train$Close,
        xreg = setdiff(colnames(full_train), c("Time", "Close")),
        seasonal = TRUE
    )
    print(summary(sarimax))

    predictions <- forecast(sarimax, h=length(test))
    
    errors <- error(predictions, test$Close)

    return errors
}


## Vietcombank
vcb_results <- run_pipeline(
    "./dataset/official_dataset/vcb/vcb_train.csv",
    "./dataset/official_dataset/vcb/vcb_validation.csv",
    "./dataset/official_dataset/vcb/vcb_test.csv"
)

## BIDV
bid_results <- run_pipeline(
    "./dataset/official_dataset/bid/bid_train.csv",
    "./dataset/official_dataset/bid/bid_validation.csv",
    "./dataset/official_dataset/bid/bid_test.csv"
)

## Techcombank
tcb_results <- run_pipeline(
    "./dataset/official_dataset/tcb/tcb_train.csv",
    "./dataset/official_dataset/tcb/tcb_validation.csv",
    "./dataset/official_dataset/tcb/tcb_test.csv"
)
