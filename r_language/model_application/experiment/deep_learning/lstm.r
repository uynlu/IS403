library(dplyr)
library(keras)
library(caTools)
library(caret)

load_and_prepare_data <- function(path, look_back, n_steps) {
    dataset <- read_csv(path)
  
    feature_cols <- setdiff(colnames(dataset), c("Time", "Close"))

    features_scaler <- preProcess(dataset[, feature_cols], method = "range")
    new_features_dataset <- predict(features_scaler, dataset[, feature_cols])

    close_scaler <- preProcess(dataset[, "Close"], method = "range")
    new_close_dataset <- predict(close_scaler, dataset[, "Close"])

    dataset <- cbind(new_features_dataset, new_close_dataset)

    features <- as.matrix(dataset)
    targets <- as.numeric(dataset$Close)

    X <- list()
    y <- list()
    for (i in 1:(nrow(dataset) - look_back)) {
        feature_window <- features[i:(i + look_back - 1), ]
        target_window <- targets[(i + look_back):(i + look_back + n_steps - 1)]
        
        if (length(target_window) == n_steps) {
            X[[length(X) + 1]] <- feature_window
            y[[length(y) + 1]] <- target_window
        }
    }
    
    X <- array(unlist(X), dim = c(length(X), look_back, ncol(features)))
    y <- array(unlist(y), dim = c(length(y), n_steps))

    return(list(X = X, y = y))
}

LSTM_model <- function(input_size, hidden_size = 50, num_layers = 3, dropout_prob = 0.3, n_steps = 1) {
    model <- keras_model_sequential() %>%
    layer_input(shape = c(NULL, input_size)) %>%
    layer_lstm(units = hidden_size, num_layers = num_layers, return_sequences = TRUE) %>%
    layer_dropout(rate = dropout_prob) %>%
    layer_layernorm() %>%
    layer_lambda(function(x) x[, dim(x)[2], , drop = FALSE]) %>%
    layer_dense(units = n_steps, activation = "sigmoid")
    return(model)
}


run_pipeline <- function(train_path, validation_path, test_path, look_back, n_steps) {
    X_train, y_train <- load_and_prepare_data(train_path, look_back, n_steps)
    X_validation, y_validation <- load_and_prepare_data(validation_path, look_back, n_steps)
    X_test, y_test <- load_and_prepare_data(test_path, look_back, n_steps)

    lstm <- LSTM_model(input_size = 9, hidden_size = 50, num_layers = 3, dropout_prob = 0.3, n_steps = 1)
    print(summary(lstm))

    lstm %>% compile(
    loss = "mse",
    optimizer = "adam",
    metrics = c("mse", "rmse", "mae", "mape")
    )

    history <- lstm %>% fit(
        x = X_train,
        y = y_train,
        validation_data = list(X_validation, y_validation)
        batch_size = 64,
        epochs = 10,
        verbose = 1
    )

    lstm %>% evaluate(X_test, y_test)
}


LOOK_BACK = 30
## One step
### Vietcombank
vcb_results_one_step <- run_pipeline(
    "./dataset/official_dataset/vcb/vcb_train.csv",
    "./dataset/official_dataset/vcb/vcb_validation.csv",
    "./dataset/official_dataset/vcb/vcb_test.csv",
    LOOK_BACK,
    n_steps = 1
)

### BIDV
bid_results_one_step <- run_pipeline(
    "./dataset/official_dataset/bid/bid_train.csv",
    "./dataset/official_dataset/bid/bid_validation.csv",
    "./dataset/official_dataset/bid/bid_test.csv",
    LOOK_BACK,
    n_steps = 1
)

### Techcombank
tcb_results_one_step <- run_pipeline(
    "./dataset/official_dataset/tcb/tcb_train.csv",
    "./dataset/official_dataset/tcb/tcb_validation.csv",
    "./dataset/official_dataset/tcb/tcb_test.csv",
    LOOK_BACK,
    n_steps = 1
)

## Multi steps
### Vietcombank
vcb_results_multi_steps <- run_pipeline(
    "./dataset/official_dataset/vcb/vcb_train.csv",
    "./dataset/official_dataset/vcb/vcb_validation.csv",
    "./dataset/official_dataset/vcb/vcb_test.csv",
    LOOK_BACK,
    n_steps = 4
)

### BIDV
bid_results_multi_steps <- run_pipeline(
    "./dataset/official_dataset/bid/bid_train.csv",
    "./dataset/official_dataset/bid/bid_validation.csv",
    "./dataset/official_dataset/bid/bid_test.csv",
    LOOK_BACK,
    n_steps = 4
)
### Techcombank

tcb_results_multi_steps <- run_pipeline(
    "./dataset/official_dataset/tcb/tcb_train.csv",
    "./dataset/official_dataset/tcb/tcb_validation.csv",
    "./dataset/official_dataset/tcb/tcb_test.csv",
    LOOK_BACK,
    n_steps = 4
)
