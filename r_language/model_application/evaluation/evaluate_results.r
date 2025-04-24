library(jsonlite)
library(plotly)
library(car)
library(multcomp)
library(dplyr)


visualize_results <- function(result, model) {
    fig <- plot_ly() %>%
        add_trace(
            x = seq_along(result$targets),
            y = result$targets,
            type = "scatter",
            mode = "lines",
            name = "Targets"
        ) %>%
        add_trace(
            x = seq_along(result$predictions),
            y = result$predictions,
            type = "scatter",
            mode = "lines",
            name = "Predictions"
        ) %>%
        layout(
            title = paste(model, "Predictions"),
            xaxis = list(title = "Time", showticklabels = FALSE),
            yaxis = list(title = "Close")
        )
    fig
}

evaluate_results <- function(data, error) {
    formula_str <- as.formula(paste(error, "~ Model"))
    model <- lm(formula_str, data = data)
    
    anova_result <- anova(model)
    print(anova_result)
    cat("\n")
    
    if (anova_result["Model", "Pr(>F)"] < 0.02) {
        tukey <- glht(model, linfct = mcp(Model = "Tukey"))
        tukey_result <- summary(tukey)
        tukey_df <- as.data.frame(tukey_result$test$pvalues)
        names(tukey_df) <- "p_value"
        
        sig_diff <- data.frame(
        Comparison = rownames(tukey_df),
        p_value = tukey_df$p_value
        ) %>%
        filter(p_value < 0.05)

        print(sig_diff)
    } else {
        cat("Không có sự khác biệt giữa các chỉ số!\n")
    }
}


vcb_sarimax <- fromJSON("./model_application/experiment/saved_checkpoint/sarimax/vcb/test_results.json")
bid_sarimax <- fromJSON("./model_application/experiment/saved_checkpoint/sarimax/bid/test_results.json")
tcb_sarimax <- fromJSON("./model_application/experiment/saved_checkpoint/sarimax/tcb/test_results.json")

vcb_svm_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/svm_test_results.json")
vcb_svm_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/svm_test_results.json")
bid_svm_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/svm_test_results.json")
bid_svm_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/svm_test_results.json")
tcb_svm_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/svm_test_results.json")
tcb_svm_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/svm_test_results.json")

vcb_random_forest_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/random_forest_test_results.json")
vcb_random_forest_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/random_forest_test_results.json")
bid_random_forest_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/random_forest_test_results.json")
bid_random_forest_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/random_forest_test_results.json")
tcb_random_forest_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/random_forest_test_results.json")
tcb_random_forest_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/random_forest_test_results.json")

vcb_linear_regression_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/linear_regression_test_results.json")
vcb_linear_regression_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/linear_regression_test_results.json")
bid_linear_regression_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/linear_regression_test_results.json")
bid_linear_regression_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/linear_regression_test_results.json")
tcb_linear_regression_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/linear_regression_test_results.json")
tcb_linear_regression_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/linear_regression_test_results.json")

vcb_ridge_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/ridge_test_results.json")
vcb_ridge_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/ridge_test_results.json")
bid_ridge_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/ridge_test_results.json")
bid_ridge_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/ridge_test_results.json")
tcb_ridge_one_step <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/ridge_test_results.json")
tcb_ridge_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/ridge_test_results.json")

vcb_lstm_one_step <- ("./model_application/experiment/saved_checkpoint/deep_learning/lstm/vcb/one_step/test_results.json")
vcb_lstm_one_step$predictions <- sapply(vcb_lstm_one_step$predictions, function(x) x[1])
vcb_lstm_one_step$targets <- sapply(vcb_lstm_one_step$targets, function(x) x[1])
vcb_lstm_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/vcb/multi_steps/test_results.json")
bid_lstm_one_step <- ("./model_application/experiment/saved_checkpoint/deep_learning/lstm/bid/one_step/test_results.json")
bid_lstm_one_step$predictions <- sapply(bid_lstm_one_step$predictions, function(x) x[1])
bid_lstm_one_step$targets <- sapply(bid_lstm_one_step$targets, function(x) x[1])
bid_lstm_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/bid/multi_steps/test_results.json")
tcb_lstm_one_step <- ("./model_application/experiment/saved_checkpoint/deep_learning/lstm/tcb/one_step/test_results.json")
tcb_lstm_one_step$predictions <- sapply(tcb_lstm_one_step$predictions, function(x) x[1])
tcb_lstm_one_step$targets <- sapply(tcb_lstm_one_step$targets, function(x) x[1])
tcb_lstm_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/tcb/multi_steps/test_results.json")

vcb_gru_one_step <- ("./model_application/experiment/saved_checkpoint/deep_learning/gru/vcb/one_step/test_results.json")
vcb_gru_one_step$predictions <- sapply(vcb_gru_one_step$predictions, function(x) x[1])
vcb_gru_one_step$targets <- sapply(vcb_gru_one_step$targets, function(x) x[1])
vcb_gru_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/vcb/multi_steps/test_results.json")
bid_gru_one_step <- ("./model_application/experiment/saved_checkpoint/deep_learning/gru/bid/one_step/test_results.json")
bid_gru_one_step$predictions <- sapply(bid_gru_one_step$predictions, function(x) x[1])
bid_gru_one_step$targets <- sapply(bid_gru_one_step$targets, function(x) x[1])
bid_gru_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/bid/multi_steps/test_results.json")
tcb_gru_one_step <- ("./model_application/experiment/saved_checkpoint/deep_learning/gru/tcb/one_step/test_results.json")
tcb_gru_one_step$predictions <- sapply(tcb_gru_one_step$predictions, function(x) x[1])
tcb_gru_one_step$targets <- sapply(tcb_gru_one_step$targets, function(x) x[1])
tcb_gru_multi_steps <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/tcb/multi_steps/test_results.json")


visualize_results(vcb_sarimax, "SARIMAX")
visualize_results(bid_sarimax, "SARIMAX")
visualize_results(tcb_sarimax, "SARIMAX")

visualize_results(vcb_svm_one_step, "SVM")
visualize_results(bid_svm_one_step, "SVM")
visualize_results(tcb_svm_one_step, "SVM")

visualize_results(vcb_random_forest_one_step, "Random Forest")
visualize_results(bid_random_forest_one_step, "Random Forest")
visualize_results(tcb_random_forest_one_step, "Random Forest")

visualize_results(vcb_linear_regression_one_step, "Linear Regression")
visualize_results(bid_linear_regression_one_step, "Linear Regression")
visualize_results(tcb_linear_regression_one_step, "Linear Regression")

visualize_results(vcb_ridge_one_step, "Ridge")
visualize_results(bid_ridge_one_step, "Ridge")
visualize_results(tcb_ridge_one_step, "Ridge")

visualize_results(vcb_lstm_one_step, "LSTM")
visualize_results(bid_lstm_one_step, "LSTM")
visualize_results(tcb_lstm_one_step, "LSTM")

visualize_results(vcb_gru_one_step, "GRU")
visualize_results(bid_gru_one_step, "GRU")
visualize_results(bid_gru_one_step, "GRU")


sarimax_mse <- c(bid_sarimax$mse, tcb_sarimax$mse)
svm_mse_one_step <- c(vcb_svm_one_step$mse, bid_svm_one_step$mse, tcb_svm_one_step$mse)
random_forest_mse_one_step <- c(vcb_random_forest_one_step$mse, bid_random_forest_one_step$mse, tcb_random_forest_one_step$mse)
linear_regression_mse_one_step <- c(vcb_linear_regression_one_step$mse, bid_linear_regression_one_step$mse, tcb_linear_regression_one_step$mse)
ridge_mse_one_step <- c(vcb_ridge_one_step$mse, bid_ridge_one_step$mse, tcb_ridge_one_step$mse)
lstm_mse_one_step <- c(vcb_lstm_one_step$mse, bid_lstm_one_step$mse, tcb_lstm_one_step$mse)
gru_mse_one_step <- c(vcb_gru_one_step$mse, bid_gru_one_step$mse, tcb_gru_one_step$mse)
mse_one_step <- data.frame(
    Model = c(
        rep("SARIMAX", length(sarimax_mse)),
        rep("SVM", length(svm_mse_one_step)),
        rep("Random Forest", length(random_forest_mse_one_step)),
        rep("Linear Regression", length(linear_regression_mse_one_step)),
        rep("Ridge", length(ridge_mse_one_step)),
        rep("LSTM", length(lstm_mse_one_step)),
        rep("GRU", length(gru_mse_one_step))
    ),
    MSE = c(
        sarimax_mse,
        svm_mse_one_step,
        random_forest_mse_one_step,
        linear_regression_mse_one_step,
        ridge_mse_one_step,
        lstm_mse_one_step,
        gru_mse_one_step
    )
)

sarimax_rmse <- c(bid_sarimax$rmse, tcb_sarimax$rmse)
svm_rmse_one_step <- c(vcb_svm_one_step$rmse, bid_svm_one_step$rmse, tcb_svm_one_step$rmse)
random_forest_rmse_one_step <- c(vcb_random_forest_one_step$rmse, bid_random_forest_one_step$rmse, tcb_random_forest_one_step$rmse)
linear_regression_rmse_one_step <- c(vcb_linear_regression_one_step$rmse, bid_linear_regression_one_step$rmse, tcb_linear_regression_one_step$rmse)
ridge_rmse_one_step <- c(vcb_ridge_one_step$rmse, bid_ridge_one_step$rmse, tcb_ridge_one_step$rmse)
lstm_rmse_one_step <- c(vcb_lstm_one_step$rmse, bid_lstm_one_step$rmse, tcb_lstm_one_step$rmse)
gru_rmse_one_step <- c(vcb_gru_one_step$rmse, bid_gru_one_step$rmse, tcb_gru_one_step$rmse)
rmse_one_step <- data.frame(
    Model = c(
        rep("SARIMAX", length(sarimax_rmse)),
        rep("SVM", length(svm_rmse_one_step)),
        rep("Random Forest", length(random_forest_rmse_one_step)),
        rep("Linear Regression", length(linear_regression_rmse_one_step)),
        rep("Ridge", length(ridge_rmse_one_step)),
        rep("LSTM", length(lstm_rmse_one_step)),
        rep("GRU", length(gru_rmse_one_step))
    ),
    RMSE = c(
        sarimax_rmse,
        svm_rmse_one_step,
        random_forest_rmse_one_step,
        linear_regression_rmse_one_step,
        ridge_rmse_one_step,
        lstm_rmse_one_step,
        gru_rmse_one_step
    )
)

sarimax_mae <- c(bid_sarimax$mae, tcb_sarimax$mae)
svm_mae_one_step <- c(vcb_svm_one_step$mae, bid_svm_one_step$mae, tcb_svm_one_step$mae)
random_forest_mae_one_step <- c(vcb_random_forest_one_step$mae, bid_random_forest_one_step$mae, tcb_random_forest_one_step$mae)
linear_regression_mae_one_step <- c(vcb_linear_regression_one_step$mae, bid_linear_regression_one_step$mae, tcb_linear_regression_one_step$mae)
ridge_mae_one_step <- c(vcb_ridge_one_step$mae, bid_ridge_one_step$mae, tcb_ridge_one_step$mae)
lstm_mae_one_step <- c(vcb_lstm_one_step$mae, bid_lstm_one_step$mae, tcb_lstm_one_step$mae)
gru_mae_one_step <- c(vcb_gru_one_step$mae, bid_gru_one_step$mae, tcb_gru_one_step$mae)
mae_one_step <- data.frame(
    Model = c(
        rep("SARIMAX", length(sarimax_mae)),
        rep("SVM", length(svm_mae_one_step)),
        rep("Random Forest", length(random_forest_mae_one_step)),
        rep("Linear Regression", length(linear_regression_mae_one_step)),
        rep("Ridge", length(ridge_mae_one_step)),
        rep("LSTM", length(lstm_mae_one_step)),
        rep("GRU", length(gru_mae_one_step))
    ),
    MAE = c(
        sarimax_mae,
        svm_mae_one_step,
        random_forest_mae_one_step,
        linear_regression_mae_one_step,
        ridge_mae_one_step,
        lstm_mae_one_step,
        gru_mae_one_step
    )
)

sarimax_mape <- c(bid_sarimax$mape, tcb_sarimax$mape)
svm_mape_one_step <- c(vcb_svm_one_step$mape, bid_svm_one_step$mape, tcb_svm_one_step$mape)
random_forest_mape_one_step <- c(vcb_random_forest_one_step$mape, bid_random_forest_one_step$mape, tcb_random_forest_one_step$mape)
linear_regression_mape_one_step <- c(vcb_linear_regression_one_step$mape, bid_linear_regression_one_step$mape, tcb_linear_regression_one_step$mape)
ridge_mape_one_step <- c(vcb_ridge_one_step$mape, bid_ridge_one_step$mape, tcb_ridge_one_step$mape)
lstm_mape_one_step <- c(vcb_lstm_one_step$mape, bid_lstm_one_step$mape, tcb_lstm_one_step$mape)
gru_mape_one_step <- c(vcb_gru_one_step$mape, bid_gru_one_step$mape, tcb_gru_one_step$mape)
mape_one_step <- data.frame(
    Model = c(
        rep("SARIMAX", length(sarimax_mape)),
        rep("SVM", length(svm_mape_one_step)),
        rep("Random Forest", length(random_forest_mape_one_step)),
        rep("Linear Regression", length(linear_regression_mape_one_step)),
        rep("Ridge", length(ridge_mape_one_step)),
        rep("LSTM", length(lstm_mape_one_step)),
        rep("GRU", length(gru_mape_one_step))
    ),
    MAPE = c(
        sarimax_mape,
        svm_mape_one_step,
        random_forest_mape_one_step,
        linear_regression_mape_one_step,
        ridge_mape_one_step,
        lstm_mape_one_step,
        gru_mape_one_step
    )
)

svm_mse_multi_steps <- c(vcb_svm_multi_steps$mse, bid_svm_multi_steps$mse, tcb_svm_multi_steps$mse)
random_forest_mse_multi_steps <- c(vcb_random_forest_multi_steps$mse, bid_random_forest_multi_steps$mse, tcb_random_forest_multi_steps$mse)
linear_regression_mse_multi_steps <- c(vcb_linear_regression_multi_steps$mse, bid_linear_regression_multi_steps$mse, tcb_linear_regression_multi_steps$mse)
ridge_mse_multi_steps <- c(vcb_ridge_multi_steps$mse, bid_ridge_multi_steps$mse, tcb_ridge_multi_steps$mse)
lstm_mse_multi_steps <- c(vcb_lstm_multi_steps$mse, bid_lstm_multi_steps$mse, tcb_lstm_multi_steps$mse)
gru_mse_multi_steps <- c(vcb_gru_multi_steps$mse, bid_gru_multi_steps$mse, tcb_gru_multi_steps$mse)
mse_multi_steps <- data.frame(
    Model = c(
        rep("SVM", length(svm_mse_multi_steps)),
        rep("Random Forest", length(random_forest_mse_multi_steps)),
        rep("Linear Regression", length(linear_regression_mse_multi_steps)),
        rep("Ridge", length(ridge_mse_multi_steps)),
        rep("LSTM", length(lstm_mse_multi_steps)),
        rep("GRU", length(gru_mse_multi_steps))
    ),
    MSE = c(
        svm_mse_multi_steps,
        random_forest_mse_multi_steps,
        linear_regression_mse_multi_steps,
        ridge_mse_multi_steps,
        lstm_mse_multi_steps,
        gru_mse_multi_steps
    )
)

svm_rmse_multi_steps <- c(vcb_svm_multi_steps$rmse, bid_svm_multi_steps$rmse, tcb_svm_multi_steps$rmse)
random_forest_rmse_multi_steps <- c(vcb_random_forest_multi_steps$rmse, bid_random_forest_multi_steps$rmse, tcb_random_forest_multi_steps$rmse)
linear_regression_rmse_multi_steps <- c(vcb_linear_regression_multi_steps$rmse, bid_linear_regression_multi_steps$rmse, tcb_linear_regression_multi_steps$rmse)
ridge_rmse_multi_steps <- c(vcb_ridge_multi_steps$rmse, bid_ridge_multi_steps$rmse, tcb_ridge_multi_steps$rmse)
lstm_rmse_multi_steps <- c(vcb_lstm_multi_steps$rmse, bid_lstm_multi_steps$rmse, tcb_lstm_multi_steps$rmse)
gru_rmse_multi_steps <- c(vcb_gru_multi_steps$rmse, bid_gru_multi_steps$rmse, tcb_gru_multi_steps$rmse)
rmse_multi_steps <- data.frame(
    Model = c(
        rep("SVM", length(svm_rmse_multi_steps)),
        rep("Random Forest", length(random_forest_rmse_multi_steps)),
        rep("Linear Regression", length(linear_regression_rmse_multi_steps)),
        rep("Ridge", length(ridge_rmse_multi_steps)),
        rep("LSTM", length(lstm_rmse_multi_steps)),
        rep("GRU", length(gru_rmse_multi_steps))
    ),
    RMSE = c(
        svm_rmse_multi_steps,
        random_forest_rmse_multi_steps,
        linear_regression_rmse_multi_steps,
        ridge_rmse_multi_steps,
        lstm_rmse_multi_steps,
        gru_rmse_multi_steps
    )
)

sarimax_mae <- c(bid_sarimax$mae, tcb_sarimax$mae)
svm_mae_multi_steps <- c(vcb_svm_multi_steps$mae, bid_svm_multi_steps$mae, tcb_svm_multi_steps$mae)
random_forest_mae_multi_steps <- c(vcb_random_forest_multi_steps$mae, bid_random_forest_multi_steps$mae, tcb_random_forest_multi_steps$mae)
linear_regression_mae_multi_steps <- c(vcb_linear_regression_multi_steps$mae, bid_linear_regression_multi_steps$mae, tcb_linear_regression_multi_steps$mae)
ridge_mae_multi_steps <- c(vcb_ridge_multi_steps$mae, bid_ridge_multi_steps$mae, tcb_ridge_multi_steps$mae)
lstm_mae_multi_steps <- c(vcb_lstm_multi_steps$mae, bid_lstm_multi_steps$mae, tcb_lstm_multi_steps$mae)
gru_mae_multi_steps <- c(vcb_gru_multi_steps$mae, bid_gru_multi_steps$mae, tcb_gru_multi_steps$mae)
mae_multi_steps <- data.frame(
    Model = c(
        rep("SVM", length(svm_mae_multi_steps)),
        rep("Random Forest", length(random_forest_mae_multi_steps)),
        rep("Linear Regression", length(linear_regression_mae_multi_steps)),
        rep("Ridge", length(ridge_mae_multi_steps)),
        rep("LSTM", length(lstm_mae_multi_steps)),
        rep("GRU", length(gru_mae_multi_steps))
    ),
    MAE = c(
        svm_mae_multi_steps,
        random_forest_mae_multi_steps,
        linear_regression_mae_multi_steps,
        ridge_mae_multi_steps,
        lstm_mae_multi_steps,
        gru_mae_multi_steps
    )
)

sarimax_mape <- c(bid_sarimax$mape, tcb_sarimax$mape)
svm_mape_multi_steps <- c(vcb_svm_multi_steps$mape, bid_svm_multi_steps$mape, tcb_svm_multi_steps$mape)
random_forest_mape_multi_steps <- c(vcb_random_forest_multi_steps$mape, bid_random_forest_multi_steps$mape, tcb_random_forest_multi_steps$mape)
linear_regression_mape_multi_steps <- c(vcb_linear_regression_multi_steps$mape, bid_linear_regression_multi_steps$mape, tcb_linear_regression_multi_steps$mape)
ridge_mape_multi_steps <- c(vcb_ridge_multi_steps$mape, bid_ridge_multi_steps$mape, tcb_ridge_multi_steps$mape)
lstm_mape_multi_steps <- c(vcb_lstm_multi_steps$mape, bid_lstm_multi_steps$mape, tcb_lstm_multi_steps$mape)
gru_mape_multi_steps <- c(vcb_gru_multi_steps$mape, bid_gru_multi_steps$mape, tcb_gru_multi_steps$mape)
mape_multi_steps <- data.frame(
    Model = c(
        rep("SVM", length(svm_mape_multi_steps)),
        rep("Random Forest", length(random_forest_mape_multi_steps)),
        rep("Linear Regression", length(linear_regression_mape_multi_steps)),
        rep("Ridge", length(ridge_mape_multi_steps)),
        rep("LSTM", length(lstm_mape_multi_steps)),
        rep("GRU", length(gru_mape_multi_steps))
    ),
    MAPE = c(
        svm_mape_multi_steps,
        random_forest_mape_multi_steps,
        linear_regression_mape_multi_steps,
        ridge_mape_multi_steps,
        lstm_mape_multi_steps,
        gru_mape_multi_steps
    )
)


evaluate_results(mse_one_step, "MSE")
evaluate_results(rmse_one_step, "RMSE")
evaluate_results(mae_one_step, "MAE")
evaluate_results(mape_one_step, "MAPE")

evaluate_results(mse_multi_steps, "MSE")
evaluate_results(rmse_multi_steps, "RMSE")
evaluate_results(mae_multi_steps, "MAE")
evaluate_results(mape_multi_steps, "MAPE")
