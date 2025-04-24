library(jsonlite)
library(plotly)
library(car)
library(multcomp)
library(dplyr)


evaluate_results <- function(data) {
    formula_str <- as.formula("Time ~ Model"))
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


vcb_sarimax_results <- fromJSON("./model_application/experiment/saved_checkpoint/sarimax/vcb/test_results.json")
vcb_time_sarimax <- vcb_sarimax_results$time
bid_sarimax_results <- fromJSON("./model_application/experiment/saved_checkpoint/sarimax/bid/test_results.json")
bid_time_sarimax <- bid_sarimax_results$time
tcb_sarimax_results <- fromJSON("./model_application/experiment/saved_checkpoint/sarimax/tcb/test_results.json")
tcb_time_sarimax <- tcb_sarimax_results$time

vcb_svm_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/svm_test_results.json")
vcb_time_svm_one_step <- vcb_svm_one_step_results$time
vcb_svm_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/svm_test_results.json")
vcb_time_svm_multi_steps <- vcb_svm_multi_steps_results$time
bid_svm_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/svm_test_results.json")
bid_time_svm_one_step <- bid_svm_one_step_results$time
bid_svm_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/svm_test_results.json")
bid_time_svm_multi_steps <- bid_svm_multi_steps_results$time
tcb_svm_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/svm_test_results.json")
tcb_time_svm_one_step <- tcb_svm_one_step_results$time
tcb_svm_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/svm_test_results.json")
tcb_time_svm_multi_steps <- tcb_svm_multi_steps_results$time

vcb_random_forest_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/random_forest_test_results.json")
vcb_time_random_forest_one_step <- vcb_random_forest_one_step_results$time
vcb_random_forest_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/random_forest_test_results.json")
vcb_time_random_forest_multi_steps <- vcb_random_forest_multi_steps_results$time
bid_random_forest_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/random_forest_test_results.json")
bid_time_random_forest_one_step <- bid_random_forest_one_step_results$time
bid_random_forest_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/random_forest_test_results.json")
bid_time_random_forest_multi_steps <- bid_random_forest_multi_steps_results$time
tcb_random_forest_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/random_forest_test_results.json")
tcb_time_random_forest_one_step <- tcb_random_forest_one_step_results$time
tcb_random_forest_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/random_forest_test_results.json")
tcb_time_random_forest_multi_steps <- tcb_random_forest_multi_steps_results$time

vcb_linear_regression_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/linear_regression_test_results.json")
vcb_time_linear_regression_one_step <- vcb_linear_regression_one_step_results$time
vcb_linear_regression_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/linear_regression_test_results.json")
vcb_time_linear_regression_multi_steps <- vcb_linear_regression_multi_steps_results$time
bid_linear_regression_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/linear_regression_test_results.json")
bid_time_linear_regression_one_step <- bid_linear_regression_one_step_results$time
bid_linear_regression_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/linear_regression_test_results.json")
bid_time_linear_regression_multi_steps <- bid_linear_regression_multi_steps_results$time
tcb_linear_regression_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/linear_regression_test_results.json")
tcb_time_linear_regression_one_step <- tcb_linear_regression_one_step_results$time
tcb_linear_regression_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/linear_regression_test_results.json")
tcb_time_linear_regression_multi_steps <- tcb_linear_regression_multi_steps_results$time

vcb_ridge_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/vcb/ridge_test_results.json")
vcb_time_ridge_one_step <- vcb_ridge_one_step_results$time
vcb_ridge_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/vcb/ridge_test_results.json")
vcb_time_ridge_multi_steps <- vcb_ridge_multi_steps_results$time
bid_ridge_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/bid/ridge_test_results.json")
bid_time_ridge_one_step <- bid_ridge_one_step_results$time
bid_ridge_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/bid/ridge_test_results.json")
bid_time_ridge_multi_steps <- bid_ridge_multi_steps_results$time
tcb_ridge_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/one_step/tcb/ridge_test_results.json")
tcb_time_ridge_one_step <- tcb_ridge_one_step_results$time
tcb_ridge_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/machine_learning/multi_steps/tcb/ridge_test_results.json")
tcb_time_ridge_multi_steps <- tcb_ridge_multi_steps_results$time

vcb_lstm_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/vcb/one_step/test_results.json")
vcb_time_lstm_one_step <- vcb_lstm_one_step_results$time
vcb_lstm_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/vcb/multi_steps/test_results.json")
vcb_time_lstm_multi_steps <- vcb_lstm_multi_steps_results$time
bid_lstm_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/bid/one_step/test_results.json")
bid_time_lstm_one_step <- bid_lstm_one_step_results$time
bid_lstm_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/bid/multi_steps/test_results.json")
bid_time_lstm_multi_steps <- bid_lstm_multi_steps_results$time
tcb_lstm_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/tcb/one_step/test_results.json")
tcb_time_lstm_one_step <- tcb_lstm_one_step_results$time
tcb_lstm_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/lstm/tcb/multi_steps/test_results.json")
tcb_time_lstm_multi_steps <- tcb_lstm_multi_steps_results$time

vcb_gru_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/vcb/one_step/test_results.json")
vcb_time_gru_one_step <- vcb_gru_one_step_results$time
vcb_gru_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/vcb/multi_steps/test_results.json")
vcb_time_gru_multi_steps <- vcb_gru_multi_steps_results$time
bid_gru_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/bid/one_step/test_results.json")
bid_time_gru_one_step <- bid_gru_one_step_results$time
bid_gru_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/bid/multi_steps/test_results.json")
bid_time_gru_multi_steps <- bid_gru_multi_steps_results$time
tcb_gru_one_step_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/tcb/one_step/test_results.json")
tcb_time_gru_one_step <- tcb_gru_one_step_results$time
tcb_gru_multi_steps_results <- fromJSON("./model_application/experiment/saved_checkpoint/deep_learning/gru/tcb/multi_steps/test_results.json")
tcb_time_gru_multi_steps <- tcb_gru_multi_steps_results$time


sarimax_time <- c(vcb_time_sarimax, bid_time_sarimax, tcb_time_sarimax)
svm_time_one_step <- c(vcb_time_svm_one_step, bid_time_svm_one_step, tcb_time_svm_one_step)
random_forest_time_one_step <- c(vcb_time_random_forest_one_step, bid_time_random_forest_one_step, tcb_time_random_forest_one_step)
linear_regression_time_one_step <- c(vcb_time_linear_regression_one_step, bid_time_linear_regression_one_step, tcb_time_linear_regression_one_step)
ridge_time_one_step <- c(vcb_time_ridge_one_step, bid_time_ridge_one_step, tcb_time_ridge_one_step)
lstm_time_one_step <- c(vcb_time_lstm_one_step, bid_time_lstm_one_step, tcb_time_lstm_one_step)
gru_time_one_step <- c(vcb_time_gru_one_step, bid_time_gru_one_step, tcb_time_gru_one_step)
time_one_step <- data.frame(
    Model = c(
        rep("SARIMAX", length(sarimax_time)),
        rep("SVM", length(svm_time_one_step)),
        rep("Random Forest", length(random_forest_time_one_step)),
        rep("Linear Regression", length(linear_regression_time_one_step)),
        rep("Ridge", length(ridge_time_one_step)),
        rep("LSTM", length(lstm_time_one_step)),
        rep("GRU", length(gru_time_one_step))
    ),
    Time = c(
        sarimax_time,
        svm_time_one_step,
        random_forest_time_one_step,
        linear_regression_time_one_step,
        ridge_time_one_step,
        lstm_time_one_step,
        gru_time_one_step
    )
)

sarimax_time <- c(vcb_time_sarimax, bid_time_sarimax, tcb_time_sarimax)
svm_time_multi_steps <- c(vcb_time_svm_multi_steps, bid_time_svm_multi_steps, tcb_time_svm_multi_steps)
random_forest_time_multi_steps <- c(vcb_time_random_forest_multi_steps, bid_time_random_forest_multi_steps, tcb_time_random_forest_multi_steps)
linear_regression_time_multi_steps <- c(vcb_time_linear_regression_multi_steps, bid_time_linear_regression_multi_steps, tcb_time_linear_regression_multi_steps)
ridge_time_multi_steps <- c(vcb_time_ridge_multi_steps, bid_time_ridge_multi_steps, tcb_time_ridge_multi_steps)
lstm_time_multi_steps <- c(vcb_time_lstm_multi_steps, bid_time_lstm_multi_steps, tcb_time_lstm_multi_steps)
gru_time_multi_steps <- c(vcb_time_gru_multi_steps, bid_time_gru_multi_steps, tcb_time_gru_multi_steps)
time_multi_steps <- data.frame(
    Model = c(=
        rep("SVM", length(svm_time_multi_steps)),
        rep("Random Forest", length(random_forest_time_multi_steps)),
        rep("Linear Regression", length(linear_regression_time_multi_steps)),
        rep("Ridge", length(ridge_time_multi_steps)),
        rep("LSTM", length(lstm_time_multi_steps)),
        rep("GRU", length(gru_time_multi_steps))
    ),
    Time = c(
        svm_time_multi_steps,
        random_forest_time_multi_steps,
        linear_regression_time_multi_steps,
        ridge_time_multi_steps,
        lstm_time_multi_steps,
        gru_time_multi_steps
    )
)


evaluate_results(time_one_step)
evaluate_results(time_multi_steps)
