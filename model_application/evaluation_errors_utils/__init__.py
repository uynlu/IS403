from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error
)


def error(y_predict, y_true):
    mse = mean_squared_error(y_true, y_predict)
    rmse = root_mean_squared_error(y_true, y_predict)
    mae = mean_absolute_error(y_true, y_predict)
    mape = mean_absolute_percentage_error(y_true, y_predict)
    return mse, rmse, mae, mape
