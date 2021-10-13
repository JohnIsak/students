import datetime
import pandas as pd
import numpy as np
import joblib
import holidays
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor


# Encodes sin-cos of a 24 hour cycle, sin-cos of a 365 year cycle. One-hot encodes a week. and adds a feature that is 0
# when it is not a holiday and 1 when it is a holiday, saturday or sunday.
def encode_with_holidays(data, dates):
    size = len(data["Hour"])
    sin_cos_of_a_day = np.zeros((size, 2))
    sin_cos_of_a_year = np.zeros((size, 2))
    day_of_week = np.zeros(size)
    holiday_or_weekend = np.zeros(size)
    norwegian_holidays = holidays.Norway()
    for i in range(0, size):
        sin_cos_of_a_day[i][0] = np.cos((data["Hour"].iloc[i] * 2 * np.pi) / 24)
        sin_cos_of_a_day[i][1] = np.sin((data["Hour"].iloc[i] * 2 * np.pi) / 24)
        day_of_year = (dates[i] - datetime.datetime(dates[i].year, 1, 1)).days + 1
        sin_cos_of_a_year[i][0] = np.cos((day_of_year * 2 * np.pi) / 365)
        sin_cos_of_a_year[i][1] = np.sin((day_of_year * 2 * np.pi) / 365)
        day_of_week[i] = dates[i].weekday()
        if dates[i] in norwegian_holidays or day_of_week[i] == 5 or day_of_week[i] == 6:
            holiday_or_weekend[i] = 1

    encoder = OneHotEncoder()
    day_of_week_encoded = encoder.fit_transform(day_of_week.reshape(-1, 1)).toarray()
    encoded_data = np.zeros((size, day_of_week_encoded[0].size + 5))
    for i in range(0, size):
        encoded_data[i] = np.append(np.append(day_of_week_encoded[i], holiday_or_weekend[i]),
                                    np.append(sin_cos_of_a_day[i], sin_cos_of_a_year[i]))
    return encoded_data


def main():
    data = pd.read_csv('data.csv')
    data_2020 = pd.read_csv("data_2020.csv")

    data = data.rename(columns={"År": "Year", "Måned": "Month", "Dag": "Day", "Fra_time": "Hour"})
    data_2020 = data_2020.rename(columns={"År": "Year", "Måned": "Month", "Dag": "Day", "Fra_time": "Hour"})

    dates = pd.to_datetime(data[["Year", "Month", "Day", "Hour"]], yearfirst=True, infer_datetime_format=True)
    dates_2020 = pd.to_datetime(data_2020[["Year", "Month", "Day", "Hour"]], yearfirst=True, infer_datetime_format=True)

    seed = 42

    # Tested using hours since the first data point on some models, it did not work well. Also used it on lin-reg
    hours_since_first = np.zeros(dates.size)
    for i in range(1, dates.size):
        delta_time = dates[i] - dates[0]
        hours_since_first[i] = delta_time.days * 24
        hours_since_first[i] += delta_time.seconds / 3600

    # Sin-cos encoding one year and one day. one-hot: week. binary: holiday or not holiday, weekends included.
    encoded_data_holiday = encode_with_holidays(data, dates)
    encoded_data_holiday_2020 = encode_with_holidays(data_2020, dates_2020)

    # You can change "Volum totalt" to "Volum til DNP" or "Volum til SNTR" if you wish to predict these targets instead.
    x_plot = linear_regression(hours_since_first, data["Volum totalt"], seed)

    k_neighbours_regression(encoded_data_holiday, data["Volum totalt"], seed, x_plot)

    gradient_descent(encoded_data_holiday, data["Volum totalt"], seed, x_plot)

    mlp_regressor(encoded_data_holiday, data["Volum totalt"], seed, encoded_data_holiday_2020,
                  data_2020["Volum totalt"], x_plot)


# The best model.
# You can comment out the if,elif statements and uncomment the regr = MLPRegressor....
# If you wish to train a new mlp instead of using the already saved ones.
def mlp_regressor(X, y, seed, x_2020, y_2020, x_plot):
    x_train, x_val, x_test, y_train, y_val, y_test = my_train_test_split(X, y, seed)

    # Code for training the MLP regressor instead of loading.
    # regr = MLPRegressor(max_iter=100, hidden_layer_sizes=(100, 100, 100, 100), verbose=True, activation="relu").fit(x_train, y_train)
    #joblib.dump(regr, "MLPregressor_Volum_Til_SNTR.sav")

    if y.name == "Volum totalt":
        regr = joblib.load("MLPregressor_Total.sav")
    elif y.name == "Volum til SNTR":
        regr = joblib.load("MLPregressor_Volum_Til_SNTR.sav")
    elif y.name == "Volum til DNP":
        regr = joblib.load("MLPregressor_Volum_Til_DNP.sav")

    pred = regr.predict(x_val)
    print_acc(pred, y_val, "Multi-layer perceptron", "Validation data")
    plot_data(x_plot, y_val, pred, "MLP")

    pred = regr.predict(x_test)
    print_acc(pred, y_test, "Multi-layer perceptron", "Test data")

    pred = regr.predict(x_2020)
    print_acc(pred, y_2020, "Multi-layer perceptron", "Test-2020 data")


def gradient_descent(X, y, seed, x_plot):
    x_train, x_val, x_test, y_train, y_val, y_test = my_train_test_split(X, y, seed)
    clf = GradientBoostingRegressor(n_estimators=100, max_depth=3)
    clf.fit(x_train, y_train)
    pred = clf.predict(x_val)
    print_acc(pred, y_val, "Gradient descent", "Validation data")

    plot_data(x_plot, y_val, pred, "Gradient descent")


def linear_regression(X, y, seed):
    x_train, x_val, x_test, y_train, y_val, y_test = my_train_test_split(X, y, seed)
    phi_train = phi_cosine(np.copy(x_train))
    y_train = np.copy(y_train)
    W = np.matmul(np.linalg.pinv(phi_train), y_train)  # fit model

    phi_val = phi_cosine(np.copy(x_val))
    pred = np.matmul(phi_val, W)  # predict Val Data
    print_acc(pred, y_val, "Linear regression", "Validation data")

    plot_data(x_val, y_val, pred, "Linear regression")
    return x_val


def k_neighbours_regression(X, y, seed, x_plot):
    x_train, x_val, x_test, y_train, y_val, y_test = my_train_test_split(X, y, seed)

    k_neighbors_regressor = KNeighborsRegressor(n_neighbors=5)
    k_neighbors_regressor.fit(x_train, y_train)

    predictions = k_neighbors_regressor.predict(x_val)
    print_acc(predictions, y_val, "K neighbours regression", "Validation data")

    plot_data(x_plot, y_val, predictions, "K-nearest neighbour")


# Keeps a fixed seed for splitting x_train_val and x_test so the test data will never be used when changing the seed
# variable
def my_train_test_split(data, y, seed):
    # Fixed seed so test data will be kept separate
    x_train_val, x_test, y_train_val, y_test = train_test_split(data, y, shuffle=True, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25,
                                                      shuffle=True, random_state=seed)
    return x_train, x_val, x_test, y_train, y_val, y_test


def print_acc(predictions, y, model, datatype):
    print(model, "Mean Squared Error on", datatype, ":", mean_squared_error(y, predictions))
    print(model, "Coefficient of Determination on", datatype, ":", r2_score(y, predictions))


# Transformation function for linear regression, it attempts to capture a daily, weekly and yearly cycle.
def phi_cosine(X):
    phi = np.zeros((X.size, 6))
    for i in range(X.size):
        phi[i][0] = 1
        phi[i][1] = np.cos(X[i] * (2 * np.pi / 24))
        phi[i][2] = np.cos(X[i] * (2 * np.pi / 168))
        phi[i][3] = np.cos(X[i] * (2 * np.pi / 8766))
    return phi


def plot_data(x, y, pred, model_used):
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, label="Expected values")
    plt.scatter(x, pred, c="r", ls="--", lw=1, label=model_used)
    plt.legend()
    plt.show()


# Earlier attempt at encoding.
# It sin-cos encodes as 24hour cycle and One hot encodes an entire year.
# One hot encoding an entire year did not seem to work well.
def encode_with_day_of_year_as_feature(data, dates):
    size = len(data["Hour"])
    sin_cos_of_a_day = np.zeros((size, 2))
    day_of_year = np.zeros(size)
    for i in range(0, size):
        sin_cos_of_a_day[i][0] = np.cos((data["Hour"].iloc[i] * 2 * np.pi) / 24)
        sin_cos_of_a_day[i][1] = np.sin((data["Hour"].iloc[i] * 2 * np.pi) / 24)
        day_of_year[i] = (dates[i] - datetime.datetime(dates[i].year, 1, 1)).days + 1

    encoder = OneHotEncoder()
    year = encoder.fit_transform(day_of_year.reshape(-1, 1)).toarray()
    encoded_data = np.zeros((size, year[0].size + 2))
    for i in range(len(encoded_data)):
        encoded_data[i] = np.append(year[i], sin_cos_of_a_day[i])
    return encoded_data


# Earlier attempt at encoding
# This one works pretty well,
# it sin-cos encodes a 24 hour cycle and and 365 day cycle.
def encode_with_day_of_week_as_feature(data, dates):
    size = len(data["Hour"])
    sin_cos_of_a_day = np.zeros((size, 2))
    sin_cos_of_a_year = np.zeros((size, 2))
    day_of_week = np.zeros(size)
    for i in range(0, size):
        sin_cos_of_a_day[i][0] = np.cos((data["Hour"].iloc[i] * 2 * np.pi) / 24)
        sin_cos_of_a_day[i][1] = np.sin((data["Hour"].iloc[i] * 2 * np.pi) / 24)
        day_of_year = (dates[i] - datetime.datetime(dates[i].year, 1, 1)).days + 1
        sin_cos_of_a_year[i][0] = np.cos((day_of_year * 2 * np.pi) / 365)
        sin_cos_of_a_year[i][1] = np.sin((day_of_year * 2 * np.pi) / 365)
        day_of_week[i] = dates[i].weekday()

    encoder = OneHotEncoder()
    day_of_week_encoded = encoder.fit_transform(day_of_week.reshape(-1, 1)).toarray()
    encoded_data = np.zeros((size, day_of_week_encoded[0].size + 4))
    for i in range(0, size):
        encoded_data[i] = np.append(day_of_week_encoded[i], np.append(sin_cos_of_a_day[i], sin_cos_of_a_year[i]))
    return encoded_data


main()
