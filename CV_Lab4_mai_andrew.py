import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# Variable declaration.
dataset = pd.read_csv("data_lab4.csv")
degrees = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
x_column = dataset.loc[:, "x"]
y_column = dataset.loc[:, "y"]
k = 5


# k-folds cross validation for polynomial regression
# PARAM: x - training input
# PARAM: y - training input
# PARAM: p - degree of the fitting polynomial
# PARAM: k - number of folds
# RETURN: train_error: Average MAE of training sets across all K folds.
# RETURN: cv_error: average MAE of the validation sets across all K folds.
# NOTES: train_error should return 1.0355, and cv_error should return 1.0848
def poly_kfold_cv(x, y, p, k):
    kframes = int(len(x) / k)
    train_error = []
    cv_error = []
    for i in range(k):
        index_start = i * kframes
        index_end = x.shape[0] - kframes * (k - i - 1)
        validation_y = y[index_start:index_end]
        validation_x = x[index_start:index_end]
        train_x = x.drop(x.index[index_start:index_end])
        train_y = y.drop(y.index[index_start:index_end])
        test_coefficient = np.polyfit(x=train_x, y=train_y, deg=p)
        train_values = np.polyval(test_coefficient, train_x)
        test_values = np.polyval(test_coefficient, validation_x)
        for l in range(len(train_values)):
            train_error.append(abs(train_values[l] - train_y.tolist()[l]))

        for l in range(len(test_values)):
            cv_error.append(abs(test_values[l] - validation_y.tolist()[l]))
    return np.mean(train_error), np.mean(cv_error)


# Part 2
# Loops through the degrees and calls a function to get training error and CV error.
# plots it to a graph.
train_error_plot = []
cv_error_plot = []
for i in degrees:
    te, cve = poly_kfold_cv(x_column, y_column, i, k)
    train_error_plot.append(te)
    cv_error_plot.append(cve)

plt.plot(degrees, train_error_plot, label='Train Error')
plt.plot(degrees, cv_error_plot, label='CV Error')
plt.axvline(5, 0, label="Ideal model complexity X = 5", color="green")
plt.title("Error vs Degrees")
plt.xlabel("Degrees")
plt.ylabel("Error")
plt.legend()
plt.show()

# Part 3, plots based on sample size and P,
sample_size = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
p = [2, 7, 10, 16]
for i in p:
    learning_train_error = []
    learning_cv_error = []
    for l in sample_size:
        sample_x = x_column[:l]
        sample_y = y_column[:l]
        te, cve = poly_kfold_cv(sample_x, sample_y, i, k)
        learning_cv_error.append(cve)
        learning_train_error.append(te)

    plt.plot(sample_size, learning_train_error, label="Training error")
    plt.plot(sample_size, learning_cv_error, label="Cross Verification error")
    plt.xlabel("Sample Size")
    plt.ylabel("Error")
    plt.title("Sample size vs Error rate")
    plt.legend()
    plt.show()

print("3)a)p 2 and p7 shows high bias, with the training error appearing the highest")
print("3)b)Graph 4 gave a ridiculous amount of variance I question whether or not I did this correctly..")
print("4)a)Graph p2 would give the best results for 50 points, with both the training and cv error curve in a negative "
      "slope and about to intersect.")
print("4b) Graph with p7 would be ideal since the graph almost converges together at this point.")