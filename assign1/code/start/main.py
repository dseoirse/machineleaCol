# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import pandas as pd
from sklearn import tree, metrics, neighbors
import matplotlib.pyplot as plt







# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #details for dataset
    training_file = "wildfires_training.csv"
    test_file = "wildfires_test.csv"
    independent_cols = ["year", "temp", "humidity", "rainfall", "drought_code", "buildup_index", "day", "month", "wind_speed"]
    dependent_col = "fire"

    #load dataset
    df_training = pd.read_csv(training_file)
    #print(df_training.head())
    #print(df_training.shape)

    # set up a matrix X containing the independent variables from the training data
    X_training = df_training.loc[:, independent_cols]
    #print(X_training.head())
    #print(X_training.shape)


    # Set up a vector y containing the dependent variable / target attribute for the training data
    y_training = df_training.loc[:, dependent_col]
    #print(y_training.head())
    #print(y_training.shape)

    # Next we load our test dataset in from the file iris_test.csv
    df_test = pd.read_csv(test_file)
    #print(df_test.head())
    #print(df_test.shape)


    # set up a matrix X containing the independent variables from the test data
    X_test = df_test.loc[:, independent_cols]
    #print(X_test.head())
    #print(X_test.shape)


    # Set up a vector y containing the dependent variable / target attribute for the training data
    y_test = df_test.loc[:, dependent_col]
    #print(y_test.head())
    #print(y_test.shape)

    # create a model using the default settings for k-NN, n_neighbors=5, weights=uniform, p=2 (Euclidean distance)
    model = tree.DecisionTreeClassifier()
    model.fit(X_training.values, y_training.values)
    #model = neighbors.KNeighborsClassifier()
    #model.fit(X_training.values, y_training.values)

    # compute the predictions for the training and test sets
    predictions_training = model.predict(X_training.values)
    predictions_test = model.predict(X_test.values)

    # compute the accuracy on the training and test set predictions
    accuracy_training = metrics.accuracy_score(y_training, predictions_training)
    accuracy_test = metrics.accuracy_score(y_test, predictions_test)
    print("Accuracy on training data with default settings:", accuracy_training)
    print("Accuracy on test data with default settings:", accuracy_test)





    depth_values = []
    depth_values = list(range(2, 22, 1))
    max_leaves = []
    max_leaves = list(range(3, 20, 1))

    accuracy_training_depth = []
    accuracy_test_depth = []


    for d in depth_values:
        model_depth = tree.DecisionTreeClassifier(min_samples_split=d)
        model_depth.fit(X_training.values, y_training.values)

        #tree.plot_tree(model_depth)

        # compute the predictions for the training and test sets
        predictions_training_d = model_depth.predict(X_training.values)
        predictions_test_d = model_depth.predict(X_test.values)


        # compute the accuracy on the training and test set predictions
        accuracy_training_depth.append(metrics.accuracy_score(y_training, predictions_training_d))
        accuracy_test_depth.append(metrics.accuracy_score(y_test, predictions_test_d))

        #print(accuracy_training_depth)
        #print(accuracy_test_depth)

    # # let's plot the accuracy on the training and test set
    plt.scatter(depth_values, accuracy_training_depth, marker="x")
    plt.scatter(depth_values, accuracy_test_depth, marker="+")
    #plt.scatter(max_leaves, accuracy_training_depth, marker="T")
    #plt.scatter(max_leaves, accuracy_test_depth, marker="Y")
    plt.xlim([0, max(depth_values)+2])
    plt.ylim([0.0, 1.1])
    plt.xlabel("Value of depth")
    plt.ylabel("Accuracy")
    legend_labels = ["Training (Euclidian dist.)", "Test (Euclidian dist.)"]
    plt.legend(labels=legend_labels, loc=4, borderpad=1)
    plt.title("Effect of max depth of tree on training and test set accuracy", fontsize=10)
    plt.show()

    # Now let's evaluate the effect of using different k values
    # start at k=1 and test all odd k values up to 21
    #k_values = list(range(1, 31, 2))
    #print(k_values)

    # accuracy_training_k = []
    # accuracy_test_k = []
    # for k in k_values:
    #     model_k = neighbors.KNeighborsClassifier(k)
    #     model_k.fit(X_training.values, y_training.values)
    #
    #     # compute the predictions for the training and test sets
    #     predictions_training_k = model_k.predict(X_training.values)
    #     predictions_test_k = model_k.predict(X_test.values)
    #
    #     # compute the accuracy on the training and test set predictions
    #     accuracy_training_k.append(metrics.accuracy_score(y_training, predictions_training_k))
    #     accuracy_test_k.append(metrics.accuracy_score(y_test, predictions_test_k))
    #
    # print(accuracy_training_k)
    # print(accuracy_test_k)






    # # let's plot the accuracy on the training and test set
    # plt.scatter(k_values, accuracy_training_k, marker="x")
    # plt.scatter(k_values, accuracy_test_k, marker="+")
    # plt.xlim([0, max(k_values)+2])
    # plt.ylim([0.0, 1.1])
    # plt.xlabel("Value of k")
    # plt.ylabel("Accuracy")
    # legend_labels = ["Training (Euclidian dist.)", "Test (Euclidian dist.)"]
    # plt.legend(labels=legend_labels, loc=4, borderpad=1)
    # plt.title("Effect of k on training and test set accuracy", fontsize=10)
    # plt.show()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
