import argparse
import pandas as pd
from sklearn import neural_network
import sklearn.metrics as metric
import numpy as np
import math
import matplotlib.pyplot as plt

from methods.MLP import MLP

parser = argparse.ArgumentParser(description='Neural Networks.')
parser.add_argument('-training', dest='training_path')
parser.add_argument('-test', dest='test_path')


def multilayer_perceptron(train_input, train_label):
    print("Starting Multilayer Perceptron...")

    model = MLP.build_model(train_input, train_label, [2, 5, 5, 1], 'sigmoid', 0.1, 3, 20000)


def scikit_multilayer_perceptron(train_input, train_label):
    print("Starting Scikit Multilayer Preceptron...")

    model = neural_network.MLPClassifier(activation='relu', max_iter=20000, hidden_layer_sizes=5)
    model.fit(train_input, train_label)

    print(model.coefs_)

    y_pred = model.predict(train_input)  # prediction
    print(y_pred)
    accuracy = metric.accuracy_score(np.array(train_label).flatten(), np.array(y_pred).flatten(), normalize=True)
    print('accuracy=', accuracy)  # show accuracy score


def init_dataset(training_path):
    print("Initializing dataset...")

    # Read training dataset
    df_train = pd.read_csv(training_path)

    training_input = df_train.iloc[:, 1:df_train.shape[1]].values
    training_label = df_train.iloc[:, 0].values

    # Normalize pixel value between 0 and 1
    #training_input = training_input/255

    return training_input, training_label


def main():
    args = parser.parse_args()

    training_input, training_label = init_dataset(args.training_path)

    print('Choose your method:')
    print('1 - Multilayer Perceptron')
    print('2 - Scikit Multilayer Perceptron')
    print('Anyone - Exit')

    option = int(input('Option: '))

    if option == 1:
        multilayer_perceptron(training_input, training_label)
    elif option == 2:
        scikit_multilayer_perceptron(training_input, training_label)


if __name__ == '__main__':
    main()
