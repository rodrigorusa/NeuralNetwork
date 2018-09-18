import argparse
import numpy as np
import math
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Neural Networks.')
parser.add_argument('-training', dest='training_path')
parser.add_argument('-test', dest='test_path')


def multilayer_perceptron():
    print("Starting Multilayer Perceptron...")


def scikit_multilayer_perceptron():
    print("Starting Scikit Multilayer Preceptron...")


def init_dataset():
    print("Initializing dataset...")


def main():
    args = parser.parse_args()

    init_dataset(args)

    print('Choose your method:')
    print('1 - Multilayer Perceptron')
    print('2 - Scikit Multilayer Perceptron')
    print('Anyone - Exit')

    option = int(input('Option: '))

    if option == 1:
        multilayer_perceptron()
    elif option == 2:
        scikit_multilayer_perceptron()


if __name__ == '__main__':
    main()
