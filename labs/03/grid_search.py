#!/usr/bin/env python3
import argparse

import numpy as np
import sklearn.datasets
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x: int(x) if x.isdigit() else float(x), help="Test size")
# If you add more arguments, ReCodEx will keep them with your default values.


def main(args: argparse.Namespace) -> float:
    # Load digit dataset
    dataset = sklearn.datasets.load_digits()
    dataset.target = dataset.target % 2

    # If you want to learn about the dataset, you can print some information
    # about it using `print(dataset.DESCR)`.

    # TODO: Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data = sklearn.model_selection.train_test_split(dataset.data, test_size=args.test_size, random_state=args.seed)
    train_target, test_target = sklearn.model_selection.train_test_split(dataset.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a pipeline, which
    # 1. performs sklearn.preprocessing.MinMaxScaler()
    # 2. performs sklearn.preprocessing.PolynomialFeatures()
    # 3. performs sklearn.linear_model.LogisticRegression(random_state=args.seed)
    mms = sklearn.preprocessing.MinMaxScaler()
    pf = sklearn.preprocessing.PolynomialFeatures()
    lgr = sklearn.linear_model.LogisticRegression(random_state=args.seed)
    pl = sklearn.pipeline.Pipeline([("mms", mms), ("pf", pf), ("lgr", lgr)])

    train_data = pl.fit_transform(train_data)
    test_data = pl.transform(test_data)

    # TODO: Then, using sklearn.model_selection.StratifiedKFold(5), evaluate cross-validated
    # train performance of all combinations of the following parameters:
    # - polynomial degree: 1, 2
    # - LogisticRegression regularization C: 0.01, 1, 100
    # - LogisticRegression solver: lbfgs, sag
    skf = sklearn.model_selection.StratifiedKFold(5)
    parameters = {'solver':['lbfgs', 'sag'], 'C':[.01, 1, 10]}





    # For the best combination of parameters, compute the test set accuracy.
    #
    # The easiest way is to use `sklearn.model_selection.GridSearchCV`.

    test_accuracy = sklearn.model_selection.GridSearchCV(param_grid=parameters, cv=skf)

    return test_accuracy


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)
    print("Test accuracy: {:.2f}".format(100 * test_accuracy))
