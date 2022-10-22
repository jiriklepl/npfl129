#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import sys
from typing import Optional
import urllib.request

import numpy as np
import numpy.typing as npt
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.linear_model
import sklearn.pipeline
import sklearn.preprocessing

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Path to the dataset to predict")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="thyroid_competition.model", type=str, help="Model path")


class Dataset:
    """Thyroid Dataset.

    The dataset contains real medical data related to thyroid gland function,
    classified either as normal or irregular (i.e., some thyroid disease).
    The data consists of the following features in this order:
    - 15 binary features
    - 6 real-valued features

    The target variable is binary, with 1 denoting a thyroid disease and
    0 normal function.
    """
    def __init__(self,
                 name="thyroid_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2223/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


def main(args: argparse.Namespace) -> Optional[npt.ArrayLike]:
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        normalized=list(range(15,21))
        Cs = np.geomspace(.0625, 2, num=30)

        ct = sklearn.compose.ColumnTransformer([
            ("quant", sklearn.preprocessing.QuantileTransformer(random_state=args.seed), normalized)
        ])
        pf = sklearn.preprocessing.PolynomialFeatures()
        lgr = sklearn.linear_model.LogisticRegression(random_state=args.seed, max_iter=500)
        pl = sklearn.pipeline.Pipeline([("ct", ct), ("pf", pf), ("lgr", lgr)])

        skf = sklearn.model_selection.StratifiedKFold(7)
        parameters = {'lgr__solver':['sag', 'lbfgs'], 'pf__degree': [3, 4, 5], 'lgr__C': Cs}

        clf = sklearn.model_selection.GridSearchCV(estimator=pl, param_grid=parameters, cv=skf, verbose=3, n_jobs=-1)
        clf.fit(X=train.data, y=train.target)

        # TODO: Train a model on the given dataset and store it in `model`.
        model = clf

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(test.data)
        
        if not args.recodex:
            accuracy = np.count_nonzero(test.target == predictions) / np.size(test.target)
            print('accuracy: ' + str(accuracy))

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
