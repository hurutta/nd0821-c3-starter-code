import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .model import train_model, compute_model_metrics, inference


# UNIT TESTS

def test_train_model():
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])

    model = train_model(X_train, y_train)

    assert isinstance(model, RandomForestClassifier)
    preds = model.predict(X_train)
    assert len(preds) == len(y_train)


def test_compute_model_metrics():
    y_true = np.array([0, 1, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 0, 1])

    precision, recall, fbeta = compute_model_metrics(y_true, y_pred)

    assert precision == 1.0
    assert recall == 1.0
    assert fbeta == 1.0


def test_inference():
    X_train = np.array([[0, 1], [1, 0], [1, 1], [0, 0]])
    y_train = np.array([0, 1, 1, 0])
    X_test = np.array([[1, 0], [0, 1]])

    model = train_model(X_train, y_train)
    preds = inference(model, X_test)

    assert isinstance(preds, np.ndarray)
    assert preds.shape[0] == X_test.shape[0]
