"""Loads a trained model checkpoint and makes predictions on a dataset."""

from chemprop_solvation.parsing import parse_predict_args
from chemprop_solvation.train import make_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    make_predictions(args)
