import argparse

def build_parser():
    parser = argparse.ArgumentParser(
        description="Train logistic regression on Hogwarts dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "csv",
        nargs="?",
        default="dataset_train.csv",
        help="Path to training CSV",
    )
    training = parser.add_argument_group("Training")
    training.add_argument(
        "-o",
        "--optimizer",
        choices=("gd", "sgd"),
        default="gd",
        help="Optimization algorithm",
    )
    training.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    training.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=5000,
        help="Number of training epochs",
    )
    training.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Minibatch size (only used with --optimizer sgd)",
    )
    return parser
