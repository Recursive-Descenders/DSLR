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
        choices=("gd", "mbgd", "sgd"),
        default="gd",
        help="gd = full-batch GD; mbgd = minibatch GD (see --batch-size); sgd = batch size 1 (no --batch-size)",
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
        default=200,
        help="Number of training epochs",
    )
    training.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=None,
        metavar="N",
        help="Minibatch size for --optimizer mbgd only (>= 2); default 25%% of dataset if omitted; not used with gd or sgd",
    )
    return parser
