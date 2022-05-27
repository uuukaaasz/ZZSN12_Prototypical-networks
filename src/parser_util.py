import argparse


def parse_dataset():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-d",
        "--dataset",
        choices=["omniglot", "mini_imagenet"],
        help="choose the dataset",
        required=True,
    )

    return parser.parse_args()
