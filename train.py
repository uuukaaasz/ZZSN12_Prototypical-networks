import pickle
import sys
from os import mkdir, path

import numpy as np
import torch
import torch.optim as optim
from tqdm import trange

from config.env import (
    DECAY_EVERY,
    EPOCH_SIZE,
    GAMMA,
    HID_DIM,
    LERNING_RATE,
    MAX_EPOCH,
    N_TEST,
    NUM_QUERY,
    NUM_SHOT,
    NUM_WAY,
    X_DIM,
    Z_DIM,
)
from src.dataset import load_img
from src.eval import evaluate_n_times
from src.logging import get_logger
from src.parser_util import parse_dataset
from src.protonet import load_protonet
from src.utils import extract_episode


def train(model, opt, train_data, valid_data, logger, result_dir):

    optimizer = optim.Adam(model.parameters(), lr=LERNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, DECAY_EVERY, gamma=GAMMA, last_epoch=-1
    )

    model.train()

    epochs = 0

    while epochs < MAX_EPOCH and not opt["stop"]:
        epoch_loss = 0.0
        epoch_acc = 0.0

        logger.info(f"==> Epoch {epochs + 1}")
        logger.info("> TRAINING <")

        for _ in trange(EPOCH_SIZE):
            episode_dict = extract_episode(
                train_data["train_x"],
                train_data["train_y"],
                NUM_WAY,
                NUM_SHOT,
                NUM_QUERY,
            )
            optimizer.zero_grad()
            loss, output = model.set_forward_loss(episode_dict)
            epoch_loss += output["loss"]
            epoch_acc += output["acc"]
            loss.backward()
            optimizer.step()

        epoch_loss = epoch_loss / EPOCH_SIZE
        epoch_acc = epoch_acc / EPOCH_SIZE

        logger.info(f"Loss: {epoch_loss} / Acc: {epoch_acc * 100} %")

        valid(model, opt, valid_data, epochs + 1, logger, result_dir)

        epochs += 1
        scheduler.step()

    best_epoch = opt["best_epoch"]

    logger.info(
        f"Best loss: {best_epoch['loss']} / Best Acc: {best_epoch['acc'] * 100} %"
    )
    with open(path.join(result_dir, "best_epoch.pkl"), "wb") as f:
        pickle.dump(best_epoch, f, pickle.HIGHEST_PROTOCOL)


def valid(model, opt, valid_data, curr_epoch, logger, result_dir):
    model.eval()

    valid_loss = 0.0
    valid_acc = 0.0

    logger.info("> VALIDATION <")

    for _ in trange(EPOCH_SIZE):
        episode_dict = extract_episode(
            valid_data["valid_x"],
            valid_data["valid_y"],
            NUM_WAY,
            NUM_SHOT,
            NUM_QUERY,
        )

        loss, output = model.set_forward_loss(episode_dict)
        valid_loss += output["loss"]
        valid_acc += output["acc"]

    valid_loss = valid_loss / EPOCH_SIZE
    valid_acc = valid_acc / EPOCH_SIZE

    logger.info(f"Loss: {valid_loss} / Acc: {valid_acc * 100} %")

    if opt["best_epoch"]["loss"] > valid_loss:
        opt["best_epoch"]["number"] = curr_epoch
        opt["best_epoch"]["loss"] = valid_loss
        opt["best_epoch"]["acc"] = valid_acc
        model_file = path.join(result_dir, "best_model.pth")
        torch.save(model.state_dict(), model_file)
        logger.info("> BEST MODEL FOUND <")


if __name__ == "__main__":
    args = parse_dataset()

    result_dir = f"results_{args.dataset}_{NUM_WAY}_{NUM_SHOT}"

    if path.exists(path.join(result_dir)):
        print(
            "Result is already in directory. Delete it or change its name before running."
        )
        sys.exit()
    mkdir(path.join(result_dir))
    mkdir(path.join(result_dir, "logs"))

    model = load_protonet(X_DIM, HID_DIM, Z_DIM)
    opt = {"best_epoch": {"number": -1, "loss": np.inf, "acc": 0}, "stop": False}

    train_x, train_y = load_img(path.join("datasets", args.dataset, "train.pkl"))
    train_data = {"train_x": train_x, "train_y": train_y}

    valid_x, valid_y = load_img(path.join("datasets", args.dataset, "valid.pkl"))
    valid_data = {"valid_x": valid_x, "valid_y": valid_y}

    logger = get_logger(path.join(result_dir, "logs"), "train.log")

    train(model, opt, train_data, valid_data, logger, result_dir)

    best_epoch_file = path.join(result_dir, "best_epoch.pkl")

    with open(best_epoch_file, "rb") as f:
        number = pickle.load(f)["number"]

    logger.info(f"Best epoch was the number: {number}")

    test_x, test_y = load_img(path.join("datasets", args.dataset, "test.pkl"))

    test_data = {"test_x": test_x, "test_y": test_y}

    logger = get_logger(path.join(result_dir, "logs"), "test.log")

    evaluate_n_times(N_TEST, model, test_data, logger, result_dir)
