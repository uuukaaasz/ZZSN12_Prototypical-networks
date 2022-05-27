from math import fsum
from os import path

import torch
from tqdm import trange

from config.env import EPOCH_SIZE_TEST, NUM_QUERY, NUM_SHOT, NUM_WAY
from src.utils import extract_episode


def evaluate(model, test_data, logger, result_dir):
    state_dict = torch.load(path.join(result_dir, "best_model.pth"))
    model.load_state_dict(state_dict)

    model.eval()

    test_loss = 0.0
    test_acc = []

    logger.info("> TESTING <")

    for _ in trange(EPOCH_SIZE_TEST):
        episode_dict = extract_episode(
            test_data["test_x"], test_data["test_y"], NUM_WAY, NUM_SHOT, NUM_QUERY
        )

        loss, output = model.set_forward_loss(episode_dict)

        test_loss += output["loss"]
        test_acc.append(output["acc"])

    test_loss = test_loss / EPOCH_SIZE_TEST
    test_acc_avg = sum(test_acc) / EPOCH_SIZE_TEST
    test_acc_dev = fsum([((x - test_acc_avg) ** 2) for x in test_acc])
    test_acc_dev = (test_acc_dev / (EPOCH_SIZE_TEST - 1)) ** 0.5
    error = 1.96 * test_acc_dev / (EPOCH_SIZE_TEST**0.5)
    logger.info(f"Loss: {test_loss} / Acc: {test_acc_avg * 100} +/- {error * 100} %")

    return test_acc_avg


def evaluate_n_times(n, model, test_data, logger, result_dir):
    test_acc_list = []

    test_acc = 0
    std_dev = 0

    for _ in range(n):
        output = evaluate(model, test_data, logger, result_dir)

        test_acc_list.append(output)
        test_acc += output

    test_acc = test_acc / n
    std_dev = fsum([((x - test_acc) ** 2) for x in test_acc_list])
    std_dev = (std_dev / (n - 1)) ** 0.5
    error = 1.96 * std_dev / (n**0.5)

    logger.info(f"With {n} run(s), Acc: {test_acc * 100} +/- {error * 100} %")
