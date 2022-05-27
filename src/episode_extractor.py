import numpy as np
import torch


def extract_episode(img_set_x, img_set_y, num_way, num_shot, num_query):

    chosen_labels = np.random.choice(np.unique(img_set_y), num_way, replace=False)
    examples_per_label = num_shot + num_query
    episode = []

    for label_l in chosen_labels:

        images_with_label_l = img_set_x[img_set_y == label_l]
        shuffled_images = np.random.permutation(images_with_label_l)
        chosen_images = shuffled_images[:examples_per_label]
        episode.append(chosen_images)

    episode = np.array(episode)
    episode = torch.from_numpy(episode).float()
    episode = episode.permute(0, 1, 4, 2, 3)

    episode_dict = {
        "images": episode,
        "num_way": num_way,
        "num_shot": num_shot,
        "num_query": num_query,
    }

    return episode_dict
