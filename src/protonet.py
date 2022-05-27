import torch
import torch.nn as nn
from torch.autograd import Variable

from src.dist_measurement import euclidean_dist

dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class ProtoNet(nn.Module):
    def __init__(self, encoder):
        super(ProtoNet, self).__init__()
        self.encoder = encoder.to(dev)

    def set_forward_loss(self, episode_dict):
        images = episode_dict["images"].to(dev)

        num_way = episode_dict["num_way"]
        num_shot = episode_dict["num_shot"]
        num_query = episode_dict["num_query"]

        x_support = images[:, :num_shot]
        x_query = images[:, num_shot:]

        target_inds = (
            torch.arange(0, num_way)
            .view(num_way, 1, 1)
            .expand(num_way, num_query, 1)
            .long()
        )
        target_inds = Variable(target_inds, requires_grad=False).to(dev)

        x_support = x_support.contiguous().view(
            num_way * num_shot, *x_support.size()[2:]
        )
        x_query = x_query.contiguous().view(num_way * num_query, *x_query.size()[2:])
        x = torch.cat([x_support, x_query], 0)

        z = self.encoder.forward(x)
        z_dim = z.size(-1)
        z_proto = z[: (num_way * num_shot)].view(num_way, num_shot, z_dim).mean(1)
        z_query = z[(num_way * num_shot) :]

        dists = euclidean_dist(z_query, z_proto)

        log_p_y = nn.functional.log_softmax(-dists, dim=1).view(num_way, num_query, -1)

        loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        acc_val = torch.eq(y_hat, target_inds.squeeze()).float().mean()

        return loss_val, {
            "loss": loss_val.item(),
            "acc": acc_val.item(),
            "y_hat": y_hat,
        }


def load_protonet(x_dim, hid_dim, z_dim):
    def conv_block(layer_input, layer_output):
        conv = nn.Sequential(
            nn.Conv2d(layer_input, layer_output, 3, padding=1),
            nn.BatchNorm2d(layer_output),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        return conv

    encoder = nn.Sequential(
        conv_block(x_dim[0], hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, hid_dim),
        conv_block(hid_dim, z_dim),
        Flatten(),
    )
    return ProtoNet(encoder)
