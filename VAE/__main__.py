import json

from VAE.test import test
from VAE.train import train
from VAE.train_config import ExperimentationConfig


def run(args=None):
    from argparse import ArgumentParser

    parser = ArgumentParser("MosMask UNet")
    parser.add_argument(
        "--mode",
        type=str,
        default="train",
        choices=["train", "test"],
        help="Overwrite mode from the configuration file",
    )

    parser.add_argument(
        "--config", type=str, help="The configuration file for training"
    )
    parser.add_argument(
        "--exp-dir", type=str, help="The experiment directory for tests"
    )
    parser.add_argument(
        "--epoch", type=str, help="Which epoch weights to use", default=None
    )

    args = parser.parse_args(args)

    import torch

    torch.manual_seed(0)

    if args.mode == "train":
        with open(args.config) as json_file:
            settings_json = json.load(json_file)
            if "mode" in args and settings_json["mode"] != args.mode:
                settings_json["mode"] = args.mode
            train_settings = ExperimentationConfig.parse_obj(settings_json)
            train(train_settings)
    else:
        test(args.exp_dir, args.epoch)


if __name__ == "__main__":
    run()
