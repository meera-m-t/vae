import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    MultiStepLR,
    ReduceLROnPlateau,
    StepLR,
)
from torchsummary import summary

from VAE.train_config import ExperimentationConfig
from VAE.utils import SimpleLogger, make_dirs

from .pytorchtools import EarlyStopping


def train_model(model, config, logger):
    TrainDataset = config.get_train_dataset()
    train_set = TrainDataset(**config.train_set_kwargs)
    logger.log(f"Training set size {len(train_set)}")

    if config.valid_set_name:
        ValidationDataset = config.get_valid_dataset()

        valid_set = ValidationDataset(**config.valid_set_kwargs)

        logger.log(f"Validation set size {len(valid_set)}")

    logger.log("Model Summary:")
    # print(summary(model=model, input_size=train_set[0][0].shape))

    Optimizer = config.get_optimizer()
    opt = Optimizer(model.parameters(), **config.optimizer_kwargs)

    logger.log(f"Using optimizer: {config.optimizer}")

    if config.scheduler == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(opt, T_max=config.epochs, eta_min=1e-5)
    elif config.scheduler == "StepLR":
        scheduler = StepLR(opt, step_size=50, gamma=0.1)
    elif config.scheduler == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            opt, factor=0.1, patience=2, verbose=1, min_lr=1e-5
        )
    elif config["scheduler"] == "MultiStepLR":
        scheduler = MultiStepLR(
            opt, milestones=[int(e) for e in "1,2".split(",")], gamma=2 / 3
        )
    elif config.scheduler == "ConstantLR":
        scheduler = None
    else:
        raise NotImplementedError

    Loss = config.get_loss()
    criterion = Loss(**config.loss_kwargs)
    logger.log(f"Using loss: {config.loss}")

    scaler = torch.cuda.amp.GradScaler()

    trainloader = torch.utils.data.DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    if config.valid_set_name:
        validloader = torch.utils.data.DataLoader(
            valid_set,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
        )

    save_dir = (
        Path(config.save_dir).resolve()
        / f"{model.get_save_dir(len(train_set))}-epochs-{config.epochs}"
    )

    if not save_dir.exists():
        make_dirs([save_dir])

    if train_set.num_dimensions in {1, 2, 3}:
        train_set.plot_dist(save_dir / "data-set-distribution.png")

    with (save_dir / "TrainConfig.json").open("w") as frozen_settings_file:
        json.dump(config.dict(exclude_none=True), frozen_settings_file, indent=2)
        logger.log(f"Saved training configuration in {save_dir / 'TrainConfig.json'}")

    best_valid_loss = 5e33
    trigger = 0

    for epoch in range(config.epochs):
        start = time.time()
        train_loss, train_acc, n = 0, 0, 0
        for i, (X) in enumerate(trainloader):
            model.train()
            X = X.float().cuda()
            opt.zero_grad()
            with torch.cuda.amp.autocast():
                loss = criterion(model, X)

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            Loss = loss.item()
            train_loss += Loss
            n += 1

        model.eval()

        location = save_dir / f"epoch_{epoch}_weights.pt"
        if config.save_frequency and (epoch + 1) % config.save_frequency == 0:
            torch.save(model.state_dict(), location)
            logger.log(f"Saved the model in {location}")
            print(train_set.num_dimensions)
            if train_set.num_dimensions in {1, 2, 3}:
                all_ys = []
                for i, (X) in enumerate(trainloader):
                    Y = model(X.float().cuda())
                    all_ys.append(torch.squeeze(Y))
                ys = torch.cat(all_ys)
                kwargs = config.train_set_kwargs
                kwargs["data"] = ys
                new_dataset = TrainDataset(**kwargs)
                new_dataset.plot_dist(
                    save_dir / f"epoch-{epoch}-reconstructed-distribution.png"
                )

        location = save_dir / "best_weights.pt"

        scheduler.step(epoch)
        lr = opt.param_groups[0]["lr"]
        logger.log(
            f"Epoch: {epoch} | Train Loss: {train_loss / n:.4f} | Time: {time.time() - start:.1f}, lr: {lr:.6f}"
        )

        if config.save_frequency and (epoch + 1) % config.save_frequency == 0:
            location = save_dir / f"epoch_{epoch}_weights.pt"
            torch.save(model.state_dict(), location)
            logger.log(f"Saved the model in {location}")

        # ## EarlyStopping
        # if config.Early_Stopping:
        #     early_stopping = EarlyStopping(patience=15, verbose=True)
        #     # early_stopping needs the validation loss to check if it has decresed,
        #     # and if it has, it will make a checkpoint of the current model
        #     early_stopping(val_loss, model)

        #     if early_stopping.early_stop:
        #         print("Early stopping")
        #         break
    return save_dir


def train(config: ExperimentationConfig):
    assert config.mode == "train", "Incorrect settings"
    logger = SimpleLogger(config.model_name + "-Trainer")

    logger.log("Training Settings Dump: ")
    print(json.dumps(config.dict(exclude_none=True), indent=2))

    Model = config.get_model()
    logger.log(f"Using model {config.model_name}")

    kwargs = config.model_kwargs
    model = Model(**kwargs)

    if torch.cuda.is_available():
        model = model.cuda()
    model.train()
    save_dir = train_model(model, config, logger)
