from typing import Any, Callable, ClassVar, Dict, List, Optional, Type

import torch.utils.data
from pydantic import BaseModel, Field, ValidationError, validator
from torch.nn import L1Loss, MSELoss
from torch.optim import SGD, Adam, AdamW, RMSprop
from torch.optim.lr_scheduler import (ConstantLR, CosineAnnealingLR,
                                      MultiStepLR, ReduceLROnPlateau, StepLR)
from torch.utils.data import Dataset

from VAE.datasets import Dataset_LHS
from VAE.loss import customLoss
# from VAE.metrics import rastrigin
from VAE.models.vanilla_vae import VariationalAutoencoder


class ExperimentationConfig(BaseModel):
    """The training/testing configuration."""

    models: ClassVar[Dict[str, torch.nn.Module]] = {
        "VariationalAutoencoder": VariationalAutoencoder,
    }

    datasets: ClassVar[Dict[str, Dataset]] = {
        "Dataset_LHS": Dataset_LHS,
        # "DatasetRandomUniform": DataSetRandomUniform,
    }

    evaluation_metrics: ClassVar[Dict[str, Callable]] = {}

    optimizers: ClassVar[Dict[str, Type]] = {
        "AdamW": AdamW,
        "SGD": SGD,
        "RMSProp": RMSprop,
        "Adam": Adam,
    }

    losses: ClassVar[Dict[str, Type]] = {
        "L1Loss": L1Loss,
        "MSELoss": MSELoss,
        "customLoss": customLoss,
    }

    schedulers: ClassVar[Dict[str, Type]] = {
        "StepLR": StepLR,
        "ReduceLROnPlateau": ReduceLROnPlateau,
        "ConstantLR": ConstantLR,
        "CosineAnnealingLR": CosineAnnealingLR,
        "MultiStepLR": MultiStepLR,
    }

    model_name: str = Field(..., description="The model to train/test with")

    mode: str = Field(
        default="train",
        description="The network mode i.e. `train` or `test` or `finetune`",
    )
    scheduler: str = Field(
        default="CosineAnnealingLR",
        description=" a learning rate scheduling technique to adjust the learning rate during training ",
    )

    epochs: Optional[int] = Field(
        default=30, description="The number of epochs when training"
    )

    Early_Stopping: Optional[bool] = Field(
        default=False,
        description="early stopping to stop the training when the model starts to overfit to the training data.",
    )

    batch_size: int = Field(default=64, description="The batch size when training")

    model_kwargs: Dict[str, Any] = Field(
        ..., description="The keyword arguments to the model"
    )

    train_set_name: str = Field(..., description="The training set")

    train_set_kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="The keyword arguments for the training set"
    )

    valid_set_name: Optional[str] = Field(
        default=None, description="The validation set"
    )

    valid_set_kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="The keyword arguments for the validation set"
    )

    test_set_name: Optional[str] = Field(default=None, description="The test set")

    test_set_kwargs: Optional[Dict[str, Any]] = Field(
        default={}, description="The keyword arguments for the test set"
    )

    optimizer: str = Field(..., description="The optimizer to use")

    optimizer_kwargs: Optional[Dict] = Field(
        default={}, description="The keyword arguments for the optimizer"
    )

    loss: str = Field(default="L1Loss", description="The loss to use")

    loss_kwargs: Optional[Dict] = Field(
        default={}, description="The loss keyword arguments"
    )
    scheduler_kwargs: Optional[Dict] = Field(
        default={}, description="The scheduler keyword arguments"
    )

    save_frequency: Optional[int] = Field(
        default=None, description="How frequently to save image"
    )

    save_dir: Optional[str] = Field(
        default=None,
        description="The directory for the saved models while training",
    )

    weights_path: Optional[str] = Field(
        default=None,
        description="Where to load the model weights from while testing",
    )

    num_workers: int = Field(
        default=4, description="The number of workers to use in dataloaders"
    )

    test_metrics: Optional[List[str]] = Field(
        default=None, description="The test/evaluation metrics"
    )

    def get_model(self) -> torch.nn.Module:
        return self.models[self.model_name]

    def get_train_dataset(self) -> torch.utils.data.Dataset:
        return self.datasets[self.train_set_name]

    def get_valid_dataset(self) -> torch.utils.data.Dataset:
        if self.valid_set_name:
            return self.datasets[self.valid_set_name]
        else:
            raise AttributeError("No validation set name provided.")

    def get_test_dataset(self) -> torch.utils.data.Dataset:
        if self.test_set_name:
            return self.datasets[self.test_set_name]
        else:
            raise AttributeError("No test set name provided.")

    def get_test_metrics(self) -> Dict[str, Callable]:
        return {metric: self.evaluation_metrics[metric] for metric in self.test_metrics}

    def get_optimizer(self) -> Callable:
        return self.optimizers[self.optimizer]

    def get_loss(self) -> Callable:
        return self.losses[self.loss]

    def get_scheduler(self) -> Callable:
        return self.schedulers[self.scheduler]

    @validator("model_name", always=True)
    def model_name_validator(cls, value):
        if value not in cls.models:
            raise ValidationError(f"Model name {value} is not valid")

        return value

    @validator(
        "train_set_name",
        "valid_set_name",
        "test_set_name",
        pre=True,
        always=True,
    )
    def dataset_name_validator(cls, value):
        if value is not None and value not in cls.datasets:
            raise ValidationError(f"Dataset name {value} is not valid")

        return value

    @validator("optimizer", always=True)
    def optimizer_name_validator(cls, value):
        if value is not None and value not in cls.optimizers:
            raise ValidationError(f"Optimizer name {value} is not valid")

        return value

    @validator("loss", always=True)
    def loss_name_validator(cls, value):
        if value is not None and value not in cls.losses:
            raise ValidationError(f"Loss name {value} is not valid")

        return value

    @validator("scheduler", always=True)
    def scheduler_name_validator(cls, value):
        if value is not None and value not in cls.schedulers:
            raise ValidationError(f"Scheduler name {value} is not valid")

        return value

    class Config:
        arbitrary_types_allowed = True
        allow_extra = False
        allow_mutation = False
