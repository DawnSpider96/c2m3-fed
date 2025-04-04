import logging
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics
from hydra.utils import instantiate
from torch.optim import Optimizer

from nn_core.common import PROJECT_ROOT
from nn_core.model_logging import NNLogger

from ccmm.data.datamodule import MetaData

pylogger = logging.getLogger(__name__)


class MyLightningModule(pl.LightningModule):
    logger: NNLogger

    def __init__(self, model, metadata: Optional[MetaData] = None, num_classes: int = None, *args, **kwargs) -> None:
        super().__init__()

        assert metadata is not None or num_classes is not None, "Either metadata or num_classes must be provided"
        # Populate self.hparams with args and kwargs automagically!
        # We want to skip metadata since it is saved separately by the NNCheckpointIO object.
        # Be careful when modifying this instruction. If in doubt, don't do it :]
        self.save_hyperparameters(logger=False, ignore=("metadata",))

        self.metadata = metadata
        self.num_classes = num_classes if num_classes else len(metadata.class_vocab)

        metric = torchmetrics.Accuracy()
        self.train_accuracy = metric.clone()
        self.val_accuracy = metric.clone()
        self.test_accuracy = metric.clone()

        if isinstance(model, omegaconf.DictConfig):
            model = instantiate(model, num_classes=self.num_classes)

        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Method for the forward pass.

        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.
        """
        return self.model(x)

    def unwrap_batch(self, batch):
        if isinstance(batch, Dict):
            x, y = batch["x"], batch["y"]
        else:
            x, y = batch
        return x, y

    def step(self, x, y) -> Mapping[str, Any]:

        output = self(x)
        if isinstance(output, tuple):
            logits, embeddings = output
        else:
            logits = output
            embeddings = None
        loss = F.cross_entropy(logits, y)

        return {"logits": logits.detach(), "loss": loss, "embeddings": embeddings}

    def training_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = self.unwrap_batch(batch)

        step_out = self.step(x, y)
        self.log_dict(
            {"loss/train": step_out["loss"].cpu().detach()},
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        self.train_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/train": self.train_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def validation_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = self.unwrap_batch(batch)

        step_out = self.step(x, y)

        self.log_dict(
            {"loss/val": step_out["loss"].cpu().detach()},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        self.val_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/val": self.val_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def test_step(self, batch: Any, batch_idx: int) -> Mapping[str, Any]:
        x, y = self.unwrap_batch(batch)

        step_out = self.step(x, y)

        self.log_dict(
            {"loss/test": step_out["loss"].cpu().detach()},
        )

        self.test_accuracy(torch.softmax(step_out["logits"], dim=-1), y)
        self.log_dict(
            {
                "acc/test": self.test_accuracy,
            },
            on_epoch=True,
        )

        return step_out

    def configure_optimizers(
        self,
    ) -> Union[Optimizer, Tuple[Sequence[Optimizer], Sequence[Any]]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.

        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.
            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        if "optimizer" not in self.hparams:
            return [torch.optim.Adam(self.parameters(), lr=1e-3)]
        opt = hydra.utils.instantiate(self.hparams.optimizer, params=self.parameters(), _convert_="partial")
        if "lr_scheduler" not in self.hparams:
            return [opt]
        scheduler = hydra.utils.instantiate(self.hparams.lr_scheduler, optimizer=opt)
        return [opt], [scheduler]


# @hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default", version_base="1.1")
# def main(cfg: omegaconf.DictConfig) -> None:
#     """Debug main to quickly develop the Lightning Module.

#     Args:
#         cfg: the hydra configuration
#     """
#     _: pl.LightningModule = hydra.utils.instantiate(
#         cfg.model,
#         optim=cfg.optim,
#         _recursive_=False,
#     )


# if __name__ == "__main__":
#     main()
