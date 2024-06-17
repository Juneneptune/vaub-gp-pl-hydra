from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
from lightning import LightningModule
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification.accuracy import Accuracy
import torch.nn.functional as F

from ..utils.utils_gp import calculate_gp_loss
from ..utils.utils_visual import display_reconstructed_and_flip_images

class VAUBGPModule(LightningModule):

    def __init__(
        self,
        vae1: torch.nn.Module,
        vae2: torch.nn.Module,
        score_prior: torch.nn.Module,
        unet: torch.nn.Module,
        classifier: torch.nn.Module,
        optimizer_vae_1: torch.optim.Optimizer,
        optimizer_vae_2: torch.optim.Optimizer,
        optimizer_score: torch.optim.Optimizer,
        optimizer_cls: torch.optim.Optimizer,
        # scheduler: torch.optim.lr_scheduler,
        beta: float,
        gp_lambda: float,
        cls_lambda: float,
        is_vanilla: bool,
        loops: int,
        compile: bool,
    ) -> None:

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)
        self.automatic_optimization = False

        # initialize all modules
        self.mnist_vae = vae1
        self.flip_mnist_vae = vae2
        self.classifier = classifier
        self.score_model = score_prior(model=unet)

        # metric objects for calculating and averaging accuracy across batches
        self.train_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_src_acc = Accuracy(task="multiclass", num_classes=10)
        self.val_tgt_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_src_acc = Accuracy(task="multiclass", num_classes=10)
        self.test_tgt_acc = Accuracy(task="multiclass", num_classes=10)

        # for averaging loss across batches
        self.train_loss = MeanMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        pass

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_src_acc.reset()
        self.val_tgt_acc.reset()

    def vae_loss(self, recon_x, x, mean, logvar, beta=0.01, score=None, DSM=None):
        if isinstance(recon_x, list) and isinstance(recon_x, list):
            for i in range(len(recon_x)):
                if i == 0:
                    recon_loss = nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
                else:
                    recon_loss += nn.functional.mse_loss(recon_x[i], x[i], reduction='sum')
        else:
            recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
        kld_encoder_posterior = 0.5 * torch.sum(- 1 - logvar)
        kld_prior = 0.5 * torch.sum(mean.pow(2) + logvar.exp())
        kld_loss = kld_encoder_posterior + kld_prior
        if score is not None and DSM is None:
            kld_loss = kld_encoder_posterior - score
        elif DSM is not None:
            kld_loss = kld_encoder_posterior + DSM
        return recon_loss + beta * kld_loss, recon_loss, kld_encoder_posterior, kld_prior

    def training_step(
        self, batch, batch_idx
    ) -> torch.Tensor:

        _, (x1, label1), (x2, label2) = batch
        optimizer_vae_1, optimizer_vae_2, optimizer_score, optimizer_cls = self.optimizers()

        recon_x1, z1, mean1, logvar1 = self.mnist_vae(x1)
        recon_x2, z2, mean2, logvar2 = self.flip_mnist_vae(x2)

        # update per loops
        if (self.global_step % self.hparams.loops) == 0:

            optimizer_vae_1.zero_grad()
            optimizer_vae_2.zero_grad()
            optimizer_cls.zero_grad()

            x, recon_x = [x1.view((x1.shape[0], -1)), x2.view((x2.shape[0], -1))], [recon_x1.view((x1.shape[0], -1)), recon_x2.view((x2.shape[0], -1))]
            z = torch.vstack((z1.view((z1.shape[0], -1)), z2.view((z2.shape[0], -1))))
            mean, logvar = torch.vstack((mean1, mean2)), torch.vstack((logvar1, logvar2))

            # Score loss
            score = self.score_model.get_mixing_score_fn(z, 30 * torch.ones(z.shape[0], device=z.device).type(torch.long),
                                                    detach=True, is_residual=True, is_vanilla=self.hparams.is_vanilla,
                                                    alpha=None) - 0.05 * z
            score = torch.matmul(score.unsqueeze(1), z.unsqueeze(-1)).sum()

            # VAE loss
            vae_loss, recon_loss, kld_encoder_posterior, kld_prior = self.vae_loss(recon_x, x, mean, logvar,
                                                                                   self.hparams.beta,
                                                                                   score=score, DSM=None)

            gp_loss = calculate_gp_loss([x1.view((x1.shape[0], -1)), x2.view((x1.shape[0], -1))], [z1, z2])

            output_cls = self.classifier(z1)
            classifier_loss = F.cross_entropy(output_cls, label1, reduction='sum')

            tot_loss = vae_loss + self.hparams.gp_lambda*gp_loss + self.hparams.cls_lambda*classifier_loss

            tot_loss.backward()
            optimizer_vae_1.step()
            optimizer_vae_2.step()
            optimizer_cls.step()

            # update and log metrics per batch
            self.log("loss/tot_loss", tot_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss/vae_loss", vae_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss/classifier_loss", classifier_loss, on_step=True, on_epoch=False, sync_dist=True)

            self.log("loss_detail/recon_loss", recon_loss, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss_detail/kld_encoder_posterior", kld_encoder_posterior, on_step=True, on_epoch=False, sync_dist=True)
            self.log("loss_detail/kld_prior", kld_prior, on_step=True, on_epoch=False, sync_dist=True)

            self.train_acc(output_cls, label1)
            self.log("train/acc", self.train_acc, on_step=True, on_epoch=False, sync_dist=True)

        # update the score model
        dsm_loss = self.score_model.update_score_fn(z.detach(), optimizer=optimizer_score, max_timestep=None, is_mixing=True,
                                                    is_residual=True, is_vanilla=self.hparams.is_vanilla, alpha=None)

        self.log("loss_detail/dsm_loss", dsm_loss, on_step=True, on_epoch=False, sync_dist=True)

        # update and log metrics per epoch
        self.train_loss(tot_loss)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def model_step(self, batch) -> tuple[torch.Tensor, torch.Tensor]:

        _, (x1, label1), (x2, label2) = batch

        _, z1, _, _ = self.mnist_vae(x1)
        outputs_ori = self.classifier(z1.view((z1.shape[0], 1, 16, 16)))
        _, z2, _, _ = self.flip_mnist_vae(x2)
        outputs_flip = self.classifier(z2.view((z2.shape[0], 1, 16, 16)))

        self.val_src_acc(outputs_ori, label1)
        self.val_tgt_acc(outputs_flip, label2)

        return outputs_ori, outputs_flip

    def validation_step(self, batch, batch_idx: int) -> None:

        _, (x1, label1), (x2, label2) = batch
        outputs_ori, outputs_flip = self.model_step(batch)

        # update and log metrics
        self.val_src_acc(outputs_ori, label1)
        self.val_tgt_acc(outputs_flip, label2)
        self.log("val/src_acc", self.val_src_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/tgt_acc", self.val_tgt_acc, on_step=False, on_epoch=True, prog_bar=True)

        plt_ori = display_reconstructed_and_flip_images(
            epoch=self.current_epoch,
            vae_model=self.mnist_vae,
            flip_vae_model=self.flip_mnist_vae,
            data=x1,
            dim=[1, 28, 28],
            flip_dim=[1, 28, 28]
        )
        plt_flip = display_reconstructed_and_flip_images(
            epoch=self.current_epoch,
            vae_model=self.flip_mnist_vae,
            flip_vae_model=self.mnist_vae,
            data=x2,
            dim=[1, 28, 28],
            flip_dim=[1, 28, 28]
        )

        logger_experiment = self.logger.experiment
        logger_experiment.add_figure("Original to Flipped", plt_ori.gcf(), self.global_step)
        logger_experiment.add_figure("Flipped to Original", plt_flip.gcf(), self.global_step)

        plt_ori.close()
        plt_flip.close()

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        self.val_src_acc.compute()  # get current val acc
        self.val_tgt_acc.compute()  # get current val acc

    def test_step(self, batch, batch_idx: int) -> None:

        _, (x1, label1), (x2, label2) = batch
        outputs_ori, outputs_flip = self.model_step(batch)

        # update and log metrics
        self.test_src_acc(outputs_ori, label1)
        self.test_tgt_acc(outputs_flip, label2)
        self.log("test/src_acc", self.test_src_acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/tgt_acc", self.test_tgt_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self) -> None:
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.mnist_vae = torch.compile(self.mnist_vae)
            self.flip_mnist_vae = torch.compile(self.flip_mnist_vae)
            self.classifier = torch.compile(self.classifier)
            self.score_model = torch.compile(self.score_model)

    def configure_optimizers(self) -> tuple[list[Any], list[Any]]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer_vae_1 = self.hparams.optimizer_vae_1(params=self.mnist_vae.parameters())
        optimizer_vae_2 = self.hparams.optimizer_vae_2(params=self.flip_mnist_vae.parameters())
        optimizer_score = self.hparams.optimizer_score(params=self.score_model.parameters())
        optimizer_cls = self.hparams.optimizer_cls(params=self.classifier.parameters())

        # if self.hparams.scheduler is not None:
        #     scheduler = self.hparams.scheduler(optimizer=optimizer)
        #     return {
        #         "optimizer": optimizer,
        #         "lr_scheduler": {
        #             "scheduler": scheduler,
        #             "monitor": "val/loss",
        #             "interval": "epoch",
        #             "frequency": 1,
        #         },
        #     }

        return [optimizer_vae_1, optimizer_vae_2, optimizer_score, optimizer_cls], []

