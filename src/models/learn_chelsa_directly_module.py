import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric
import matplotlib.pyplot as plt

import pickle

class LearnChelsaDirectlyModule(LightningModule):
    """
    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```
    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        location_encoder,
        pos_embedding,
        optimizer,
        scheduler,
        compile,
        loss_fn,
        val_cases = None,
        test_cases = None,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["location_encoder", "pos_embedding"])

        self.location_encoder = location_encoder # SirenNet
        self.pos_embedding = pos_embedding # SphereGrid or SH
        
        self.loss_fn = loss_fn

        self.test_cases = test_cases
        self.val_cases = val_cases

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.location_encoder(x)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch):
        """With this model, we directly learn to predict chelsa from the location & month."""
        lonlat, month, chelsa = batch

        # get features
        loc = self.pos_embedding(lonlat) # SphereGrid or SH
        loc = loc.squeeze(dim=1).to(lonlat.device)

        # Append a sin/cos transform of the month to the vector
        loc_month = torch.concat([loc,
            torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),
            torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)

        pred_chelsa = self.location_encoder(loc_month) # SirenNet

        loss = self.loss_fn(pred_chelsa, chelsa)

        return loss


    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss)

        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_end(self):
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch, batch_idx):
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss = self.model_step(batch)

        if batch_idx == 0 and self.logger and self.val_cases:
            wb = self.logger.experiment
            for _, case in self.val_cases.items():
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type="cuda", enabled=False):
                            pe_copy = pickle.loads(pickle.dumps(self.pos_embedding))
                            le_copy = pickle.loads(pickle.dumps(self.location_encoder))
                            case(pe_copy, le_copy, wb, section="val/")

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        "Lightning hook that is called when a validation epoch ends."
        pass

    def test_step(self, batch, batch_idx):
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """

        # Create Switzerland visual
        # We only use wandb logging
        if batch_idx == 0 and self.logger and self.test_cases:
            wb = self.logger.experiment
            for _, case in self.test_cases.items():
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type="cuda", enabled=False):
                            pe_copy = pickle.loads(pickle.dumps(self.pos_embedding))
                            le_copy = pickle.loads(pickle.dumps(self.location_encoder))
                            case(pe_copy, le_copy, wb, section="test/")

        loss = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_test_epoch_end(self):
        """Lightning hook that is called when a test epoch ends."""
        pass

    def setup(self, stage):
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        #params = list(self.chelsa_encoder.parameters()) + list(self.location_encoder.parameters()) + [self.t_prime] + [self.b]
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

if __name__ == "__main__":
    emb_size = 32
    # 4096, 8192, 16384, 32768, 65536
    BS, EXPS, W_UPD, DEV = 8192, 50, True, "cuda"
    print("Testing for BS", BS, "- EXPS", EXPS, "- W_UPD", W_UPD, "- DEV", DEV)

    rand_loc = torch.rand(BS, 34).to(DEV) # REPLACE
    rand_chelsa = torch.rand(BS, 11).to(DEV) # REPLACE

    location_encoder = torch.nn.Linear(34, emb_size)
    chelsa_encoder = torch.nn.Linear(11, emb_size)

    cpm = ChelsaCLIPModule(location_encoder, chelsa_encoder, torch.optim.Adam, None, False).to(DEV)

    params = list(cpm.chelsa_encoder.parameters()) + list(cpm.location_encoder.parameters()) + [cpm.t_prime] + [cpm.b]
    opt = torch.optim.Adam(params=params)

    from tqdm import tqdm
    for _ in tqdm(range(EXPS)):
        if W_UPD:
            opt.zero_grad()
            loss = cpm.model_step((rand_loc, rand_chelsa))
            loss.backward()
            opt.step()
        else:
            cpm.model_step((rand_loc, rand_chelsa))
