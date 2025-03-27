import torch
from lightning import LightningModule
from torchmetrics import MinMetric, MeanMetric

import wandb

import pickle

class ChelsaCLIPModule(LightningModule):
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
        chelsa_encoder,
        pos_embedding,
        optimizer,
        compile,
        loss_fn,
        scheduler = None,
        val_cases = None,
        test_cases = None,
        provide_chelsa_similarity_matrix = False,
        future_climatologies = False,
        regress_loc = False,
        regress_PE = False,
        regress_chelsa = False,
        chelsa_loss_only = False,
    ):

        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False, ignore=["location_encoder", "chelsa_encoder", "pos_embedding"])

        self.location_encoder = location_encoder # SirenNet
        self.chelsa_encoder = chelsa_encoder # FF-ResNet
        self.pos_embedding = pos_embedding # SphereGrid or SH
        
        self.loss_fn = loss_fn

        self.test_cases = test_cases
        self.val_cases = val_cases

        self.regress_loc = regress_loc
        self.regress_PE = regress_PE
        self.regress_chelsa = regress_chelsa
        self.chelsa_loss_only = chelsa_loss_only

        if self.regress_loc:
            self.loc_regressor = torch.nn.Linear(self.chelsa_encoder.final_transform.out_features, 2)
            self.rec_loss = torch.nn.MSELoss()
        if self.regress_PE:
            self.PE_regressor = torch.nn.Linear(self.chelsa_encoder.final_transform.out_features, self.location_encoder.layers[0].dim_in)
            self.rec_loss = torch.nn.MSELoss()
        if self.regress_chelsa:
            self.chelsa_regressor = torch.nn.Linear(self.chelsa_encoder.final_transform.out_features, self.chelsa_encoder.res_layers[0].in_features)
            self.rec_loss = torch.nn.MSELoss()

        # Note from Sigm? or SatCLIP? paper: We opt for setting ADAM β2 = 0.95 for all our experiments.

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # For keeping track of lowest losses
        self.val_loss_best = MinMetric()
        self.train_loss_best = MinMetric()

        # for tracking best so far validation accuracy
        self.val_loss_best = MinMetric()

        self.first_epoch = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.location_encoder(x)

    def on_train_start(self):
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()

    def model_step(self, batch, verbose=False, plot_learnability=False):
        if self.hparams.provide_chelsa_similarity_matrix:
            #lonlat, month, chelsa, similarity = batch
            raise NotImplementedError()
            pass
        elif self.hparams.future_climatologies:
            lonlat, month, tf, ssp, chelsa = batch
            similarity = None
        else:
            lonlat, month, chelsa = batch
            #lonlat, chelsa = batch
            similarity = None
        if verbose:
            print("lonlat",lonlat.max(), lonlat.min(), lonlat.mean())
            print("chelsa",chelsa.max(), chelsa.min(), chelsa.mean())

        # get features
        loc = self.pos_embedding(lonlat) # SphereGrid or SH
        loc = loc.squeeze(dim=1).to(lonlat.device)
        if verbose:
            print("loc",loc.max(), loc.min(), loc.mean())

        if self.hparams.future_climatologies:
            # Beside month also append tf scaled to [-1;-0.33;0.33;1] and ssp scaled to [-1;0;1]
            loc_month = torch.concat([loc,
            torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),
            torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1),
            ((tf-1.5)/1.5).unsqueeze(dim=-1),
            (ssp-1).unsqueeze(dim=-1)], dim=-1)
        else:
            # Append a sin/cos transform of the month to the vector
            loc_month = torch.concat([loc, torch.sin(month/12*torch.pi*2).unsqueeze(dim=-1),torch.cos(month/12*torch.pi*2).unsqueeze(dim=-1)], dim=-1)
        if verbose:
            print("loc_month",loc_month.max(), loc_month.min(), loc_month.mean())
        
        loc_month_emb = self.location_encoder(loc_month) # SirenNet
        if self.chelsa_loss_only:
            rec = self.chelsa_regressor(loc_month_emb)
            regress_loss = self.rec_loss(rec, chelsa.float())
            #self.log("train/loss_chelsa_reg", regress_loss.detach().cpu())
            return self.regress_chelsa * regress_loss
        #loc_emb = self.location_encoder(loc)
        chelsa_emb = self.chelsa_encoder(chelsa) # FF-ResNet or CNN or PCA or more
        if verbose:
            print("loc_month_emb",loc_month_emb.max(), loc_month_emb.min(), loc_month_emb.mean())
            print("chelsa_emb",chelsa_emb.max(), chelsa_emb.min(), chelsa_emb.mean())

        if hasattr(self.loss_fn, "t_prime"):
            try:
                loss = self.loss_fn(loc_month_emb, chelsa_emb, similarity, verbose=verbose, l_module=self, plot_learnability=plot_learnability)
            except:
                loss = self.loss_fn(loc_month_emb, chelsa_emb, similarity, verbose=verbose)
        else:
            loss = self.loss_fn(loc_month_emb, chelsa_emb, similarity, verbose=verbose)

        if self.regress_chelsa or self.regress_loc or self.regress_PE:
            self.log("train/loss_clip", loss.detach().cpu())

        if self.regress_loc:
            rec = self.loc_regressor(loc_month_emb)
            regress_loss = self.rec_loss(rec, lonlat.float())
            self.log("train/loss_loc_reg", regress_loss.detach().cpu())
            loss += self.regress_loc * regress_loss

        if self.regress_chelsa:
            rec = self.chelsa_regressor(loc_month_emb)
            regress_loss = self.rec_loss(rec, chelsa.float())
            self.log("train/loss_chelsa_reg", regress_loss.detach().cpu())

        if self.regress_PE:
            rec = self.PE_regressor(loc_month_emb)
            regress_loss = self.rec_loss(rec, loc.float())
            self.log("train/loss_PE_reg", regress_loss.detach().cpu())
            loss += self.regress_PE * regress_loss

        if verbose:
            raise ValueError()

        return loss


    def training_step(self, batch, batch_idx):
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        #if batch_idx == 0 :
        #    print("-----------Train verbose -------------")
        loss = self.model_step(batch)
        #if batch_idx == 0 :
        #    print("-----------End train verbose -------------")

        # update and log metrics
        self.train_loss(loss)
        self.log("train/loss", self.train_loss)
        if hasattr(self.loss_fn, "t_prime"):
            self.log("trainer/loss t_prime", self.loss_fn.t_prime)
            self.log("trainer/loss bias", self.loss_fn.b)
        if hasattr(self.loss_fn, "logit_scale"):
            self.log("trainer/loss logit_scale", self.loss_fn.logit_scale)

        """self.train_loss_best(loss)
        self.log(
            "train/loss_best",
            self.train_loss_best.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )"""

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

        if batch_idx == 0 and self.logger and self.val_cases and not self.first_epoch:
            wb = self.logger.experiment
            for _, case in self.val_cases.items():
                with torch.inference_mode(False):
                    with torch.set_grad_enabled(True):
                        with torch.autocast(device_type="cuda", enabled=False):
                            pe_copy = pickle.loads(pickle.dumps(self.pos_embedding))
                            le_copy = pickle.loads(pickle.dumps(self.location_encoder))
                            case(pe_copy, le_copy, wb, section="val/")
        
        if self.first_epoch:
            self.first_epoch = False

        #if batch_idx == 0 :
        #    print("-----------Val verbose -------------")
        #loss = self.model_step(batch, plot_learnability=(True if batch_idx == 0 else False))
        loss = self.model_step(batch)
        #if batch_idx == 0 :
        #    print("-----------End val verbose -------------")

        # update and log metrics
        self.val_loss(loss)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)

        """self.val_loss_best(loss)
        self.log(
            "val/loss_best",
            self.val_loss_best.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )"""

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
                            case(pe_copy, le_copy, wb)


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
    BS, EXPS, W_UPD, DEV = 8192, 1, False, "cuda"
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
