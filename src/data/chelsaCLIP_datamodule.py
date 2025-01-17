from typing import Any, Dict, Optional, Tuple

import torch
from lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.transforms import transforms
from tqdm import tqdm
import numpy as np

from src.data.chelsaCLIP_dataset import ChelsaCLIPDataset
from src.data.world_sampler import WeightedRandomWorldSampler


class ChelsaCLIPDataModule(LightningDataModule):
    """`LightningDataModule` for the ChelsaCLIP dataset.

    A `LightningDataModule` implements 7 key methods:

    ```python
        def prepare_data(self):
        # Things to do on 1 GPU/TPU (not on every GPU/TPU in DDP).
        # Download data, pre-process, split, save to disk, etc...

        def setup(self, stage):
        # Things to do on every process in DDP.
        # Load data, set variables, etc...

        def train_dataloader(self):
        # return train dataloader

        def val_dataloader(self):
        # return validation dataloader

        def test_dataloader(self):
        # return test dataloader

        def predict_dataloader(self):
        # return predict dataloader

        def teardown(self, stage):
        # Called on every process in DDP.
        # Clean up after fit or test.
    ```

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://lightning.ai/docs/pytorch/latest/data/datamodule.html
    """

    def __init__(
        self,
        climatology_dir: str,
        input_dir: str,
        batch_size: int,
        num_workers: int = 0,
        pin_memory: bool = False,
        skip_samples: int = None,
        use_all_for_training: bool = False,
        return_size: int = 1,
        provide_chelsa_similarity_matrix: bool = False,
        local_multi_sampling: bool = False,
        sampler: str = "default",
        whiten_with_pca: bool = False,
        months = "march",
    ) -> None:
        """Initialize a `ChelsaCLIPDataModule`.

        :param data_dir: The data directory. Defaults to `"data/"`.
        :param batch_size: The batch size. Defaults to `64`.
        :param num_workers: The number of workers. Defaults to `0`.
        :param pin_memory: Whether to pin memory. Defaults to `False`.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
        #TODO

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.monthly_arrays = None

        self.batch_size_per_device = batch_size

        def collate_fn(batch):
            if provide_chelsa_similarity_matrix:
                if local_multi_sampling:
                    lonlat = torch.cat([x[0] for x in batch], 0)
                    months = torch.cat([x[1] for x in batch], 0)
                    chelsa = torch.cat([x[2] for x in batch], 0)
                else:
                    lonlat = torch.stack([torch.tensor(x[0]) for x in batch])
                    months = torch.stack([torch.tensor(x[1]) for x in batch])
                    chelsa = torch.stack([torch.tensor(x[2]) for x in batch])
                norm = torch.norm(chelsa, dim=1).view(-1,1).type(torch.float)
                similarity = torch.mm(chelsa.type(torch.float),chelsa.T.type(torch.float)) / torch.mm(norm, norm.T)
                if isinstance(provide_chelsa_similarity_matrix, float):
                    # The case where we pass a cutoff
                    similarity = (similarity > provide_chelsa_similarity_matrix).type(torch.float)
                    similarity = similarity * 2 - 1
                #total = len(chelsa) * len(chelsa)
                #print("Total is", total)
                #pos_samples = len(similarity) * len(similarity) + similarity.sum()
                #sim = (similarity > 0.999999).sum()
                #print("Similarity cutoff 0.999999", sim, "->", (sim/total)*100, "percent")
                return lonlat, months, chelsa, similarity
            elif local_multi_sampling:
                lonlat = torch.cat([x[0] for x in batch], 0)
                months = torch.cat([x[1] for x in batch], 0)
                chelsa = torch.cat([x[2] for x in batch], 0)
                return lonlat, months, chelsa
            else:
                lonlat = torch.stack([torch.tensor(x[0]) for x in batch])
                months = torch.stack([torch.tensor(x[1]) for x in batch])
                chelsa = torch.stack([torch.tensor(x[2]) for x in batch])
                #chelsa = torch.stack([x[1] for x in batch])
                return lonlat, months, chelsa

        self.collate_fn = collate_fn

    def prepare_data(self) -> None:
        """Download data if needed. Lightning ensures that `self.prepare_data()` is called only
        within a single process on CPU, so you can safely add your downloading logic within. In
        case of multi-node training, the execution of this hook depends upon
        `self.prepare_data_per_node()`.

        Do not use it to assign state (self.x = y).
        """
        pass

    def setup(self, stage: Optional[str] = None) -> None:
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by Lightning before `trainer.fit()`, `trainer.validate()`, `trainer.test()`, and
        `trainer.predict()`, so be careful not to execute things like random split twice! Also, it is called after
        `self.prepare_data()` and there is a barrier in between which ensures that all the processes proceed to
        `self.setup()` once the data is prepared and available for use.

        :param stage: The stage to setup. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`. Defaults to ``None``.
        """
        if self.monthly_arrays is None:
            #self.var_names = ["clt", "cmi", "hurs", "pet", "pr", "rsds", "sfcWind", "tas", "tasmax", "tasmin", "vpd"]
            if self.hparams.months == "all":
                self.months = ["01", "02", "03", "04", "05", "06", "07", "08", "09", "10", "11", "12"]
            elif self.hparams.months == "seasons":
                self.months = ["03", "06", "09", "12"]
            elif self.hparams.months == "march":
                self.months = ["03"]
            #self.months = ["03", "09"]
            #self.months = ["03"]
            self.monthly_arrays = {}
            print("Loading monthly rasters")
            for month in tqdm(self.months):
                #self.monthly_arrays[month] = np.load(self.hparams.climatology_dir + month + "_monthly_float16.npy")#, mmap_mode='r')
                self.monthly_arrays[month] = np.load(self.hparams.climatology_dir + month + "_monthly_float16_land_only.npy")

        # load and split datasets only if not loaded already
        if (stage == "validate" or stage == "fit" or stage == "test") and not self.data_train:
            self.data_train = ChelsaCLIPDataset(
                monthly_arrays=self.monthly_arrays,
                land_coordinates_file=self.hparams.input_dir+("land_coordinates.npy" if self.hparams.use_all_for_training else "land_coordinates_train.npy"),
                point_to_coord_file=self.hparams.input_dir+("point_to_coord.npy" if self.hparams.use_all_for_training else "point_to_coord_train.npy"),
                months=self.months,
                skip_samples = self.hparams.skip_samples,
                return_size=self.hparams.return_size,
                local_multi_sampling=self.hparams.local_multi_sampling,
                whiten_with_pca=self.hparams.whiten_with_pca,
            )

        if (stage == "validate" or stage == "fit") and not self.data_val:
            """self.data_val = ChelsaCLIPDataset(
                monthly_arrays=self.monthly_arrays,
                land_coordinates_file=self.hparams.input_dir+"land_coordinates_val.npy",
                point_to_coord_file=self.hparams.input_dir+"point_to_coord_val.npy",
                months=self.months,
                skip_samples = self.hparams.skip_samples,
                return_size=self.hparams.return_size,
                local_multi_sampling=self.hparams.local_multi_sampling,
                whiten_with_pca=self.hparams.whiten_with_pca,
            )"""
            _, self.data_val, _ = torch.utils.data.random_split(self.data_train, [0.9, 0.05, 0.05])

        if stage == "test" and not self.data_test:
            """self.data_test = ChelsaCLIPDataset(
                monthly_arrays=self.monthly_arrays,
                land_coordinates_file=self.hparams.input_dir+"land_coordinates_test.npy",
                point_to_coord_file=self.hparams.input_dir+"point_to_coord_test.npy",
                months=self.months,
                skip_samples = self.hparams.skip_samples,
                return_size=self.hparams.return_size,
                local_multi_sampling=self.hparams.local_multi_sampling,
                whiten_with_pca=self.hparams.whiten_with_pca,
            )"""
            _, _, self.data_test = torch.utils.data.random_split(self.data_train, [0.9, 0.05, 0.05])

    def train_dataloader(self) -> DataLoader[Any]:
        """Create and return the train dataloader.

        :return: The train dataloader.
        """
        if self.hparams.sampler == "rw_samp":
            weights = np.load(self.hparams.input_dir+"idx_to_weight.npy")
            if self.hparams.skip_samples:
                weights = weights[::self.hparams.skip_samples]
            if not self.hparams.use_all_for_training:
                raise ValueError("Can't use weighted random sampling with train sub-set. Need to set data.use_all_for_training: true")
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=weights,
                num_samples=len(self.data_train),
                replacement=True,
                generator=None)
            shuffle = False
        elif self.hparams.sampler == "rw_samp_world":
            weights = np.load(self.hparams.input_dir+"idx_to_weight.npy")
            if self.hparams.skip_samples:
                weights = weights[::self.hparams.skip_samples]
            if not self.hparams.use_all_for_training:
                raise ValueError("Can't use weighted random sampling with train sub-set. Need to set data.use_all_for_training: true")
            sampler = WeightedRandomWorldSampler(weights, len(self.data_train))
            shuffle = False
        else:
            sampler = None
            shuffle = True
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=self.collate_fn,
            sampler=sampler
        )

    def val_dataloader(self) -> DataLoader[Any]:
        """Create and return the validation dataloader.

        :return: The validation dataloader.
        """
        # IMPORTANT: shuffle needs to be True, as the loss is sensitive to the batch composition.
        # Thus we need to match the random composition of the training batches in the validation and test data
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self) -> DataLoader[Any]:
        """Create and return the test dataloader.

        :return: The test dataloader.
        """
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.batch_size_per_device,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True,
            collate_fn=self.collate_fn
        )

    def teardown(self, stage: Optional[str] = None) -> None:
        """Lightning hook for cleaning up after `trainer.fit()`, `trainer.validate()`,
        `trainer.test()`, and `trainer.predict()`.

        :param stage: The stage being torn down. Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
            Defaults to ``None``.
        """
        pass

    def state_dict(self) -> Dict[Any, Any]:
        """Called when saving a checkpoint. Implement to generate and save the datamodule state.

        :return: A dictionary containing the datamodule state that you want to save.
        """
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Called when loading a checkpoint. Implement to reload datamodule state given datamodule
        `state_dict()`.

        :param state_dict: The datamodule state returned by `self.state_dict()`.
        """
        pass


if __name__ == "__main__":
    _ = ChelsaCLIPDataModule(
        climatology_dir="/shares/wegner.ics.uzh/CHELSA/climatologies/1981-2010_numpy/",
        input_dir="/shares/wegner.ics.uzh/CHELSA/input/",
        batch_size=32000)
