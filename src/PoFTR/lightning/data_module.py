import pytorch_lightning as pl
from typing import Sequence, Optional
from torch.utils.data import DataLoader
from src.dataset.mono_ds import MonochromeDs


def load_datasets(config, types=None, is_analysis_mode=False):
    if types is None:
        types = ('train', 'val', 'test')
    dataset_version = config['data']['dataset_version']
    return {
        t: MonochromeDs(config, dataset_version=dataset_version, dataset_type=t, is_analysis_mode=is_analysis_mode)
        for t in types
    }

class SATDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        splits: Optional[Sequence[str]] = None,
        is_analysis_mode: bool = False,
    ):
        """
        Args:
            config: configuration object with .train.batch_size, .run.num_workers, .run.prefetch_factor
            splits: which datasets to prepare; defaults to ('train','val','test')
        """
        super().__init__()
        self.config = config
        self.splits = splits or ('train', 'val', 'test')
        self.is_analysis_mode = is_analysis_mode
        self.batch_size =  config['train']['batch_size']
        self.num_workers = config['run']['num_workers']
        self.prefetch_factor = config['run']['prefetch_factor']

        # will be populated in setup()
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def setup(self, stage: Optional[str] = None):

        # instantiate only once per process
        ds_dict = load_datasets(self.config, types=self.splits, is_analysis_mode=self.is_analysis_mode)
        self.train_dataset = ds_dict.get('train')
        self.val_dataset   = ds_dict.get('val')
        self.test_dataset  = ds_dict.get('test')


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )
