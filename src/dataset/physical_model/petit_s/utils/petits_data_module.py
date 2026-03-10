import pytorch_lightning as pl
from typing import Sequence, Optional
from torch.utils.data import DataLoader
from src.dataset.physical_model.petit_s.dataset.petits_ds import PetitsDs


def load_datasets(config, wl, fold_idx, types=None):
    if types is None:
        types = ('train', 'val', 'test')
    return {
        t: PetitsDs(config, dataset_type=t, wl=wl, fold_idx=fold_idx)
        for t in types
    }

class PetitSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        config,
        splits: Optional[Sequence[str]] = None,
        wl: Optional[str] = None,
        fold_idx: Optional[int] = None,
    ):
        """
        Args:
            config: configuration object with .train.batch_size, .run.num_workers, .run.prefetch_factor
            splits: which datasets to prepare; defaults to ('train','val','test')
        """
        super().__init__()
        self.config = config
        self.splits = splits or ('train', 'val', 'test')
        self.batch_size = config['train']['batch_size']
        self.num_workers = config['run']['num_workers']
        self.prefetch_factor = config['run']['prefetch_factor']
        self.wl = wl if wl is not None else config['data']['wl']
        self.fold_idx = fold_idx if fold_idx is not None else config['run']['fold_idx']

        # will be populated in setup()
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    def setup(self, stage: Optional[str] = None):
        # instantiate only once per process
        ds_dict = load_datasets(self.config, types=self.splits, wl=self.wl, fold_idx=self.fold_idx)
        self.train_dataset = ds_dict.get('train')
        self.val_dataset   = ds_dict.get('val')
        self.test_dataset  = ds_dict.get('test')


    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=False,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor,
            drop_last=False,
        )
