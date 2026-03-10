import pytorch_lightning as pl
from typing import Sequence, Optional
from torch.utils.data import DataLoader
from src.third_party.MatchAnything.dataset.matchanything_ds import MatchAnythingDs


class MatchAnythingDataModule(pl.LightningDataModule):
    def __init__(
            self,
            config,
            splits: Optional[Sequence[str]] = None,
            is_analysis_mode: bool = False,
    ):
        super().__init__()
        self.config = config
        self.splits = splits or ('test',)
        self.is_analysis_mode = is_analysis_mode

        # Read version for logging/logic
        self.dataset_version = config['data']['dataset_version']
        self.batch_size = 1 # config['train']['batch_size']
        self.num_workers = config['run']['num_workers']

        self.test_dataset = None

    def setup(self, stage: Optional[str] = None):
        # We only care about TEST for zero-shot
        if 'test' in self.splits:
            self.test_dataset = MatchAnythingDs(
                config=self.config,
                dataset_version=self.dataset_version,
                dataset_type='test',
                is_analysis_mode=self.is_analysis_mode
            )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
        )