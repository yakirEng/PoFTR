import numpy as np
from pathlib import Path

from src.dataset.physical_model.petit_s.utils.petits_configs import get_config, lower_config
from src.dataset.physical_model.physical_model import PhysModel
from src.utils.utils import center_crop

class PetitsSample:
    """
    Class to represent a sample from the Petits dataset.
    """

    def __init__(self, config, idx, image_name):
        """
        Initialize the PetitsSample with the given data.

        :param data: The data for the sample.
        """
        self.config = config
        self.idx = idx
        self.image_name = image_name

        self.raw_data_path = Path(config['data']['raw_data_path'])
        self.data_path = Path(config['data']['data_path']) / config['proj']['base_model']
        self.pan_path = self.raw_data_path / config['data']['wls'][2] # pan
        self.mono_path = self.raw_data_path / config['data']['wl'] # 11um or 9um

        self.pan_image = None
        self.mono_image = None
        self.pan_temp_map = None
        self.t_fpa = None

        self.pm = PhysModel(config=config['phys'])


    @classmethod
    def generate(cls, config, image_name, idx):
        obj = cls(config=config, idx=idx, image_name=image_name)
        obj.load_data()
        obj.predict_pan_temp()
        return obj

    def load_data(self):
        """
        Load the data for the sample.
        """
        pan_npz = np.load(self.pan_path / f"{self.image_name.split('.')[0]}.npz")
        self.pan_image = center_crop(pan_npz['image'], (256, 256))
        self.t_fpa = np.array([pan_npz['fpa'] / 100.0])
        self.mono_image = center_crop(np.load(self.mono_path / f"{self.image_name}"), (256, 256))

    def predict_pan_temp(self):
        self.pan_temp_map = self.pm.predict_pan_temp(
            pan_image=self.pan_image,
            t_fpa=self.t_fpa
        )

    def to_dict(self):
        """
        Convert the sample to a dictionary.
        """
        return {
            'idx': self.idx,
            'image_name': self.image_name, #metadata

            'mono_image': self.mono_image,
            't_fpa': self.t_fpa, # input

            'pan_temp_map': self.pan_temp_map # gt
        }

