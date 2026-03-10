import os
import numpy as np
import pickle
from pathlib import Path

from src.dataset.simulator.sensorviewsimulator import SensorViewSimulator
from src.utils.misc import center_crop


class MonoSample:
    def __init__(self, config, sim_config, idx=0, image_name='0.npy', wls=None, ablation_type='standard'):
        self.config = config
        self.idx = idx
        self.image_name = image_name
        self.sim_config = sim_config
        self.wls = wls if wls is not None else config["data"]["wls"][:2]
        self.ablation_type = ablation_type
        self.data_config = config["data"]
        self.sample_shape = self.data_config['sample_shape']
        self.data_path = Path(config["proj"]["data_path"])

        self.wl9_prior_path = self.data_path / 'raw' / "priors" / str(self.data_config["wls"][0]) / self.image_name
        self.wl11_prior_path = self.data_path / 'raw' / "priors" / str(self.data_config["wls"][1]) / self.image_name
        self.pan_prior_path = self.data_path / 'raw'/ "priors" / str(self.data_config["wls"][2]) / self.image_name

        self.wl9_image_path = self.data_path / 'raw' / "images" / str(self.data_config["wls"][0]) / self.image_name
        self.wl11_image_path = self.data_path / 'raw' / "images" / str(self.data_config["wls"][1]) / self.image_name
        self.pan_image_path = self.data_path / 'raw'/ 'images' / str(self.data_config["wls"][2]) / (self.image_name.split('.')[0] + '.npz')

        self.image0 = None
        self.image1 = None
        self.H0 = None
        self.H1 = None
        self.H0to1 = None
        self.H1to0 = None
        self.depth0 = None
        self.depth1 = None
        self.mask0 = None
        self.mask1 = None
        self.t_fpa = None
        self.T_0to1 = None
        self.T_1to0 = None
        self.co_visibility = None
        self.valid_pixels = [None, None]

        self.alt = None
        self.lat = None
        self.long = None
        self.sim_level = None

    @classmethod
    def generate(cls, config, idx, image_name, sim_scale=None, wls=None, ablation_type='standard'):
        for _ in range(10):  # max 10 retries
            try:
                sim_config = config["sim"]
                obj = cls(config=config, sim_config=sim_config, idx=idx, image_name=image_name, wls=wls, ablation_type=ablation_type)
                obj.load_raw_images_and_priors()
                obj.load_metadata()
                SensorViewSimulator.generate_view(obj, sim_scale=sim_scale, wls=wls)
                return obj
            except RuntimeError as e:
                if str(e) != "RestartSimulation":
                    raise
        # If all retries fail, raise an error
        raise ValueError("Failed to generate a valid simulation after multiple attempts.")

    @classmethod
    def load(cls, config, sample_dict):
        obj = cls(config, sample_dict["idx"], sample_dict["image_name"])
        obj.load_raw_images_and_priors()
        obj.from_json(sample_dict)
        SensorViewSimulator.load_view(obj)
        return obj


    def load_raw_images_and_priors(self):
        if self.ablation_type == 'upper_bound':
            images_wls = self.wls
            priors_wls = ['pan', 'pan']
        elif self.ablation_type in ('standard', 'noisy_priors', 'zeroed_priors'):
            images_wls = self.wls
            priors_wls = self.wls
        else:
            raise ValueError(f"Unknown ablation_type: {self.ablation_type}")

        self.wl0_img = self.load_image(wl=images_wls[0])
        self.wl1_img = self.load_image(wl=images_wls[1])

        self.phys0 = self.load_priors(wl=priors_wls[0])
        self.phys1 = self.load_priors(wl=priors_wls[1])


    def load_image(self, wl='pan'):
        paths = {
            '9um': self.wl9_image_path,
            '11um': self.wl11_image_path,
            'pan': self.pan_image_path,
        }
        path = paths.get(wl, self.pan_image_path)
        data = np.load(path)
        if wl == 'pan':
            return center_crop(data['image'], (256, 256))
        return data

    def load_priors(self, wl='pan'):
        paths = {
            '9um': self.wl9_prior_path,
            '11um': self.wl11_prior_path,
            'pan': self.pan_prior_path,
        }
        path = paths.get(wl, self.pan_image_path)
        data = np.load(path)
        if self.ablation_type == 'zeroed_priors':
            data = np.zeros_like(data)
        # elif self.ablation_type == 'noisy_priors':
        #    noise = np.random.normal(0, 0.001, size=data.shape)
        #    data = data + noise
        return data

    def load_metadata(self):
        """
        Load the FPA (Focal Plane Array) temperature from a CSV file.
        The FPA values are stored in a CSV file with the image names as keys.
        """
        npz_file = np.load(self.pan_image_path)
        self.t_fpa = np.array([npz_file['fpa'] / 100])
        self.alt = npz_file['alt']
        self.lat = npz_file['lat']
        self.long = npz_file['long']


    def from_json(self, sample_dict):
        """
        Load the sample from a dictionary that is JSON serializable.
        """
        self.idx = sample_dict['idx']
        self.image_name = sample_dict['image_name']
        if sample_dict['t_fpa'] is None:
            self.t_fpa = None
        elif isinstance(sample_dict['t_fpa'], (int, float)):
            self.t_fpa = np.array([sample_dict['t_fpa'] / 100.0], dtype=float)
        else:
            arr = np.asarray(sample_dict['t_fpa'], dtype=float)
            self.t_fpa = arr / 100.0

    def _get_pair_names(self):
        wl0_path = self.data_path / 'raw' / "images" / f"{self.wls[0]}"
        wl1_path = self.data_path / 'raw' / "images" / f"{self.wls[1]}"
        return str(wl0_path), str(wl1_path), str(self.idx)

    def _load_priors(self):
        phys0 = np.load(self.data_path / 'raw' / "priors" / str(self.wls[0]) / self.image_name)
        phys1 = np.load(self.data_path / 'raw' / "priors" / str(self.wls[1]) / self.image_name)
        priors = {
            "phys0": phys0,
            "phys1": phys1,
        }
        return priors

    def to_dict(self):
        """
        Convert the sample to a dictionary that is JSON serializable.
        """
        return {
            'image0': self.image0,
            'phys0': self.phys0,
            'mask0': self.mask0,
            'depth0': self.depth0,
            'image1': self.image1,
            'phys1': self.phys1,
            'mask1': self.mask1,
            'depth1': self.depth1,
            "T_0to1": self.T_0to1,
            "T_1to0": self.T_1to0,
            'pair_names': (self._get_pair_names()),
            'pair_id': self.idx,

            # additional terms for analysis
            'H0': self.H0,
            'H1': self.H1,
            'co_visibility': self.co_visibility,
            'valid_pixels': self.valid_pixels,

            # 't_fpa': self.t_fpa if self.t_fpa is not None else None,
            # 'idx': self.idx,
            # 'image_name': str(self.image_name),
            # 'raw_image0': self.wl0_img,
            # 'raw_image1': self.wl1_img,
        }




