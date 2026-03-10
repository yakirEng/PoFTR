from functools import partial
from pathlib import Path
from typing import Union
import cv2
import os
from tqdm import tqdm


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from p_tqdm import t_map

from src.configs.poftr_configs import get_config
from src.utils.misc import lower_config, center_crop
# from src.dataset.simulator.PETIT.src.pre_processing import load_measurements # for offline runs
from src.utils.configs import PhysModelConfig


def c2k(celsius):
    return celsius + 273.15


def k2c(kelvin):
    return kelvin - 273.15


class ThermalRegress:
    """The Multi-Linear-Regressor is designed to conveniently perform parallel multi-channel linear regression between big matrices for the colorization task purpose, either using the multiprocessing toolbox or in a numpy vectorized form."""

    def __init__(self, config: PhysModelConfig = None):
        # the physical independent variable (used for plotting)
        # flag for using parallel computing when fitting and predicting
        self.config = config
        self.coefficients = None
        self.coeffs_pan = None
        self.coeffs_9um = None
        self.dtype = None

    def _get_features_mat(self, t_fpa, t_bb):
        package = np if isinstance(t_bb, np.ndarray) else torch
        fpa = c2k(t_fpa.flatten())
        bb = c2k(t_bb.flatten()) ** 4
        fpa_features = package.stack([fpa**2, fpa, package.ones_like(t_fpa.flatten())])
        bb_features = bb * fpa_features
        features_mat = package.stack((*bb_features, *fpa_features)).T
        return features_mat

    def get_train_features(self, x: dict):
        """Converts a measurements dictionary into a feature matrix and target vectors"""

        # load data:
        all_frames, t_fpa, t_bb = x["frames"], x["fpa"], x["bb"]

        # shape features and targets:
        features = self._get_features_mat(t_fpa, t_bb)
        target = all_frames.reshape(all_frames.shape[0], -1)
        return features, target, t_fpa, t_bb

    def plot_data_scatter(self, x, pix_idx, label="samples", id=None):
        """Plots a scatter-plot of the input data at the provided pixel"""
        _, radiance, t_fpa, t_bb = self.get_train_features(x)

        if not plt.fignum_exists(id):
            fig = plt.figure(id, figsize=[3.375, 3.375])
            ax = fig.add_subplot(projection="3d")
        else:
            fig = plt.gcf()
            ax = plt.gca()
        # for visualization facilitation - take only one in every 100 samples:
        t_bb = t_bb[::100]
        t_fpa = t_fpa[::100]
        radiance = radiance[::100]
        ax.scatter(t_bb, t_fpa, radiance[:, pix_idx], label=label, alpha=0.5)
        ax.set_xlabel("$T_\mathit{obj}[C]$")
        ax.set_ylabel("$T_\mathit{int}[C]$")
        ax.set_zlabel("Radiometric Intensity")

        return fig, ax

    def fit(self, x: dict, rcond=-1, debug: bool = False):
        """performs a pixel-wise polynomial regression for grey_levels vs an independent variable

        Parameters:
            x: a dictionary containing the calibration measurements (frames and temperatures)
            debug: flag for plotting the regression maps.
        """

        A, b, _, _ = self.get_train_features(x)
        func = partial(np.linalg.lstsq, A, rcond=rcond)
        res = t_map(func, b.T, desc="getting coefficients")
        coeffs_list = [tup[0] for tup in res]
        regress_coeffs = np.asarray(coeffs_list).T
        self.coefficients = regress_coeffs.reshape(
            regress_coeffs.shape[0], *x["frames"].shape[1:]
        )

        if debug:
            # choose random pixel:
            rand_pix_idx_tup = np.random.randint(
                [0, 0], self.coefficients.shape[1:], size=2
            )
            rand_pix_idx = np.ravel_multi_index(
                rand_pix_idx_tup, dims=self.coefficients.shape[1:]
            )

            # scatter the data:
            fig, ax = self.plot_data_scatter(x, rand_pix_idx)

            # evaluate the modeled plane over the scatter's grid:
            ax_lims = ax.xy_viewLim
            x_grid, y_grid = np.meshgrid(
                np.linspace(ax_lims.xmin, ax_lims.xmax),
                np.linspace(ax_lims.ymin, ax_lims.ymax),
            )
            fitted_plane_z = self.predict(x_query=x_grid, t_fpa=y_grid)[
                :, rand_pix_idx_tup[0], rand_pix_idx_tup[1]
            ]
            surf = ax.plot_surface(
                x_grid,
                y_grid,
                fitted_plane_z.reshape(len(x_grid), -1),
                alpha=0.5,
                color="orange",
                label="fitted surface",
            )

            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d

            # lables and final visualizations:
            ax.legend()
            plt.rcParams.update({'font.size': 22})
            ax.set_title("Fitted Model ($AT_{BB}^4 + BT_{FPA}^2 + CT_{FPA} + D$)")
            fig.savefig('fitted_model.png', dpi=300, bbox_inches='tight')

    @staticmethod
    def _fpa_like(array, t_fpa):
        reps = len(array.flatten()) // len(t_fpa.flatten())
        if isinstance(t_fpa, np.ndarray):
            t_fpa_rep = t_fpa.repeat(reps)
        elif isinstance(t_fpa, torch.Tensor):
            t_fpa_rep = t_fpa.repeat_interleave(reps)
        return t_fpa_rep.reshape(array.shape)

    def predict(
        self,
        x_query: Union[np.ndarray, torch.Tensor],
        t_fpa: Union[np.ndarray, torch.Tensor],
        direction: str = "temp_to_rad",
        band: str = "pan"
    ):
        """Predict the target values by applying the model to the query points.

        Parameters:
            x_query: The query points for the prediction. Should be an ndarray of either radiometric intensities or black-body temperatures.
            t_fpa: The fpa temperature to each of the provided samples [C]
            direction: whether to treat input as temperature and predict radiometry or vice-versa
        Returns:
            results: len(x_query) feature maps, where results[i] corresponds to the predicted features for x_query[i]
        """
        assert self.coefficients is not None

        # For numpy: arr.ndim==2 → (H,W); for torch: arr.ndim==2 → (H,W)
        if isinstance(x_query, np.ndarray):
            if x_query.ndim == 2:
                x_query = x_query[None, ...]        # → (1, H, W)
        else:  # torch.Tensor
            if x_query.ndim == 2:
                x_query = x_query.unsqueeze(0)     # → (1, H, W)

        if isinstance(t_fpa, np.ndarray):
            if t_fpa.ndim == 2:
                t_fpa = t_fpa[None, ...]
        else:
            if t_fpa.ndim == 2:
                t_fpa = t_fpa.unsqueeze(0)
        coefficients = self.coeffs_pan if band == "pan" else self.coeffs_9um
        coeffs_for_pred = coefficients.reshape((self.coefficients.shape[0], -1))
        t_fpa = self._fpa_like(x_query, t_fpa)
        features = self._get_features_mat(t_fpa, x_query)
        if direction == "temp_to_rad":
            t_bb = x_query
            if (
                t_bb.shape[-2:] == self.coefficients.shape[-2:]
            ):  # t_bb is an array with image spatial dimensions - need to mulyiply pixel-wise
                features_reshaped = features.reshape(
                    (t_bb.shape[0], -1, features.shape[-1])
                )
                rad_hat = (features_reshaped * coeffs_for_pred.T).sum(axis=-1)
            else:  # t_bb is an array of temperatures - need to take the product with the coefficients matrix
                rad_hat = features @ coeffs_for_pred
            est = rad_hat
        elif (
            direction == "rad_to_temp"
        ):  # inverse model -> T_BB = sqrt_4((rad - B*T_FPA - C) / A)
            radiometry = x_query.reshape(x_query.shape[0], -1)
            n_fpa_feaures = features.shape[1] // 2
            fpa_features = features[:, n_fpa_feaures:]
            features_reshaped = fpa_features.reshape(
                (radiometry.shape[0], -1, fpa_features.shape[-1])
            )
            num = radiometry - (
                features_reshaped * coeffs_for_pred[n_fpa_feaures:].T
            ).sum(axis=-1)
            den = (features_reshaped * coeffs_for_pred[:n_fpa_feaures].T).sum(axis=-1)
            t_bb_hat = k2c((num / den) ** (1 / 4))
            est = t_bb_hat
        else:
            raise Exception("Invalid prediction format!")

        B = x_query.shape[0]  # true batch‐size
        H, W = self.coefficients.shape[1:]  # spatial dims


        # reshape to (B, H, W)
        est = est.reshape((-1, *self.coefficients.shape[1:]))

        # if single image, drop the batch‐axis:
        if B == 1:
            return est[0]  # → (H, W)
        return est  # → (B, H, W)


    def save(self, target_path: Path):
        np.save(target_path, self.coefficients)

    def load(self, source_path: Path):
        coefficients = np.load(source_path)
        self.coeffs_9um = coefficients["9um"]
        self.coeffs_pan = coefficients["pan"]
        if self.dtype == torch.Tensor:
            self.coefficients = torch.from_numpy(self.coefficients)

    def to(self, device):
        assert self.coefficients is not None
        self.coefficients = self.coefficients.to(device)

    def validate(
        self,
        rad: Union[np.ndarray, torch.Tensor],
        t_fpa: Union[np.ndarray, torch.Tensor],
        t_bb: Union[np.ndarray, torch.Tensor],
        direction: str = "rad_to_temp",
        debug: bool = True,
    ):
        if direction == "temp_to_rad":
            x, y = t_bb, rad
        else:
            x, y = rad, t_bb
        pred_func = partial(self.predict, direction=direction)
        batch_sz = 5
        x_batch = x.reshape(x.shape[0] // batch_sz, batch_sz, *x.shape[1:])
        t_fpa_batch = t_fpa.reshape(-1, batch_sz)
        y_hat = np.stack(
            t_map(pred_func, x_batch, t_fpa_batch, desc="predicting validation set")
        ).reshape(*x.shape)
        pred_err = np.moveaxis(y_hat, 0, -1) - y
        rmse = np.sqrt((pred_err**2).mean())
        if debug:
            y_str = "Radiometric" if direction == "temp_to_rad" else "Temperature"
            fig , ax = plt.subplots()
            ax.hist(pred_err.flatten())
            ax.set_xlabel("Error")
            ax.set_title(
                f"{y_str} Estimation Error (mean={pred_err.mean():.2f}, std={pred_err.std():.2f}), rmse={rmse:.2f}"
            )
            fig.savefig(f"pred_err_{direction}.png", dpi=300, bbox_inches='tight')
        return pred_err, rmse


class PhysModel:
    def __init__(self, config) -> None:
        self.config = config
        self.pan_model = ThermalRegress(config)
        self.dtype = np.ndarray
        self.load(config["phys"]["coeff_path"])

    def save(self, target_path: Path):
        np.savez(
            target_path,
            pan=self.pan_model.coefficients,
            mono=self.mono_model.coefficients,
        )

    def load(self, source_path: Path):
        model = np.load(source_path)
        self.pan_model.coefficients = (
            model["pan"] if self.dtype == np.ndarray else torch.from_numpy(model["pan"])
        )

        if self.dtype == torch.Tensor:
            self.to(self.config.device)

    def fit(self, x_pan, x_mono):
        print("Fitting 9um Model:")
        self.pan_model.fit(x_pan)
        print("Fitting 11um Model:")
        self.mono_model.fit(x_mono)

    def predict_pan_temp(self, pan_image , t_fpa):
        """Predicts the temperature map for the given pan image and fpa temperature."""
        tmp_map = self.pan_model.predict(
            pan_image, t_fpa=t_fpa, direction="rad_to_temp"
        )
        return tmp_map

    def predict_batch(self, source_path, output_path):
        pan_images = os.listdir(source_path)
        output_path.mkdir(parents=True, exist_ok=True)
        for idx, image_name in enumerate(tqdm(pan_images, desc="Predicting batch")):
            pan_image_np = np.load(source_path / image_name)['image']
            t_fpa = np.array([np.load(source_path / image_name)['fpa'] / 100])
            pan_image_centered = center_crop(pan_image_np, (256, 256))
            tmp_map = self.predict_pan_temp(pan_image_centered, t_fpa)
            np.save(f"{output_path}/{image_name.split('.')[0]}.npy", tmp_map)


    def predict(self, data: dict):

        # 1. misc

        image0, image1, t_fpa, H0, H1, image_name = data['raw_image0'], data['raw_image1'], data['t_fpa'], data['H0'], data['H1'], data['image_name']
        h, w = image0.shape[-2:]

        tmp_map0 = self.load_mono_prior(image_name)

        # 2. predict temperature maps on raw images
        tmp_map1 = self.pan_model.predict(
            image1, t_fpa=t_fpa, direction="rad_to_temp"
        )



        tmp_map0_w = cv2.warpPerspective(tmp_map0, H0, (w, h))

        tmp_map1_w = cv2.warpPerspective(tmp_map1, H1, (w, h))


        data.update({
            'phys0': tmp_map0_w,
            'phys1': tmp_map1_w,
        })

    def predict_pan_temp(self, pan_image, t_fpa):
        """Predicts the temperature map for the given pan image and fpa temperature."""
        tmp_map = self.pan_model.predict(
            pan_image, t_fpa=t_fpa, direction="rad_to_temp"
        )
        return tmp_map

    def load_mono_prior(self, image_name):
        prior_path = self.config.priors_path / image_name
        prior = np.load(prior_path, allow_pickle=True)
        return prior


    def to(self, device):
        self.pan_model.to(device)
        self.mono_model.to(device)


def comp_meas_profiles():
    """Compare the model estimation error incurred by the measurement profiles"""

    pan_model = ThermalRegress()
    measurements_dict = {
        "bb_cross": "$T_\mathit{int}$ ramp, $T_\mathit{obj}$ constant",
        "sawtooth": "$T_\mathit{int}$ ramp, $T_\mathit{obj}$ triangular",
        "random": "$T_\mathit{int}$ ramp, $T_\mathit{obj}$ random",
    }
    x_val = load_measurements(Path(r"physical_model\measurements\val"))
    val_err = {key: None for key in measurements_dict.keys()}
    _, ax = plt.subplots()
    err_statistics = pd.DataFrame(
        index=measurements_dict.keys(), columns=["MEAN", "STD", "RMSE"]
    )
    for meas_type, label in measurements_dict.items():
        base_path = Path(rf"physical_model\measurements\{meas_type}\pan")
        measurements = load_measurements(base_path)
        pan_model.fit(measurements, debug=False)

        radiometry_pan = x_val["frames"]
        t_fpa_pan = x_val["fpa"]
        bb_temp = x_val["bb"]
        err_vec, rmse = pan_model.validate(
            radiometry_pan, t_fpa_pan, bb_temp, direction="rad_to_temp"
        )

        err_statistics.loc[meas_type, "MEAN"] = err_vec.mean()
        err_statistics.loc[meas_type, "STD"] = err_vec.std()
        err_statistics.loc[meas_type, "RMSE"] = rmse
    return err_statistics

def compute_coeffs(wl: int, target_shape: tuple[int, int]):
    # 1. Define measurements folder
    measurements_path = Path(fr"Z:\BlackBody\Filters\2301\{wl}m\measurements")

    # 2. Load and preprocess measurements
    data = load_measurements(measurements_path)
    data["frames"] = np.stack([
        cv2.resize(f, target_shape, interpolation=cv2.INTER_AREA) for f in data["frames"]
    ])
    print("Loaded data shapes:", {k: v.shape for k, v in data.items()})
    print("Loaded data shapes:", {k: v.shape for k, v in data.items()})

    # 3. Fit regression model
    regressor = ThermalRegress()
    regressor.fit(data, debug=False)

    # 4. Save the coefficients
    output_path = Path(f"coeff/coefficients_{wl}um{target_shape[0]}.npy")
    regressor.save(output_path)
    print(f"Saved coefficients to {output_path.resolve()}")

    # 5. Validate model accuracy (optional)
    err, rmse = regressor.validate(
            rad=data["frames"], t_fpa=data["fpa"], t_bb=data["bb"],
            debug=True
        )
    print(f"Validation RMSE: {rmse:.3f}")



# === MAIN SCRIPT ===
if __name__ == "__main__":
    config = lower_config((get_config()))
    phys_model = PhysModel(config["poftr"])
    source_path = Path(config["poftr"]["proj"]["data_path"]) / "raw" / "images" / "pan"
    output_path = Path(config["poftr"]["proj"]["data_path"]) / "raw" / "priors" / "pan"
    phys_model.predict_batch(source_path, output_path)



