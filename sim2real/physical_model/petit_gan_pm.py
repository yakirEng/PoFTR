from p_tqdm import t_map
from functools import partial
from typing import Union
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def c2k(celsius):
    return celsius + 273.15


def k2c(kelvin):
    return kelvin - 273.15


class ThermalRegress:
    """Multi-Linear-Regressor for pixel-wise thermal calibration.
    Coefficients are stored as one .npy file per band under:
        sim2real/physical_model/coeffs/coefficients_{wl}.npy
    e.g. coefficients_9000nm.npy, coefficients_11000nm.npy, coefficients_pan.npy
    """

    def __init__(self, config=None):
        self.config = config
        self.coefficients = None   # active band coefficients (set by predict())
        self._coeff_map = {}       # wl_string -> np.ndarray, populated by load()
        self.dtype = None

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def load(self, coeff_dir: Path, wl0: str, wl1: str):
        """Load coefficients for both bands in a wl pair.

        Expects files named:
            {coeff_dir}/coefficients_{wl}.npy

        Args:
            coeff_dir: path to the coeffs/ folder
            wl0: first wavelength string  (e.g. '9um', '11um', 'pan')
            wl1: second wavelength string
        """
        coeff_dir = Path(coeff_dir)
        for wl in [wl0, wl1]:
            if wl in self._coeff_map:
                continue  # already loaded
            filepath = coeff_dir / f"coefficients_{wl}.npy"
            if not filepath.exists():
                raise FileNotFoundError(
                    f"Coefficient file not found: {filepath}\n"
                    f"Available files: {list(coeff_dir.glob('*.npy'))}"
                )
            self._coeff_map[wl] = np.load(filepath)
            print(f"Loaded coefficients for '{wl}': {self._coeff_map[wl].shape}")

    def save(self, coeff_dir: Path, wl: str):
        """Save current self.coefficients as coefficients_{wl}.npy"""
        assert self.coefficients is not None, "No coefficients to save."
        coeff_dir = Path(coeff_dir)
        coeff_dir.mkdir(parents=True, exist_ok=True)
        out_path = coeff_dir / f"coefficients_{wl}.npy"
        np.save(out_path, self.coefficients)
        print(f"Saved coefficients for '{wl}' -> {out_path}")

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _get_features_mat(self, t_fpa, t_bb):
        package = np if isinstance(t_bb, np.ndarray) else torch
        fpa = c2k(t_fpa.flatten())
        bb = c2k(t_bb.flatten()) ** 4
        fpa_features = package.stack([fpa ** 2, fpa, package.ones_like(t_fpa.flatten())])
        bb_features = bb * fpa_features
        features_mat = package.stack((*bb_features, *fpa_features)).T
        return features_mat

    def get_train_features(self, x: dict):
        """Converts a measurements dictionary into a feature matrix and target vectors."""
        all_frames, t_fpa, t_bb = x["frames"], x["fpa"], x["bb"]
        features = self._get_features_mat(t_fpa, t_bb)
        target = all_frames.reshape(all_frames.shape[0], -1)
        return features, target, t_fpa, t_bb

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, x: dict, rcond=-1, debug: bool = False):
        """Pixel-wise polynomial regression.

        Args:
            x:     dict with keys 'frames', 'fpa', 'bb'
            debug: if True, plots and saves the fitted surface
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
            rand_pix_idx_tup = np.random.randint(
                [0, 0], self.coefficients.shape[1:], size=2
            )
            rand_pix_idx = np.ravel_multi_index(
                rand_pix_idx_tup, dims=self.coefficients.shape[1:]
            )
            fig, ax = self.plot_data_scatter(x, rand_pix_idx)
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
            ax.legend()
            plt.rcParams.update({'font.size': 22})
            ax.set_title("Fitted Model ($AT_{BB}^4 + BT_{FPA}^2 + CT_{FPA} + D$)")
            fig.savefig('fitted_model.png', dpi=300, bbox_inches='tight')

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fpa_like(array, t_fpa):
        if not isinstance(t_fpa, (np.ndarray, torch.Tensor)):
            t_fpa = np.array([t_fpa])
        elif isinstance(t_fpa, np.ndarray) and t_fpa.ndim == 0:
            t_fpa = t_fpa.reshape(1)
        elif isinstance(t_fpa, torch.Tensor) and t_fpa.ndim == 0:
            t_fpa = t_fpa.view(1)

        reps = len(array.flatten()) // len(t_fpa.flatten())
        if isinstance(t_fpa, np.ndarray):
            t_fpa_rep = t_fpa.repeat(reps)
        elif isinstance(t_fpa, torch.Tensor):
            t_fpa_rep = t_fpa.repeat_interleave(reps)

        return t_fpa_rep.reshape(array.shape)

    def _get_coeffs(self, band: str) -> np.ndarray:
        """Retrieve coefficients for a band, with a clear error if missing."""
        if band not in self._coeff_map:
            available = list(self._coeff_map.keys()) if self._coeff_map else "none loaded"
            raise KeyError(
                f"No coefficients for band '{band}'. "
                f"Available bands: {available}. "
                f"Call load(coeff_dir, wl0, wl1) first."
            )
        return self._coeff_map[band]

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        x_query: Union[np.ndarray, torch.Tensor],
        t_fpa: Union[np.ndarray, torch.Tensor],
        direction: str = "temp_to_rad",
        band: str = "pan",
    ):
        """Apply the model for a specific band.

        Args:
            x_query:   radiometric intensities (rad_to_temp) or BB temps (temp_to_rad)
            t_fpa:     FPA temperature [C]
            direction: 'temp_to_rad' or 'rad_to_temp'
            band:      wavelength string matching a loaded key, e.g. '9um', '11um', 'pan'
        Returns:
            (H, W) array if single image, (B, H, W) if batch
        """
        # Resolve coefficients for this band
        self.coefficients = self._get_coeffs(band)
        coeffs_for_pred = self.coefficients.reshape((self.coefficients.shape[0], -1))

        # Ensure (B, H, W)
        if isinstance(x_query, np.ndarray):
            if x_query.ndim == 2:
                x_query = x_query[None, ...]
        else:
            if x_query.ndim == 2:
                x_query = x_query.unsqueeze(0)

        if isinstance(t_fpa, np.ndarray):
            if t_fpa.ndim == 2:
                t_fpa = t_fpa[None, ...]
        else:
            if t_fpa.ndim == 2:
                t_fpa = t_fpa.unsqueeze(0)

        t_fpa = self._fpa_like(x_query, t_fpa)
        features = self._get_features_mat(t_fpa, x_query)

        if direction == "temp_to_rad":
            t_bb = x_query
            if t_bb.shape[-2:] == self.coefficients.shape[-2:]:
                features_reshaped = features.reshape(
                    (t_bb.shape[0], -1, features.shape[-1])
                )
                rad_hat = (features_reshaped * coeffs_for_pred.T).sum(axis=-1)
            else:
                rad_hat = features @ coeffs_for_pred
            est = rad_hat

        elif direction == "rad_to_temp":
            radiometry = x_query.reshape(x_query.shape[0], -1)
            n_fpa_features = features.shape[1] // 2
            fpa_features = features[:, n_fpa_features:]
            features_reshaped = fpa_features.reshape(
                (radiometry.shape[0], -1, fpa_features.shape[-1])
            )
            num = radiometry - (
                features_reshaped * coeffs_for_pred[n_fpa_features:].T
            ).sum(axis=-1)
            den = (features_reshaped * coeffs_for_pred[:n_fpa_features].T).sum(axis=-1)
            t_bb_hat = k2c((num / den) ** (1 / 4))
            est = t_bb_hat

        else:
            raise ValueError(f"Invalid direction '{direction}'. Use 'temp_to_rad' or 'rad_to_temp'.")

        B = x_query.shape[0]
        est = est.reshape((-1, *self.coefficients.shape[1:]))
        return est[0] if B == 1 else est  # (H,W) or (B,H,W)

    # ------------------------------------------------------------------
    # Validation & Plotting
    # ------------------------------------------------------------------

    def plot_data_scatter(self, x, pix_idx, label="samples", id=None):
        _, radiance, t_fpa, t_bb = self.get_train_features(x)
        if not plt.fignum_exists(id):
            fig = plt.figure(id, figsize=[3.375, 3.375])
            ax = fig.add_subplot(projection="3d")
        else:
            fig = plt.gcf()
            ax = plt.gca()
        t_bb = t_bb[::100]
        t_fpa = t_fpa[::100]
        radiance = radiance[::100]
        ax.scatter(t_bb, t_fpa, radiance[:, pix_idx], label=label, alpha=0.5)
        ax.set_xlabel("$T_\mathit{obj}[C]$")
        ax.set_ylabel("$T_\mathit{int}[C]$")
        ax.set_zlabel("Radiometric Intensity")
        return fig, ax

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
        rmse = np.sqrt((pred_err ** 2).mean())
        if debug:
            y_str = "Radiometric" if direction == "temp_to_rad" else "Temperature"
            fig, ax = plt.subplots()
            ax.hist(pred_err.flatten())
            ax.set_xlabel("Error")
            ax.set_title(
                f"{y_str} Estimation Error "
                f"(mean={pred_err.mean():.2f}, std={pred_err.std():.2f}, rmse={rmse:.2f})"
            )
            fig.savefig(f"pred_err_{direction}.png", dpi=300, bbox_inches='tight')
        return pred_err, rmse

    def to(self, device):
        """Move all loaded coefficient arrays to a torch device."""
        self._coeff_map = {
            wl: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v.to(device)
            for wl, v in self._coeff_map.items()
        }
        if self.coefficients is not None:
            self.coefficients = torch.tensor(self.coefficients).to(device) \
                if isinstance(self.coefficients, np.ndarray) else self.coefficients.to(device)