import numpy as np
import cv2

from src.dataset.stats.distribution_manager import DatasetStats
from src.configs.poftr_configs import get_K_from_cfg, scale_sim

class SensorViewSimulator:

    def __init__(self, config, sim_config, sim_scale=None, wls=None):
        self.config = config
        self.wls = wls if wls is not None else config["data"]["wls"][:2]
        self.sim_config = sim_config
        self.sim_scale = sim_scale
        self.sim_level = sim_config['sim_level'] if sim_config else None
        self.image_shape = self.config['data']['image_shape']
        self.sample_shape = self.config['data']['sample_shape']
        self.K = get_K_from_cfg(self.config) # Camera intrinsic matrix
        self.K_inv = np.linalg.inv(self.K)  # Inverse of the intrinsic matrix
        self.sim_params = {
            'tx0': 0.0,
            'ty0': 0.0,
            'tx1': 0.0,
            'ty1': 0.0,
            'tz0': 0.0,
            'tz1': 0.0,
            'yaw0': 0.0,
            'yaw1': 0.0,
            'pitch0': 0.0,
            'pitch1': 0.0,
            'roll0': 0.0,
            'roll1': 0.0,
        }
        self.sim_level = None
        self.alt = None
        self.H0 = np.eye(3, dtype=np.float32)
        self.H1 = np.eye(3, dtype=np.float32)
        self.H0to1 = np.eye(3, dtype=np.float32)
        self.T_0to1 = np.eye(4, dtype=np.float32)
        self.R0_rel = np.eye(3, dtype=np.float32)
        self.R1_rel = np.eye(3, dtype=np.float32)
        self.R_a = np.diag([1, -1, -1]) # view A
        self.t_a = np.array([0, 0, 0], dtype=np.float32)
        self.R_a_inv = np.linalg.inv(self.R_a)
        self.co_visibility = None
        self.valid_pixels = [None, None]

        self.valid_view = False

    @classmethod
    def generate_view(cls, sample, sim_scale=None, wls=None):
        """
        Generates a new simulated sensor view by applying fresh transformations.

        This method is used when creating new simulated views, generating random
        transformation parameters based on the configuration.

        Args:
            sample (MonoSample): The sample object to transform
            sim_scale (float): Scale factor for simulation distribution manager

        Returns:
            MonoSample: The transformed sample with newly generated parameters
        """

        obj = cls(sample.config, sample.sim_config, sim_scale, wls)
        obj.load_generation_params(sample)
        obj.generate_sim_params()
        obj.compose_sensor_transforms()
        obj.warp_images_and_priors(sample)
        obj.warp_masks(sample)
        obj.compute_depths(sample)
        sample_dict = obj._build_sample_dict()
        sample_stats = DatasetStats.get_sample_stats(sample_dict)
        obj.co_visibility = sample_stats['co_visibility']
        obj.valid_pixels = sample_stats['valid_pixels']
        #obj.valid_view = obj.test_sample_stats(sample_stats)
        obj.update_sample_params(sample)

    @classmethod
    def load_view(cls, sample):
        """
        Loads and applies a previously generated sensor view transformation.

        This method is used when recreating a specific view using stored
        transformation parameters, useful for reproducibility or validation.

        Args:
            sample (MonoSample): The sample object containing stored transformation
                parameters (sim_params, H0, H1)

        Returns:
            MonoSample: The transformed sample using stored parameters
        """
        obj = cls(sample.config)
        obj.load_sample_params(sample)
        obj.compose_sensor_transforms()
        obj.warp_images_and_priors(sample)
        obj.warp_masks(sample)
        obj.compute_depths(sample)
        obj.update_sample_params(sample)
        return obj

    def generate_sim_params(self):
        """Generate random simulation parameters based on the configuration."""
        if self.sim_config["dataset_mode"]:
            self._generate_sim_params_dataset_mode()
        else:
            self._generate_sim_params_simulation_mode()

    def _generate_sim_params_dataset_mode(self):
        sim_scale = self.sim_scale if self.sim_scale is not None else 1.0
        sim_config_copy = self.config['sim'].copy()
        sim_config_copy = scale_sim(sim_config_copy, sim_scale)

        param_to_std_key_map = {
            'tx0': 'tx_std',
            'ty0': 'ty_std',
            'tx1': 'tx_std',
            'ty1': 'ty_std',
            'tz0': 'tz_std',
            'tz1': 'tz_std',
            'pitch0': 'pitch_std',
            'pitch1': 'pitch_std',
            'roll0': 'roll_std',
            'roll1': 'roll_std',
            'yaw0': 'yaw_std',
            'yaw1': 'yaw_std',
        }

        new_params = {
            param_name: np.random.randn() * sim_config_copy[std_key]
            for param_name, std_key in param_to_std_key_map.items()
        }

        self.sim_params.update(new_params)

    def _generate_sim_params_simulation_mode(self):

        # 1. Define the parameter-to-std-key mappings
        # These are always generated, but the rotation method changes.

        translation_map = {
            'tx0': 'tx_std',
            'ty0': 'ty_std',
            'tx1': 'tx_std',
            'ty1': 'ty_std',
            'tz0': 'tz_std',
            'tz1': 'tz_std',
        }

        rotation_map = {
            'pitch0': 'pitch_std',
            'pitch1': 'pitch_std',
            'roll0': 'roll_std',
            'roll1': 'roll_std',
            'yaw0': 'yaw_std',
            'yaw1': 'yaw_std',
        }

        # 2. Initialize the dictionary for new parameters
        # Generate translation params (always using randn)
        new_params = {
            param: np.random.randn() * self.sim_config[std_key]
            for param, std_key in translation_map.items()
        }

        # 3. Conditionally generate rotation params

        # Use dictionary access for sim_level as requested
        if self.sim_config["sim_level"] != 3:
            # Standard simulation: use Gaussian distribution (randn)
            rotation_params = {
                param: np.random.randn() * self.sim_config[std_key]
                for param, std_key in rotation_map.items()
            }
        else:
            # Special case (level 3): use Uniform distribution
            rotation_params = {
                param: np.random.uniform(-self.sim_config[std_key], self.sim_config[std_key])
                for param, std_key in rotation_map.items()
            }

        # 4. Add the rotation params to the main dictionary
        new_params.update(rotation_params)

        # 5. Apply the single update to the instance state
        self.sim_params.update(new_params)


    def load_sample_params(self, sample):
        self.sim_params = sample.sim_params
        # self.lt = sample.lt
        self.sample_shape = sample.sample_shape
        self.sim_level = sample.sim_level
        self.alt = sample.alt
        self.t_a = np.array([0, 0, self.alt], dtype=np.float32)

    def load_generation_params(self, sample):
        self.sample_shape = sample.sample_shape
        self.sim_level = sample.sim_level
        self.alt = sample.alt

    def update_sample_params(self, sample):
        sample.H0 = self.H0
        sample.H1 = self.H1
        sample.sim_params = self.sim_params
        sample.T_0to1 = self.T_0to1
        sample.T_1to0 = np.linalg.inv(self.T_0to1)
        sample.image0 = self.image0
        sample.image1 = self.image1
        sample.phys0 = self.phys0
        sample.phys1 = self.phys1
        sample.depth0 = self.depth0
        sample.depth1 = self.depth1
        sample.mask0 = self.mask0
        sample.mask1 = self.mask1
        sample.co_visibility = self.co_visibility
        sample.valid_pixels = self.valid_pixels

    def rot_y(self):
        """Build pitch rotation matrices based on the simulation parameters."""

        def pitch_matrix(pitch):
            """Create a pitch rotation matrix for a given pitch angle."""
            theta = np.radians(pitch)
            return np.array([
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)]
            ], dtype=np.float32)

        pitch0, pitch1 = self.sim_params['pitch0'], self.sim_params['pitch1']
        return pitch_matrix(pitch0), pitch_matrix(pitch1)


    def rot_x(self):
        """Build roll rotation matrices (around X-axis)."""

        def roll_matrix(roll_deg):
            """Create a 3D roll rotation matrix (around X-axis)."""
            theta = np.radians(roll_deg)
            return np.array([[1, 0, 0],
                   [0,np.cos(theta),-np.sin(theta)],
                   [0,np.sin(theta),np.cos(theta)]],
                   dtype=np.float32)

        roll0, roll1 = self.sim_params['roll0'], self.sim_params['roll1']
        return roll_matrix(roll0), roll_matrix(roll1)


    def rot_z(self):
        """Build yaw rotation matrices (around Z-axis)."""
        yaw0, yaw1 = self.sim_params['yaw0'], self.sim_params['yaw1']

        def yaw_matrix(yaw_deg):
            theta = np.radians(yaw_deg)
            return np.array([
                [np.cos(theta), -np.sin(theta), 0],
                [np.sin(theta), np.cos(theta), 0],
                [0, 0, 1]
            ], dtype=np.float32)

        return yaw_matrix(yaw0), yaw_matrix(yaw1)

    def build_rotation_matrices(self):
        # image0, image1 - 2 different images before rotation
        R_yaw0, R_yaw1 = self.rot_z() # yaw - around Z-axis
        R_pitch0, R_pitch1 = self.rot_y() # pitch - around Y-axis
        R_roll0, R_roll1 = self.rot_x() # roll - around X-axis

        self.R0_rel= R_yaw0 @ R_pitch0 @ R_roll0 # roll -> pitch -> yaw
        self.R1_rel = R_yaw1 @ R_pitch1 @ R_roll1


    def build_translation_vectors(self):
        # view a - before, view b - after
        # image0, image1 - images before translation

        # camera centers before translation
        c0_a = np.array([0, 0, self.alt], dtype=np.float32) # of image 0
        c1_a = np.array([0, 0, self.alt], dtype=np.float32) # of image 1

        # calc simulated translations
        x0, x1 = self.sim_params['tx0'], self.sim_params['tx1'] # simulated translations [m]
        y0, y1 = self.sim_params['ty0'], self.sim_params['ty1']
        z0, z1 = self.sim_params['tz0'], self.sim_params['tz1']

        # camera centers after translation
        c0_b = c0_a + np.array([x0, y0, z0], dtype=np.float32) # of image 0
        c1_b = c1_a + np.array([x1, y1, z1], dtype=np.float32) # of image 1

        # translation vectors
        self.t0_rel = self.R_a_inv @ (c0_b - c0_a)
        self.t1_rel = self.R_a_inv @ (c1_b - c1_a)

    def compute_extrinsics(self):
        R_0to1 = self.R1_rel @ self.R0_rel.T
        t_0to1 = self.t1_rel - (R_0to1 @ self.t0_rel)
        T_0to1 = np.eye(4, dtype=np.float32)
        T_0to1[:3, :3] = R_0to1
        T_0to1[:3, 3] = t_0to1
        self.T_0to1 = T_0to1
        self.T_1to0 = np.linalg.inv(T_0to1)


    def build_homography(self):
        # misc
        n = self.R_a_inv @ np.array([0, 0, 1], dtype=np.float32)
        h = self.alt
        t0 = self.t0_rel
        t1 = self.t1_rel
        K_inv = np.linalg.inv(self.K)

        # Build the homography matrices
        self.H0 = self.K @ (self.R0_rel - np.outer(t0, n) / h) @ K_inv
        self.H1 = self.K @ (self.R1_rel - np.outer(t1, n) / h) @ K_inv
        self.H0to1 = self.H1 @ np.linalg.inv(self.H0)



    def compose_sensor_transforms(self):
        """
        Compose all sensor transformation matrices and find a valid top-left patch position.

        Atthention: H0 and H1 are not the final homographies, they are built for full images.

        """

        # 1. Build rotations (roll, pitch, yaw - 3 DOf)
        self.build_rotation_matrices() # R0, R1

        # 2. Simulate translatios
        self.build_translation_vectors() # t0, t1

        # 3. Compose the final transformation matrices
        self.build_homography() # H0, H1, H0to1

        # 4. Compute extrinsic matrix
        self.compute_extrinsics()


    def warp_images_and_priors(self, sample):
        h, w = self.image_shape
        self.image0 = cv2.warpPerspective(sample.wl0_img, self.H0, (w, h), flags=cv2.INTER_NEAREST)
        self.phys0 = cv2.warpPerspective(sample.phys0, self.H0, (w, h), flags=cv2.INTER_NEAREST)
        self.image1 = cv2.warpPerspective(sample.wl1_img, self.H1, (w, h), flags=cv2.INTER_NEAREST)
        self.phys1 = cv2.warpPerspective(sample.phys1, self.H1, (w, h), flags=cv2.INTER_NEAREST)

    def warp_masks(self, sample):
        h, w = self.image_shape
        ones = np.ones((h, w), dtype=np.uint8)
        self.mask0 = cv2.warpPerspective(ones, self.H0, (w, h), flags=cv2.INTER_NEAREST)
        self.mask1 = cv2.warpPerspective(ones, self.H1, (w, h), flags=cv2.INTER_NEAREST)


    def compute_depths(self, sample):
        self.depth0 = self._compute_depth_map(sample, idx=0)
        self.depth1 = self._compute_depth_map(sample, idx=1)


    def _compute_depth_map(self, sample, idx):
        depth_map = self._compute_flat_earth_depth(idx=idx)
        # if self.sim_params.use_dem:
        #     dem_map = self._compute_dem_map(sample, idx=idx)
        #     depth_map  = self._combine_depths(depth_map, dem_map)
        #     depth_map += dem_map
        return depth_map


    def _compute_flat_earth_depth(self, idx):
        """
        Returns:
          depth_map : (H, W) array of per-pixel depth along camera Z (meters),
                      with np.nan where the ray doesn’t intersect z=0.
        """
        # 1) pick extrinsics for Cam A (nadir) and Cam B (relative)
        R_a = self.R_a
        t_a = self.t_a + np.array([0, 0, self.alt], dtype=np.float32)
        if idx == 0:
            R_rel, t_rel = self.R0_rel, self.t0_rel
            mask = self.mask0
        else:
            R_rel, t_rel = self.R1_rel, self.t1_rel
            mask = self.mask1

        # 2) compose world→CamB
        R_b = R_rel @ R_a
        t_b = R_rel @ t_a + t_rel

        # 3) back-project pixel grid into world rays
        h, w = self.image_shape
        ys, xs = np.indices((h, w), dtype=float)
        # shoot through pixel centers:
        xs += 0.5
        ys += 0.5
        pix = np.stack([xs.ravel(), ys.ravel(), np.ones(h * w)], axis=0)

        dirs_camA = self.K_inv @ pix  # in CamA frame
        dirs_world = R_a.T @ dirs_camA  # in world frame

        # 4) intersect each ray t_a + λ·d_w with the plane z=0:
        dz = dirs_world[2]
        # λ = -t_a[z] / d_w[z]
        lam = -t_a[2] / dz

        # where rays never hit (parallel or backward), mark as invalid
        valid_plane = ((dz != 0) & (lam > 0)).reshape(h, w)

        # compute 3D intersections
        X = t_a[:, None] + dirs_world * lam[None, :]

        # 5) re-project into CamB and read off Z
        P_camB = R_b @ (X - t_b[:, None])
        depth_flat = P_camB[2].reshape(h, w)

        # 6) mask out invalid pixels
        valid = valid_plane & mask & (depth_flat > 0)
        depth_flat[~valid] = np.nan

        if idx == 0:
            self.mask0 = valid
        else:
            self.mask1 = valid

        return depth_flat

    def test_sample_stats(self, sample_stats: dict) -> bool:
        co_visibility = sample_stats['co_visibility']
        valid_pixels = sample_stats['valid_pixels']
        if any(valid_pixels) == 0:
            print(f"Invalid view, sim_level:{self.sim_level}, params: {self.sim_params}, valid_pixels: {valid_pixels}")
            return False
        if co_visibility < 0.1:
            print(f"Warning: low kpts - sim_level:{self.sim_level}, params: {self.sim_params}, corr_kpts: {co_visibility}")
        return True

    def _build_sample_dict(self):
        return {
            'image0': self.image0,
            'image1': self.image1,
            'mask0': self.mask0,
            'mask1': self.mask1,
            'phys0': self.phys0,
            'phys1': self.phys1,
            'depth0': self.depth0,
            'depth1': self.depth1,
            'T_0to1': self.T_0to1,
            'H0': self.H0,
            'H1': self.H1,
        }