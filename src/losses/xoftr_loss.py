from loguru import logger

import torch
import torch.nn as nn
from kornia.geometry.conversions import convert_points_to_homogeneous
from kornia.geometry.epipolar import numeric


class XoFTRLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['xoftr']['loss']
        self.pos_w = self.loss_config['pos_weight']
        self.neg_w = self.loss_config['neg_weight']


    def compute_fine_matching_loss(self, data):
        """ Point-wise Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_fine (torch.Tensor): (N, W_f^2, W_f^2)
            conf_matrix_f_gt (torch.Tensor): (N, W_f^2, W_f^2)
            }
        """
        conf_matrix_fine = data['conf_matrix_fine']
        conf_matrix_f_gt = data['conf_matrix_f_gt']

        # Handle empty batch
        if conf_matrix_fine.numel() == 0:
            return torch.tensor(0.0, device=conf_matrix_fine.device, requires_grad=True)

        pos_mask, neg_mask = conf_matrix_f_gt > 0, conf_matrix_f_gt == 0

        conf_matrix_fine = torch.clamp(conf_matrix_fine, 1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        # Compute positive loss
        if pos_mask.any():
            loss_pos = - alpha * torch.pow(1 - conf_matrix_fine[pos_mask], gamma) * (conf_matrix_fine[pos_mask]).log()
            loss_pos = self.pos_w * loss_pos.mean()
        else:
            loss_pos = torch.tensor(0.0, device=conf_matrix_fine.device, requires_grad=True)

        # Compute negative loss
        if neg_mask.any():
            loss_neg = - alpha * torch.pow(conf_matrix_fine[neg_mask], gamma) * (1 - conf_matrix_fine[neg_mask]).log()
            loss_neg = self.neg_w * loss_neg.mean()
        else:
            loss_neg = torch.tensor(0.0, device=conf_matrix_fine.device, requires_grad=True)

        return loss_pos + loss_neg


    def _symmetric_epipolar_distance(self, pts0, pts1, E, K0, K1):
        """Squared symmetric epipolar distance.
        This can be seen as a biased estimation of the reprojection error.
        Args:
            pts0 (torch.Tensor): [N, 2]
            E (torch.Tensor): [3, 3]
        """
        pts0 = (pts0 - K0[:, [0, 1], [2, 2]]) / K0[:, [0, 1], [0, 1]]
        pts1 = (pts1 - K1[:, [0, 1], [2, 2]]) / K1[:, [0, 1], [0, 1]]
        pts0 = convert_points_to_homogeneous(pts0)
        pts1 = convert_points_to_homogeneous(pts1)

        Ep0 = (pts0[:, None, :] @ E.transpose(-2, -1)).squeeze(1)  # [N, 3]
        p1Ep0 = torch.sum(pts1 * Ep0, -1)  # [N,]
        Etp1 = (pts1[:, None, :] @ E).squeeze(1)  # [N, 3]

        d = p1Ep0 ** 2 * (1.0 / (Ep0[:, 0] ** 2 + Ep0[:, 1] ** 2 + 1e-9) + 1.0 / (
                    Etp1[:, 0] ** 2 + Etp1[:, 1] ** 2 + 1e-9))  # N
        return d

    def compute_sub_pixel_loss(self, data):
        """ symmetric epipolar distance loss.
        Args:
        data (dict): {
            m_bids (torch.Tensor): (N)
            T_0to1 (torch.Tensor): (B, 4, 4)
            mkpts0_f_train (torch.Tensor): (N, 2)  [Optional - may not exist]
            mkpts1_f_train (torch.Tensor): (N, 2)  [Optional - may not exist]
            }
        """

        # Early return if no fine matches were generated
        if 'mkpts0_f_train' not in data or 'mkpts1_f_train' not in data:
            return torch.tensor(0.0, device=data['image0'].device, requires_grad=True)

        pts0 = data['mkpts0_f_train']
        pts1 = data['mkpts1_f_train']

        # Early return if matches are empty
        if len(pts0) == 0 or len(pts1) == 0:
            return torch.tensor(0.0, device=data['image0'].device, requires_grad=True)

        Tx = numeric.cross_product_matrix(data['T_0to1'][:, :3, 3])
        E_mat = Tx @ data['T_0to1'][:, :3, :3]

        m_bids = data['m_bids']

        sym_dist = self._symmetric_epipolar_distance(pts0, pts1, E_mat[m_bids], data['K0'][m_bids], data['K1'][m_bids])

        # Filter matches with high epipolar error (only train approximately correct fine-level matches)
        loss = sym_dist[sym_dist < 1e-4]
        if len(loss) == 0:
            return torch.tensor(0.0, device=sym_dist.device, requires_grad=True)

        return loss.mean()

    def compute_coarse_loss(self, data, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
        data (dict): {
            conf_matrix_0_to_1 (torch.Tensor): (N, HW0, HW1)
            conf_matrix_1_to_0 (torch.Tensor): (N, HW0, HW1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            }
            weight (torch.Tensor): (N, HW0, HW1)
        """

        conf_matrix_0_to_1 = data["conf_matrix_0_to_1"]
        conf_matrix_1_to_0 = data["conf_matrix_1_to_0"]
        conf_gt = data["conf_matrix_gt"]

        # Handle empty batch
        if conf_gt.numel() == 0:
            return torch.tensor(0.0, device=conf_gt.device, requires_grad=True)

        pos_mask = conf_gt == 1

        conf_matrix_0_to_1 = torch.clamp(conf_matrix_0_to_1, 1e-6, 1 - 1e-6)
        conf_matrix_1_to_0 = torch.clamp(conf_matrix_1_to_0, 1e-6, 1 - 1e-6)
        alpha = self.loss_config['focal_alpha']
        gamma = self.loss_config['focal_gamma']

        if pos_mask.any():
            loss_pos = - alpha * torch.pow(1 - conf_matrix_0_to_1[pos_mask], gamma) * (
            conf_matrix_0_to_1[pos_mask]).log()
            loss_pos += - alpha * torch.pow(1 - conf_matrix_1_to_0[pos_mask], gamma) * (
            conf_matrix_1_to_0[pos_mask]).log()
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
            loss_c = self.pos_w * loss_pos.mean()
        else:
            loss_c = torch.tensor(0.0, device=conf_gt.device, requires_grad=True)

        return loss_c

    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight

    def forward(self, data):
        """
        Update:
            data (dict): update{
                'loss': [1] the reduced loss across a batch,
                'loss_scalars' (dict): loss scalars for tensorboard_record
            }
        """
        loss_scalars = {}
        # 0. compute element-wise loss weight
        c_weight = self.compute_c_weight(data)

        # 1. coarse-level loss
        # Use explicit assignment instead of *= to avoid in-place leaf modification
        loss_c_raw = self.compute_coarse_loss(data, weight=c_weight)
        loss_c = loss_c_raw * self.loss_config['coarse_weight']

        loss = loss_c
        loss_scalars.update({"loss_c": loss_c.clone().detach().cpu()})

        # 2. fine-level matching loss for windows
        loss_f_match_raw = self.compute_fine_matching_loss(data)
        loss_f_match = loss_f_match_raw * self.loss_config['fine_weight']

        loss = loss + loss_f_match
        loss_scalars.update({"loss_f": loss_f_match.clone().detach().cpu()})

        # 3. sub-pixel refinement loss
        loss_sub_raw = self.compute_sub_pixel_loss(data)
        loss_sub = loss_sub_raw * self.loss_config['sub_weight']

        loss = loss + loss_sub
        loss_scalars.update({"loss_sub": loss_sub.clone().detach().cpu()})

        loss_scalars.update({'loss': loss.clone().detach().cpu()})
        data.update({"loss": loss, "loss_scalars": loss_scalars})
