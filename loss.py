"""
loss.py - Multi-layer perceptual L1 loss using discriminator features.

Computes L1 distance between hierarchical discriminator features of
(image × predicted_mask) and (image × ground_truth_mask), encouraging
the generator to produce perceptually realistic segmentations.

Usage:
    from loss import L1Loss
    criterion = L1Loss(discriminator, num_layers=3, num_training_images=10)
    loss = criterion(images, predicted_masks, ground_truth_masks)
"""

import torch
import torch.nn as nn


class L1Loss(nn.Module):
    """
    Multi-layer perceptual L1 loss.

    Uses the discriminator's hierarchical features to compute L1 distance
    between masked predictions and masked ground truth at multiple network depths.

    Parameters
    ----------
    critic_network : nn.Module
        Discriminator network with a `forward_all_layers` method.
    num_layers : int
        Number of layers in the discriminator to extract features from.
    num_training_images : int
        Batch size (kept for API compatibility).
    """

    def __init__(self, critic_network, num_layers, num_training_images):
        super(L1Loss, self).__init__()
        self.critic_network = critic_network
        self.num_layers = num_layers
        self.num_training_images = num_training_images

    def forward(self, original_images, predicted_labels, ground_truth_labels):
        """
        Compute the multi-layer perceptual L1 loss.

        Parameters
        ----------
        original_images : torch.Tensor
            Input CT images of shape (B, 1, H, W).
        predicted_labels : torch.Tensor
            Predicted segmentation masks of shape (B, 1, H, W).
        ground_truth_labels : torch.Tensor
            Ground truth segmentation masks of shape (B, 1, H, W).

        Returns
        -------
        torch.Tensor
            Average L1 loss across all discriminator layers.
        """
        # Single forward pass for each masked input
        features_predicted = self.critic_network.forward_all_layers(
            original_images * predicted_labels
        )
        features_ground_truth = self.critic_network.forward_all_layers(
            original_images * ground_truth_labels
        )

        # Accumulate L1 loss across layers
        total_loss = 0.0
        for feat_pred, feat_gt in zip(features_predicted, features_ground_truth):
            flat_pred = torch.flatten(feat_pred, start_dim=1)
            flat_gt = torch.flatten(feat_gt, start_dim=1)
            total_loss += (flat_pred - flat_gt).abs().mean()

        return total_loss / self.num_layers
