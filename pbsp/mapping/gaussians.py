"""Resizable Gaussian and semantic-codebook storage derived from EmbodiedSplat.

Upstream attribution and license scope are recorded in THIRD_PARTY_NOTICES.md.
"""

import torch


class EfficientGaussians:
    def __init__(
            self,
            initial_capacity=1000,
            growth_factor=2.0,
            feat_dim=64,
            device='cuda',
            testing=True,
            export_ply=False,
            domain='ensemble'
        ):

        self.capacity = initial_capacity if testing else 0
        self.growth_factor = growth_factor
        self.device = device
        self.testing = testing
        self.size = 0

        # Initial allocations with guessed sizes
        self.means = torch.zeros((1, self.capacity, 3), device=device)  # Adjust the dimensionality as needed
        self.covariances = torch.zeros((1, self.capacity, 3, 3), device=device)  # Example shape
        self.harmonics = torch.zeros((1, self.capacity, 3, 9), device=device)  # Adjust based on actual shape
        self.opacities = torch.zeros((1, self.capacity), device=device)
        self.features = torch.zeros((1, self.capacity, feat_dim), device=device)  # Adjust based on actual shape

        self.domain = domain
        if domain in ['ensemble', '3d']:
            self.clip_features_3d = torch.zeros((1, self.capacity, feat_dim), device=device)
            self.clip_features_2d = {
                "invalid": torch.zeros((1, self.capacity), device=device)
            }
        if domain in ['ensemble', '2d']:
            self.clip_features_2d = {
                "feat": [], # CLIP Global Codebook
                "idx": torch.ones((1, self.capacity, 6), device=device) * -1, # Index cache
                "weight": torch.zeros((1, self.capacity, 6), device=device), # Weight cache
                "invalid": torch.zeros((1, self.capacity), device=device),
                "img_stamp": [0]
            }

        self.coords = torch.zeros((1, self.capacity, 3), device=device)  # Adjust based on actual shape
        self.densities = torch.zeros((1, self.capacity, 1, 1), device=device)  # Adjust based on actual shape
        self.weights = torch.zeros((1, self.capacity, 1, 1), device=device)  # Adjust based on actual shape
        self.extrinsics = torch.zeros((1, self.capacity, 4, 4), device=device)  # Adjust based on actual shape
        self.depths = torch.zeros((1, self.capacity), device=device)  # Adjust based on actual shape
        self.valid = torch.zeros((1, self.capacity), dtype=torch.bool, device=device)  # Track valid entries

        if export_ply:
            self.scales = torch.zeros((1, self.capacity, 3), device=device)  # Example shape
            self.rotations = torch.zeros((1, self.capacity, 4), device=device)  # Example shape

        self.export_ply = export_ply

    def append(
        self,
        means,
        covariances,
        harmonics,
        opacities,
        features,
        clip_features_3d,
        coords,
        densities,
        weights,
        extrinsics,
        depths,
        codebooks_idx,
        codebooks_weights,
        mask=None,
        scales=None,
        rotations=None,
        features_invalid=None,
        fuse=False
    ):

        if mask is not None:
            columns = self.valid[0].nonzero(as_tuple=True)[0][mask]
            self.valid[:, columns] = False
        # Find indices where new data should be placed
        invalid_indices = (~self.valid).nonzero(as_tuple=True)[1]
        num_new = means.shape[1]
        self.size = self.size + num_new

        if num_new > invalid_indices.shape[0]:  # If more new items than invalid slots
            needed_capacity = num_new - invalid_indices.shape[0]
            self._expand_storage(max(self.capacity + needed_capacity, int(self.capacity * self.growth_factor)))
            invalid_indices = (~self.valid).nonzero(as_tuple=True)[1]

        # Assuming new_gaussians is an instance of a class with similar attributes
        self.means[0, invalid_indices[:num_new]] = means
        self.covariances[0, invalid_indices[:num_new]] = covariances
        self.harmonics[0, invalid_indices[:num_new]] = harmonics
        self.opacities[0, invalid_indices[:num_new]] = opacities
        self.coords[0, invalid_indices[:num_new]] = coords
        self.densities[0, invalid_indices[:num_new]] = densities
        if features is not None:
            self.features[0, invalid_indices[:num_new]] = features

            if clip_features_3d is not None:
                self.clip_features_3d[0, invalid_indices[:num_new]] = clip_features_3d

            self.weights[0, invalid_indices[:num_new]] = weights
            self.extrinsics[0, invalid_indices[:num_new]] = extrinsics
            self.depths[0, invalid_indices[:num_new]] = depths

            if codebooks_idx is not None and codebooks_weights is not None:
                size = (self.clip_features_2d["idx"] != -1).sum(dim=-1)[0][invalid_indices[:num_new]]
                filter = size == 6
                size[filter] -= 1

                if fuse:
                    self.clip_features_2d["idx"][0, invalid_indices[:num_new], size] = codebooks_idx
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new]] *= (1 - codebooks_weights)[0, :, None]
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new], size] = codebooks_weights
                else:
                    self.clip_features_2d["idx"][0, invalid_indices[:num_new], size] = codebooks_idx
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new], size] = codebooks_weights

                if (size == 5).sum() > 0:
                    sort, sort_idx = self.clip_features_2d["weight"][0, invalid_indices[:num_new]].sort(dim=-1, descending=True)
                    self.clip_features_2d["weight"][0, invalid_indices[:num_new]] = sort
                    self.clip_features_2d["idx"][0, invalid_indices[:num_new]] = torch.gather(
                        self.clip_features_2d["idx"][0, invalid_indices[:num_new]],
                        index=sort_idx,
                        dim=-1
                    )

        if features_invalid is not None:
            self.clip_features_2d["invalid"][0, invalid_indices[:num_new]] = features_invalid

        if scales is not None:
            self.scales[0, invalid_indices[:num_new]] = scales

        if rotations is not None:
            self.rotations[0, invalid_indices[:num_new]] = rotations

        self.valid[:, invalid_indices[:num_new]] = True

    def _expand_storage(self, new_capacity):
        self.means = self._resize_tensor(self.means, new_capacity, device=self.device)
        self.covariances = self._resize_tensor(self.covariances, new_capacity, device=self.device)
        self.harmonics = self._resize_tensor(self.harmonics, new_capacity, device=self.device)
        self.opacities = self._resize_tensor(self.opacities, new_capacity, device=self.device)
        self.features = self._resize_tensor(self.features, new_capacity, device=self.device)
        self.coords = self._resize_tensor(self.coords, new_capacity, device=self.device)
        self.densities = self._resize_tensor(self.densities, new_capacity, device=self.device)
        self.weights = self._resize_tensor(self.weights, new_capacity, device=self.device)
        self.extrinsics = self._resize_tensor(self.extrinsics, new_capacity, device=self.device)
        self.depths = self._resize_tensor(self.depths, new_capacity, device=self.device)
        self.valid = self._resize_tensor(self.valid, new_capacity, device=self.device, fill=False, dtype=torch.bool)

        if self.domain in ['ensemble', '3d']:
            self.clip_features_3d = self._resize_tensor(self.clip_features_3d, new_capacity, device=self.device)
        if self.domain in ['ensemble', '2d']:
            self.clip_features_2d["idx"] = self._resize_tensor(self.clip_features_2d["idx"], new_capacity, fill=-1, device=self.device)
            self.clip_features_2d["weight"] = self._resize_tensor(self.clip_features_2d["weight"], new_capacity, device=self.device)
        self.clip_features_2d["invalid"] = self._resize_tensor(self.clip_features_2d["invalid"], new_capacity, device=self.device)

        if self.export_ply:
            self.scales = self._resize_tensor(self.scales, new_capacity, device=self.device)
            self.rotations = self._resize_tensor(self.rotations, new_capacity, device=self.device)

        self.capacity = new_capacity

    @staticmethod
    def _resize_tensor(tensor, new_capacity, fill=0, dtype=None, device='cuda'):
        old_size = tensor.size(1)
        dtype = tensor.dtype if dtype is None else dtype
        new_tensor = torch.full((1, new_capacity, *tensor.shape[2:]), fill, dtype=dtype, device=device)
        new_tensor[:, :old_size] = tensor
        return new_tensor
