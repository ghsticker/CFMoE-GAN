import os
import torch
import torch.nn as nn
import torch.nn.functional as F


class FrozenDinoV3Encoder(nn.Module):
    def __init__(
        self,
        repo,
        model_name,
        input_size=224,
        patch_size=16,
        use_hub_pretrained=False,
        pretrained_path="",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.repo = repo
        self.model_name = model_name
        self.input_size = input_size
        self.patch_size = patch_size
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)

        if use_hub_pretrained:
            local_repo = os.path.expanduser(repo)
            if os.path.isdir(local_repo):
                self.model = torch.hub.load(local_repo, model_name, pretrained=True, source="local")
            else:
                self.model = torch.hub.load(repo, model_name, pretrained=True)
        else:
            if not pretrained_path:
                raise ValueError("dino_pretrained_path is required when use_hub_pretrained is False.")
            local_repo = os.path.expanduser(repo)
            if os.path.isdir(local_repo):
                self.model = torch.hub.load(local_repo, model_name, pretrained=False, source="local")
            else:
                self.model = torch.hub.load(repo, model_name, pretrained=False)
            state = torch.load(pretrained_path, map_location="cpu")
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            self.model.load_state_dict(state, strict=False)

        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.embed_dim = self._infer_embed_dim()

    def _infer_embed_dim(self):
        if hasattr(self.model, "embed_dim"):
            return int(self.model.embed_dim)
        for attr in ("num_features", "hidden_dim"):
            if hasattr(self.model, attr):
                return int(getattr(self.model, attr))
        name = (self.model_name or "").lower()
        if "vitg" in name:
            return 1536
        if "vitl" in name:
            return 1024
        if "vitb" in name:
            return 768
        if "vits" in name:
            return 384
        return None

    def _prepare(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        if x.min() < 0:
            x = (x + 1) / 2
        if self.input_size is not None and (x.shape[-1] != self.input_size or x.shape[-2] != self.input_size):
            x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        mean = self.mean.to(x.device, dtype=x.dtype)
        std = self.std.to(x.device, dtype=x.dtype)
        return (x - mean) / std

    def _extract_patch_tokens(self, feats):
        if isinstance(feats, dict):
            if "x_norm_patchtokens" in feats:
                return feats["x_norm_patchtokens"]
            if "patch_tokens" in feats:
                return feats["patch_tokens"]
            if "x" in feats and feats["x"].dim() == 3:
                feats = feats["x"]
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        if feats.dim() == 3 and feats.shape[1] > 1:
            # Some backbones return [CLS + patch_tokens], while some already return patch_tokens only.
            # Only drop the first token when token count exactly matches expected_cls_plus_patches.
            expected_patches = (self.input_size // self.patch_size) ** 2
            if feats.shape[1] == expected_patches + 1:
                return feats[:, 1:, :]
            return feats
        return feats

    def forward(self, x):
        x = self._prepare(x)
        if hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
        else:
            feats = self.model(x)
        tokens = self._extract_patch_tokens(feats)
        if self.embed_dim is None and tokens is not None:
            self.embed_dim = int(tokens.shape[-1])
        return tokens

    def forward_intermediate_layers(self, x, layers):
        x = self._prepare(x)
        if hasattr(self.model, "get_intermediate_layers"):
            try:
                feats = self.model.get_intermediate_layers(x, n=layers, return_class_token=False)
            except TypeError:
                try:
                    feats = self.model.get_intermediate_layers(x, n=max(layers))
                except TypeError:
                    feats = self.model.get_intermediate_layers(x)
        elif hasattr(self.model, "forward_features"):
            feats = self.model.forward_features(x)
            if isinstance(feats, dict) and "intermediate_layers" in feats:
                feats = feats["intermediate_layers"]
        else:
            feats = None

        if feats is None:
            return [self.forward(x)]

        if isinstance(feats, torch.Tensor):
            feats = [feats]

        if isinstance(feats, dict) and "intermediate_layers" in feats:
            feats = feats["intermediate_layers"]

        if not isinstance(feats, (list, tuple)):
            feats = [feats]

        patch_tokens = [self._extract_patch_tokens(layer) for layer in feats]
        return patch_tokens

    @property
    def grid_size(self):
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            grid = self.model.patch_embed.grid_size
            if isinstance(grid, tuple):
                return grid
            return (grid, grid)
        size = self.input_size // self.patch_size
        return (size, size)
