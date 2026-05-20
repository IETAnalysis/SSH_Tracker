import torch
import torch.nn as nn
import torch.nn.functional as F
import config

class DynamicResidualGRUBlock(nn.Module):
    def __init__(self, in_dim, hidden_size, is_bidirectional, num_layers):
        super().__init__()
        self.gru = nn.GRU(
            input_size=in_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=is_bidirectional
        )
        self.out_dim = hidden_size * 2 if is_bidirectional else hidden_size
        self.proj = nn.Linear(in_dim, self.out_dim)
        self.attention = nn.Sequential(
            nn.Linear(self.out_dim * 2, self.out_dim),
            nn.Tanh(),
            nn.Linear(self.out_dim, self.out_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        h, _ = self.gru(x)
        x_proj = self.proj(x)
        concat_h = torch.cat([x_proj, h], dim=-1)
        alpha = self.attention(concat_h)
        out = h + alpha * x_proj
        return out

class FeatureFusionAttention(nn.Module):
    def __init__(self, total_dim, reduction):
        super().__init__()
        mid_dim = max(total_dim // reduction, 16)
        self.attention = nn.Sequential(
            nn.Linear(total_dim, mid_dim),
            nn.ReLU(),
            nn.Linear(mid_dim, total_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        weights = self.attention(x)
        return x * weights

class CascadedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.blocks = nn.ModuleList()
        current_dim = config.FEATURE_DIM
        self.out_dims = []
        block_num_layers = getattr(config, 'GRU_NUM_LAYERS')

        for hidden_size in config.GRU_HIDDEN_SIZES:
            block = DynamicResidualGRUBlock(
                in_dim=current_dim,
                hidden_size=hidden_size,
                is_bidirectional=config.RNN_BIDIRECTIONAL,
                num_layers=block_num_layers
            )
            self.blocks.append(block)
            current_dim = block.out_dim
            self.out_dims.append(current_dim)

        self.total_dim = sum(self.out_dims)
        self.fusion_attention = FeatureFusionAttention(
            total_dim=self.total_dim,
            reduction=getattr(config, 'ATTENTION_REDUCTION')
        )

    def forward(self, x: torch.Tensor):
        h_current = x
        all_h = []
        for block in self.blocks:
            h_current = block(h_current)
            all_h.append(h_current)
        h_concat = torch.cat(all_h, dim=-1)
        h_fused = self.fusion_attention(h_concat)
        return h_fused

class NeuralSplitterFeatureExtractor(nn.Module):
    def __init__(self, in_channels, windows):
        super().__init__()
        self.windows = windows

    def _compute_window_stats(self, x, w, direction):
        B, C, L = x.shape
        if direction == 'left':
            x_pad = F.pad(x, (w, 0))[:, :, :-1]
        else:
            x_pad = F.pad(x, (0, w))[:, :, 1:]

        mean_val = F.avg_pool1d(x_pad, kernel_size=w, stride=1)
        max_val = F.max_pool1d(x_pad, kernel_size=w, stride=1)

        x_sq_pad = x_pad ** 2
        mean_sq_val = F.avg_pool1d(x_sq_pad, kernel_size=w, stride=1)
        var_val = torch.clamp(mean_sq_val - mean_val ** 2, min=1e-5)
        std_val = torch.sqrt(var_val)

        return torch.cat([mean_val, max_val, std_val], dim=1)

    def forward(self, x):
        x = x.transpose(1, 2)
        multi_scale_features = []
        for w in self.windows:
            left_stats = self._compute_window_stats(x, w, direction='left')
            right_stats = self._compute_window_stats(x, w, direction='right')
            abs_diff = torch.abs(right_stats - left_stats)
            scale_feat = torch.cat([left_stats, right_stats, abs_diff], dim=1)
            multi_scale_features.append(scale_feat)
        final_features = torch.cat(multi_scale_features, dim=1)
        return final_features.transpose(1, 2)

class BoundaryDetectionBranch(nn.Module):
    def __init__(self, gru_dim):
        super().__init__()
        self.extractor = NeuralSplitterFeatureExtractor(
            in_channels=gru_dim,
            windows=config.MULTI_SCALE_WINDOWS
        )
        feature_dim = len(config.MULTI_SCALE_WINDOWS) * 9 * gru_dim
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, getattr(config, 'P1_HIDDEN_DIM')),
            nn.ReLU(),
            nn.Dropout(getattr(config, 'P1_DROPOUT')),
            nn.Linear(getattr(config, 'P1_HIDDEN_DIM'), 1)
        )

    def forward(self, gru_output):
        features = self.extractor(gru_output)
        logits = self.classifier(features)
        return logits.squeeze(-1)

class BusinessClassificationBranch(nn.Module):
    def __init__(self, gru_dim, num_classes):
        super().__init__()
        self.pooled_dim = gru_dim * 4
        self.classifier = nn.Sequential(
            nn.Linear(self.pooled_dim, getattr(config, 'P2_HIDDEN_DIM_1')),
            nn.LayerNorm(getattr(config, 'P2_HIDDEN_DIM_1')),
            nn.ReLU(),
            nn.Dropout(getattr(config, 'P2_DROPOUT_1')),
            nn.Linear(getattr(config, 'P2_HIDDEN_DIM_1'), getattr(config, 'P2_HIDDEN_DIM_2')),
            nn.ReLU(),
            nn.Dropout(getattr(config, 'P2_DROPOUT_2')),
            nn.Linear(getattr(config, 'P2_HIDDEN_DIM_2'), num_classes)
        )

    def forward(self, chunk_tensor):
        feat_mean = chunk_tensor.mean(dim=0)
        feat_max = chunk_tensor.max(dim=0)[0]
        feat_min = chunk_tensor.min(dim=0)[0]
        feat_std = chunk_tensor.std(dim=0, unbiased=False) + 1e-6
        fused_feat = torch.cat([feat_mean, feat_max, feat_min, feat_std], dim=0)
        return self.classifier(fused_feat.unsqueeze(0))

class CascadedTracker(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = CascadedEncoder()
        total_dim = sum(self.encoder.out_dims)
        self.phase1 = BoundaryDetectionBranch(gru_dim=total_dim)
        self.phase2 = BusinessClassificationBranch(
            gru_dim=total_dim,
            num_classes=config.NUM_BEHAVIORS
        )