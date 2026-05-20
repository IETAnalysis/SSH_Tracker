import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import config

class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg, gamma_pos, clip):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, x, y):
        p = torch.sigmoid(x)
        xs_pos = p
        xs_neg = (1 - p + self.clip).clamp(max=1)

        los_pos = y * torch.log(xs_pos.clamp(min=1e-8))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=1e-8))
        loss = los_pos + los_neg

        pt = p * y + (1 - p) * (1 - y)
        w = torch.pow(1 - pt, self.gamma_pos * y + self.gamma_neg * (1 - y))

        return -(loss * w).mean()

def train_step_phase1(model_p1, gru_output, batch_split_indices, device):
    B, L, _ = gru_output.shape
    labels = torch.zeros((B, L), dtype=torch.float32, device=device)
    for b in range(B):
        for sp in batch_split_indices[b]:
            start_idx = max(0, sp - config.BND_TOLERANCE)
            end_idx = min(L, sp + config.BND_TOLERANCE + 1)
            labels[b, start_idx:end_idx] = 1.0

    logits = model_p1(gru_output)

    criterion = AsymmetricLoss(
        gamma_neg=config.ASL_GAMMA_NEG,
        gamma_pos=config.ASL_GAMMA_POS,
        clip=config.ASL_CLIP
    )
    return criterion(logits, labels)

def train_step_phase2(model_p2, gru_output, batch_split_indices, batch_flow_labels, device):
    B, L, _ = gru_output.shape
    logits_list = []
    chunk_labels_list = []

    for b in range(B):
        boundaries = [0] + batch_split_indices[b] + [L]
        labels = batch_flow_labels[b]
        for i in range(len(boundaries) - 1):
            start_idx, end_idx = boundaries[i], boundaries[i + 1]
            if end_idx - start_idx < getattr(config, 'MIN_SEGMENT_LENGTH'):
                continue

            chunk_tensor = gru_output[b, start_idx:end_idx, :]
            chunk_logit = model_p2(chunk_tensor)

            logits_list.append(chunk_logit)
            chunk_labels_list.append(labels[i])

    if len(logits_list) == 0:
        return torch.tensor(0.0, requires_grad=True, device=device)

    batch_logits = torch.cat(logits_list, dim=0)
    batch_chunk_labels = torch.tensor(chunk_labels_list, dtype=torch.long, device=device)

    return F.cross_entropy(batch_logits, batch_chunk_labels)

def train_cascaded_model(model, train_loader, test_loader, device, save_path):
    print(f"\n[INFO] Starting end-to-end joint training...")
    optimizer = optim.Adam(model.parameters(), lr=config.LR)

    best_loss, patience_cnt = float('inf'), 0
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss, n = 0.0, 0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch + 1}/{config.EPOCHS}] Train", leave=False, dynamic_ncols=True)

        for feats, labels, metas in pbar:
            optimizer.zero_grad()
            batch_loss = 0.0
            valid_samples = 0

            for f, m in zip(feats, metas):
                h_fused = model.encoder(f.unsqueeze(0).to(device))

                loss_p1 = train_step_phase1(model.phase1, h_fused, [m['split_points']], device)
                loss_p2 = train_step_phase2(model.phase2, h_fused, [m['split_points']], [m['flow_label']], device)

                combined_loss = config.LOSS_ALPHA * loss_p1 + (1.0 - config.LOSS_ALPHA) * loss_p2

                batch_loss += combined_loss
                valid_samples += 1

            if valid_samples > 0:
                batch_loss = batch_loss / valid_samples
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP_NORM)
                optimizer.step()
                total_loss += batch_loss.item()
                n += 1

            pbar.set_postfix({'Loss': f"{total_loss / max(n, 1):.4f}"})

        model.eval()
        val_loss, val_n = 0.0, 0
        with torch.no_grad():
            for feats, labels, metas in test_loader:
                for f, m in zip(feats, metas):
                    h_fused = model.encoder(f.unsqueeze(0).to(device))
                    l1 = train_step_phase1(model.phase1, h_fused, [m['split_points']], device)
                    l2 = train_step_phase2(model.phase2, h_fused, [m['split_points']], [m['flow_label']], device)

                    val_loss += (config.LOSS_ALPHA * l1 + (1.0 - config.LOSS_ALPHA) * l2).item()
                    val_n += 1
        val_loss /= max(val_n, 1)

        print(f"  Epoch {epoch + 1:3d}/{config.EPOCHS} | Train Loss: {total_loss / max(n, 1):.4f} | Val Loss: {val_loss:.4f}")

        torch.cuda.empty_cache()

        if val_loss < best_loss:
            best_loss, patience_cnt = val_loss, 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_cnt += 1
            if patience_cnt >= config.PATIENCE:
                print(f"  [Early Stopping] Validation loss has not improved for {config.PATIENCE} epochs.")
                break

    model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
    model.eval()
    print("  [INFO] Training completed. Best weights loaded.")