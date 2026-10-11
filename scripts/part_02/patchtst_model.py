
'''
🛠️ Optimization: Replacing the Vector BottleneckIn your model script, 
you noted that the _create_patches loop is slow. Using native Python for 
loops inside PyTorch forces execution back to the CPU, stalling the GPU
 pipeline. We can fully vectorise this using torch.Tensor.unfold, which 
 creates sliding windows in parallel directly inside CUDA memory. 
 💾 1. Optimised Model Code (model_patchtst_regime_pe.py)This version 
 optimizes patch generation using tensor unfolding and handles channel 
 layout configurations robustly.
'''
# model_upgr_patchtst.py
import torch
import torch.nn as nn

class PatchTST(nn.Module):
    def __init__(
        self,
        c_in,
        patch_len,
        stride,
        max_len=128,         # Total lookback length (e.g. 64)
        num_layers=4,
        num_heads=4,
        d_model=128,
        d_ff=256,
        dropout=0.1,
        n_classes=None,
    ):
        super().__init__()

        self.c_in = c_in
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_classes = n_classes

        # Flattened patch size across all channels
        self.embedding = nn.Linear(c_in * patch_len, d_model)

        # Compute number of patches along the time dimension
        self.max_patches = (max_len - patch_len) // stride + 1

        # Learnable positional embedding
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.max_patches, d_model)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        if n_classes is not None:
            self.head = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Dropout(0.2),
                nn.Linear(d_model, n_classes)
            )
        else:
            self.head = None

    def _create_patches(self, x):
        """
        Fully vectorized patch creation using torch.Tensor.unfold.
        Input x shape: (B, T, C)
        Output shape:  (B, N, patch_len * C)
        """
        B, T, C = x.shape
        
        # Permute to (B, C, T) to unfold along the time axis easily
        x_transposed = x.permute(0, 2, 1)
        
        # unfold returns shape: (B, C, N, patch_len)
        # N = (T - patch_len) // stride + 1
        patches = x_transposed.unfold(dimension=2, size=self.patch_len, step=self.stride)
        
        # Rearrange and flatten to (B, N, patch_len * C)
        patches = patches.permute(0, 2, 1, 3).contiguous()
        patches = patches.view(B, patches.size(1), -1)
        
        return patches

    def forward(self, x, return_embedding=False):
        # Vectorized generation of patches
        x_patched = self._create_patches(x)  # (B, N, patch_len * C)

        # Map to d_model space
        z = self.embedding(x_patched)        # (B, N, d_model)

        # Add positional embedding sliced up to the current window size N
        N = z.size(1)
        z = z + self.pos_embedding[:, :N, :]

        # Run through Transformer core
        z = self.encoder(z)

        # Context global pooling across the patching timeline
        z_pooled = z.mean(dim=1)

        if return_embedding or self.head is None:
            return z_pooled

        return self.head(z_pooled)
