import torch
from torch import nn
from torchvision import transforms
from typing import Tuple

# backbone_name -> Hugging Face Hub id. DINOv3 weights are gated; the HF repo
# must be accepted (and `huggingface-cli login` run, or HF_TOKEN set) before
# AutoModel.from_pretrained() below will succeed.
_DINOV3_HF_IDS = {
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
}


class LimoNet(nn.Module):
    def __init__(
        self,
        goal_dim: int = 3,
        path_length: int = 50,
        se2_dim: int = 3,
        backbone_name: str = "dinov3_vits16",
        pretrained: bool = True,
        # o dinov3 usa patch_size=16, o dinov2 usa patch_size=14
        # a altura e largura da imagem devem ser múltiplos do patch_size
        image_size: Tuple[int, int] = (304, 480),
        patch_size: int = 16,
        attn_heads: int = 6,
        decoder_layers: int = 4,  # number of Transformer decoder layers
        ff_dim_factor: int = 4,  # feed-forward hidden size = embed_dim * ff_dim_factor
    ):
        super().__init__()
        self.backbone = None
        self._initialized = False

        # config
        self.goal_dim = goal_dim
        self.path_length = path_length
        self.se2_dim = se2_dim
        self.decoder_layers = decoder_layers
        self.ff_dim_factor = ff_dim_factor

        self.backbone_name = backbone_name
        self.pretrained = pretrained

        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_h = image_size[0] // patch_size
        self.grid_w = image_size[1] // patch_size

        # transforms for input images
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
            ]
        )

        self.attn_heads = attn_heads

        # build modules
        self.setup()

    def setup(self):
        if self._initialized:
            return

        if self.backbone_name.startswith("dinov3"):
            from transformers import AutoModel
            # carrega o modelo DINOv3 do Hugging Face Hub. O HF repo deve ser aceito (e `huggingface-cli login` executado, ou HF_TOKEN definido) antes que AutoModel.from_pretrained() abaixo funcione.
            hf_id = _DINOV3_HF_IDS[self.backbone_name]
            self.backbone = AutoModel.from_pretrained(hf_id)
            # define o embed_dim e num_register_tokens com base na configuração do backbone
            self.embed_dim = self.backbone.config.hidden_size
            self.num_register_tokens = self.backbone.config.num_register_tokens
            # aqui verifica se o patch_size do backbone corresponde ao patch_size configurado
            assert self.backbone.config.patch_size == self.patch_size, (
                f"backbone patch_size={self.backbone.config.patch_size} does not "
                f"match configured patch_size={self.patch_size}"
            )
        else:
            raise ValueError(f"Unsupported backbone_name: {self.backbone_name!r}")

        for p in self.backbone.parameters():
            p.requires_grad = False
        for m in self.backbone.modules():
            if isinstance(m, nn.LayerNorm):
                for p in m.parameters():
                    p.requires_grad = True

        self.row_embed = nn.Embedding(self.grid_h, self.embed_dim)
        self.col_embed = nn.Embedding(self.grid_w, self.embed_dim)

        self.goal_proj = nn.Linear(self.goal_dim, self.embed_dim)
        self.time_embed = nn.Embedding(self.path_length, self.embed_dim)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.embed_dim,
            nhead=self.attn_heads,
            dim_feedforward=self.embed_dim * self.ff_dim_factor,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=self.decoder_layers
        )

        self.out_proj = nn.Linear(self.embed_dim, self.se2_dim)

        self._initialized = True

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        self.setup()

        image = batch["image_front"]
        goal = batch["goal"]

        B, device = image.size(0), image.device


        last_hidden_state = self.backbone(pixel_values=image).last_hidden_state
        patch_tokens = last_hidden_state[:, 1 + self.num_register_tokens :, :]
        N, D = patch_tokens.size(1), patch_tokens.size(2)
        assert N == self.grid_h * self.grid_w, (
            f"backbone returned {N} patch tokens, expected {self.grid_h * self.grid_w} "
            f"({self.grid_h}x{self.grid_w}) for image_size={self.image_size}, "
            f"patch_size={self.patch_size}"
        )

        row_ids = torch.arange(self.grid_h, device=device)
        col_ids = torch.arange(self.grid_w, device=device)
        pos = self.row_embed(row_ids).unsqueeze(1) + self.col_embed(col_ids).unsqueeze(
            0
        )
        pos = pos.view(1, N, D)
        patch_tokens = patch_tokens + pos

        goal_emb = self.goal_proj(goal)

        time_ids = torch.arange(self.path_length, device=device)
        t_emb = self.time_embed(time_ids)
        t_emb = t_emb.unsqueeze(0).expand(B, -1, -1)
        queries = t_emb + goal_emb.unsqueeze(1)

        decoder_out = self.decoder(
            tgt=queries,
            memory=patch_tokens,
        )

        path = self.out_proj(decoder_out)
        return path


if __name__ == "__main__":
    model = LimoNet()

    # count total vs. trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params:,}, trainable: {trainable_params:,}")

    # dummy run
    batch = {
        "image_front": torch.randn(2, 3, 304, 480),
        "goal": torch.randn(2, 3),
    }
    out = model(batch)
    print("Output shape:", out.shape)  # expected (2, 50, 3)
