from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Attention, Flux2PosEmbed
from torch.nn.attention import SDPBackend, sdpa_kernel

class TestFlux2Attention(TestCase):
    def test(self):
        model = Flux2Attention(
            query_dim=8,
            heads=2,
            dim_head=40,
        )

        hidden_states = torch.randn(1, 3, 8)

        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = model(hidden_states)

        actual = self.cli(
            "Flux2Attention",
            "--query_dim", "8",
            "--heads", "2",
            "--dim_head", "40",
            "--hidden_states", str(hidden_states.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_image_rotary_emb(self):
        model = Flux2Attention(
            query_dim=16,
            heads=2,
            dim_head=40,
        )

        head_dim = model.inner_dim // model.heads
        axes_dim = [head_dim // 4] * 4

        pos_embed = Flux2PosEmbed(
            theta=10000,
            axes_dim=axes_dim,
        )

        hidden_states = torch.randn(1, 3, 16)
        position_ids = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])

        image_rotary_emb = pos_embed(position_ids)

        # The reference model runs in the original head layout with
        # per-axis rope: the C++ module stores its q/k projections and
        # QK norms in the MROPE head layout (the loader permutes the
        # passed weights at load time -- see Parameter::Permutation)
        # and applies one MROPE op over the permuted projections, which
        # satisfies MROPE(P q) == P(per_axis(q)) and leaves the
        # (unpermuted) v, scores, and output projection untouched.
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = model(hidden_states, image_rotary_emb=image_rotary_emb)

        actual = self.cli(
            "Flux2Attention",
            "--query_dim", "16",
            "--heads", "2",
            "--dim_head", "40",
            "--hidden_states", str(hidden_states.tolist()),
            "--image_rotary_emb-theta", "10000",
            "--image_rotary_emb-axes_dim", str(axes_dim[0]),
            "--image_rotary_emb-axes_dim", str(axes_dim[1]),
            "--image_rotary_emb-axes_dim", str(axes_dim[2]),
            "--image_rotary_emb-axes_dim", str(axes_dim[3]),
            "--image_rotary_emb-position_ids", str(position_ids.T.tolist()),
            *self.params(model),
        )

        self.assertTensors(actual, [expected])
