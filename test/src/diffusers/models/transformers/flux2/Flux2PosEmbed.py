from utils import TestCase
import torch
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_flux2 import Flux2PosEmbed

def mrope_perm(head_dim):
    # The MROPE (NEOX) layout position p holds the original (adjacent)
    # head-dim position P(p): the even layout half maps to the first
    # halves of the pairs, the odd half to the second halves.
    return [2 * p if p < head_dim // 2 else 2 * (p - head_dim // 2) + 1 for p in range(head_dim)]

class TestNNFlux2PosEmbed(TestCase):
    def test_zero(self):
        axes_dim = [2, 2, 2, 2]
        head_dim = sum(axes_dim)
        perm = torch.tensor(mrope_perm(head_dim))

        pos_embed = Flux2PosEmbed(
            theta=10000,
            axes_dim=axes_dim,
        )

        position_ids = torch.tensor([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ])

        x_orig = torch.randn(
            1,                      # batch
            position_ids.shape[0],  # sequence length
            4,                      # heads
            head_dim,               # head_dim
        )

        # The C++ module takes x in the MROPE (even/odd) head layout.
        x = x_orig[..., perm]

        cos, sin = pos_embed(position_ids)
        expected_orig = apply_rotary_emb(
            x_orig,
            (cos, sin),
            sequence_dim=1,
        )

        # The C++ output is in the MROPE layout: permute the reference
        # (adjacent) layout output the same way.
        expected = expected_orig[..., perm]

        actual = self.cli(
            "Flux2PosEmbed",
            "--theta", "10000",
            "--axes_dim", "2",
            "--axes_dim", "2",
            "--axes_dim", "2",
            "--axes_dim", "2",

            "--x", str(x.tolist()),
            "--position_ids", str(position_ids.T.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_4d_axes(self):
        axes_dim = [10, 10, 10, 10]
        head_dim = sum(axes_dim)
        perm = torch.tensor(mrope_perm(head_dim))

        pos_embed = Flux2PosEmbed(
            theta=10000,
            axes_dim=axes_dim,
        )

        position_ids = torch.tensor([
            [0, 0, 0, 0],
            [1, 2, 3, 4],
            [5, 6, 7, 8],
        ])

        x_orig = torch.randn(
            1,                      # batch
            position_ids.shape[0],  # sequence length
            4,                      # heads
            head_dim,               # head_dim
        )

        # The C++ module takes x in the MROPE (even/odd) head layout.
        x = x_orig[..., perm]

        cos, sin = pos_embed(position_ids)
        expected_orig = apply_rotary_emb(
            x_orig,
            (cos, sin),
            sequence_dim=1,
        )

        # The C++ output is in the MROPE layout: permute the reference
        # (adjacent) layout output the same way.
        expected = expected_orig[..., perm]

        actual = self.cli(
            "Flux2PosEmbed",
            "--theta", "10000",
            "--axes_dim", "10",
            "--axes_dim", "10",
            "--axes_dim", "10",
            "--axes_dim", "10",

            "--x", str(x.tolist()),
            "--position_ids", str(position_ids.T.tolist()),
        )

        self.assertTensors(actual, [expected])
