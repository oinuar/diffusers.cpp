from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Attention, Flux2PosEmbed
from torch.nn.attention import SDPBackend, sdpa_kernel

def _mrope_p(head_dim):
    # The MROPE (NEOX) layout position p holds the original (adjacent)
    # head-dim position P(p).
    p = torch.arange(head_dim)
    return torch.where(p < head_dim // 2, 2 * p, 2 * (p - head_dim // 2) + 1)

def _mrope_permute_rows(weight, heads, head_dim):
    # The C++ attention module stores the q/k projection weights in the
    # MROPE head layout: new row (h, p) = old row (h, P(p)).
    P = _mrope_p(head_dim)
    old = (torch.arange(heads).unsqueeze(1) * head_dim + P.unsqueeze(0)).flatten()
    weight.data = weight.data[old]

def _mrope_permute_elements(weight, head_dim):
    # Same, for the per-head QK-norm weights (new element p = old P(p)).
    weight.data = weight.data[_mrope_p(head_dim)]

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
        # per-axis rope, so compute the expected output *before* the
        # q/k weights are re-permuted into the MROPE layout: the C++
        # module applies one MROPE op over the permuted projections,
        # which satisfies MROPE(P q) == P(per_axis(q)) and leaves the
        # (unpermuted) v, scores, and output projection untouched.
        with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
            expected = model(hidden_states, image_rotary_emb=image_rotary_emb)

        # The C++ module stores the q/k projections and QK norms in the
        # MROPE head layout (the loader applies the permutation at load
        # time -- see PermutedParameter), so apply the same permutation to
        # the weights passed to the CLI.
        _mrope_permute_rows(model.to_q.weight, model.heads, head_dim)
        _mrope_permute_rows(model.to_k.weight, model.heads, head_dim)
        _mrope_permute_elements(model.norm_q.weight, head_dim)
        _mrope_permute_elements(model.norm_k.weight, head_dim)

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
