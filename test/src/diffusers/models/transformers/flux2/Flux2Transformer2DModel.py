import tempfile
from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel


def _mrope_p(head_dim):
    # The MROPE (NEOX) layout position p holds the original (adjacent)
    # head-dim position P(p).
    p = torch.arange(head_dim)
    return torch.where(p < head_dim // 2, 2 * p, 2 * (p - head_dim // 2) + 1)

def _mrope_permute_rows(w, heads, head_dim):
    # The C++ attention module stores the q/k projection weights in the
    # MROPE head layout: new row (h, p) = old row (h, P(p)).
    P = _mrope_p(head_dim)
    old = (torch.arange(heads).unsqueeze(1) * head_dim + P.unsqueeze(0)).flatten()
    w.data = w.data[old]

def _mrope_permute_fused_qkv_rows(w, heads, head_dim):
    # Same, for the fused qkv+mlp projection: only the q and k row
    # blocks (the first two head blocks) are permuted, v and the MLP
    # rows keep the original layout.
    P = _mrope_p(head_dim)
    inner = heads * head_dim
    old = (torch.arange(heads).unsqueeze(1) * head_dim + P.unsqueeze(0)).flatten()
    idx = torch.arange(w.shape[0])
    idx[:inner] = old
    idx[inner:2 * inner] = inner + old
    w.data = w.data[idx]

def _mrope_permute_transformer_weights(model, heads, head_dim):
    # The C++ module runs one MROPE op over the whole head dimension, so
    # its q/k projections and QK norms are pre-permuted into the MROPE
    # head layout at load time (see PermutedParameter). The reference model
    # must therefore be evaluated *before* this permutation: per-axis
    # rope in the original layout satisfies MROPE(P q) == P(per_axis(q)),
    # and the attention output, built from the unpermuted v, stays in the
    # original layout.
    for block in model.transformer_blocks:
        attn = block.attn
        _mrope_permute_rows(attn.to_q.weight, heads, head_dim)
        _mrope_permute_rows(attn.to_k.weight, heads, head_dim)
        _mrope_permute_rows(attn.add_q_proj.weight, heads, head_dim)
        _mrope_permute_rows(attn.add_k_proj.weight, heads, head_dim)
        attn.norm_q.weight.data = attn.norm_q.weight.data[_mrope_p(head_dim)]
        attn.norm_k.weight.data = attn.norm_k.weight.data[_mrope_p(head_dim)]
        attn.norm_added_q.weight.data = attn.norm_added_q.weight.data[_mrope_p(head_dim)]
        attn.norm_added_k.weight.data = attn.norm_added_k.weight.data[_mrope_p(head_dim)]

    for block in model.single_transformer_blocks:
        attn = block.attn
        _mrope_permute_fused_qkv_rows(attn.to_qkv_mlp_proj.weight, heads, head_dim)
        attn.norm_q.weight.data = attn.norm_q.weight.data[_mrope_p(head_dim)]
        attn.norm_k.weight.data = attn.norm_k.weight.data[_mrope_p(head_dim)]

# Non-zero positions for the rope axes actually used (ids columns 0 and
# 1 for axes_dims_rope=(20, 20)) so the rotary embedding is exercised,
# plus non-zero values in the unused columns 2 and 3 as a guard against
# implementations that (incorrectly) read them.
img_ids = torch.tensor([
    [
        [0, 0, 2, 0],
        [1, 1, 3, 1],
    ]
])

txt_ids = torch.tensor([
    [
        [0, 3, 0, 2],
        [4, 0, 0, 3],
        [5, 6, 0, 4],
    ]
])

class TestNNFlux2Transformer2DModel(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.TemporaryDirectory(delete=False)

    def test_minimal(self):
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=40,
            num_attention_heads=2,          # inner_dim = 80
            joint_attention_dim=16,
            timestep_guidance_channels=8,
            axes_dims_rope=(20, 20),    # sums to head_dim (=40)
            guidance_embeds=False,
        )

        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 16)

        timestep = torch.tensor([0.5])

        # Compute the reference output in the original head layout
        # (per-axis rope) *before* re-permuting the weights for the
        # CLI -- see _mrope_permute_transformer_weights.
        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        _mrope_permute_transformer_weights(model, 2, 40)

        actual = self.cli(
            "Flux2Transformer2DModel",
            "--patch_size", "1",
            "--in_channels", "8",
            "--out_channels", "8",
            "--num_layers", "1",
            "--num_single_layers", "1",
            "--attention_head_dim", "40",
            "--num_attention_heads", "2",
            "--joint_attention_dim", "16",
            "--timestep_guidance_channels", "8",
            "--axes_dims_rope", "20",
            "--axes_dims_rope", "20",
            "--guidance_embeds", "false",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--timestep", str(timestep.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, [expected])

    def test_guidance(self):
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=1,
            num_single_layers=1,
            attention_head_dim=40,
            num_attention_heads=2,
            joint_attention_dim=16,
            timestep_guidance_channels=8,
            axes_dims_rope=(20, 20),
            guidance_embeds=True,
        )

        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 16)

        timestep = torch.tensor([0.5])
        guidance = torch.tensor([1.0])

        # Compute the reference output in the original head layout
        # (per-axis rope) *before* re-permuting the weights for the
        # CLI -- see _mrope_permute_transformer_weights.
        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            guidance=guidance,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        _mrope_permute_transformer_weights(model, 2, 40)

        actual = self.cli(
            "Flux2Transformer2DModel",
            "--patch_size", "1",
            "--in_channels", "8",
            "--out_channels", "8",
            "--num_layers", "1",
            "--num_single_layers", "1",
            "--attention_head_dim", "40",
            "--num_attention_heads", "2",
            "--joint_attention_dim", "16",
            "--timestep_guidance_channels", "8",
            "--axes_dims_rope", "20",
            "--axes_dims_rope", "20",
            "--guidance_embeds", "true",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--timestep", str(timestep.tolist()),
            "--guidance", str(guidance.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, [expected])

    def test_multiple_blocks(self):
        model = Flux2Transformer2DModel(
            patch_size=1,
            in_channels=8,
            out_channels=8,
            num_layers=2,
            num_single_layers=2,
            attention_head_dim=40,
            num_attention_heads=2,          # inner_dim = 80
            joint_attention_dim=16,
            timestep_guidance_channels=8,
            axes_dims_rope=(20, 20),    # sums to head_dim (=40)
            guidance_embeds=False,
        )

        hidden_states = torch.randn(1, 2, 8)
        encoder_hidden_states = torch.randn(1, 3, 16)

        timestep = torch.tensor([0.5])

        # Compute the reference output in the original head layout
        # (per-axis rope) *before* re-permuting the weights for the
        # CLI -- see _mrope_permute_transformer_weights.
        expected = model(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            timestep=timestep,
            img_ids=img_ids,
            txt_ids=txt_ids,
        ).sample

        _mrope_permute_transformer_weights(model, 2, 40)

        actual = self.cli(
            "Flux2Transformer2DModel",
            "--patch_size", "1",
            "--in_channels", "8",
            "--out_channels", "8",
            "--num_layers", "2",
            "--num_single_layers", "2",
            "--attention_head_dim", "40",
            "--num_attention_heads", "2",
            "--joint_attention_dim", "16",
            "--timestep_guidance_channels", "8",
            "--axes_dims_rope", "20",
            "--axes_dims_rope", "20",
            "--guidance_embeds", "false",
            "--hidden_states", str(hidden_states.tolist()),
            "--encoder_hidden_states", str(encoder_hidden_states.tolist()),
            "--timestep", str(timestep.tolist()),
            "--img_ids", str(img_ids.tolist()),
            "--txt_ids", str(txt_ids.tolist()),
            *self.params(model, self.tmpdir.name),
        )

        self.assertTensors(actual, [expected])
