from utils import TestCase
import torch
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2

class TestNNAutoencoderKLFlux2(TestCase):

    def test_mode(self):
        model = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        sample = torch.randn(1, 3, 16, 16)

        expected = model(sample).sample

        actual = self.cli(
            "AutoencoderKLFlux2",

            "--sample", str(sample.tolist()),

            "--in_channels", "3",
            "--out_channels", "3",
            "--latent_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "8",
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_vae_encode_tiled(self):
        # The tiled encode must match the reference tiled encode
        # (diffusers' _tiled_encode with the same tile size; the latent
        # mode is the first half of the returned moments).
        model = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        # 48x32 image with 24x24 tiles (18 stride): a 3x3 tile grid with
        # clipped right and bottom borders, so the tiling and the tile
        # blending are exercised in both dimensions.
        sample = torch.randn(1, 3, 48, 32)

        model.tile_sample_min_size = 24
        model.tile_latent_min_size = 24
        model.use_tiling = True

        expected = model._tiled_encode(sample)[:, :model.config.latent_channels]

        actual = self.cli(
            "AutoencoderKLFlux2_encode_tiled",

            "--sample", str(sample.tolist()),
            "--tile_size", "24",

            "--in_channels", "3",
            "--out_channels", "3",
            "--latent_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "8",
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_vae_decode_tiled(self):
        # The tiled decode must match the reference tiled decode
        # (diffusers' tiled_decode with the same tile size).
        model = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        # 36x32 latent with 16x16 tiles (12 stride): a 3x3 tile grid with
        # clipped right and bottom borders, so the tiling and the tile
        # blending are exercised in both dimensions.
        sample = torch.randn(1, model.config.latent_channels, 36, 32)

        model.tile_sample_min_size = 16
        model.tile_latent_min_size = 16
        model.use_tiling = True

        expected = model.tiled_decode(sample).sample

        actual = self.cli(
            "AutoencoderKLFlux2_decode_tiled",

            "--sample", str(sample.tolist()),
            "--tile_size", "16",

            "--in_channels", "3",
            "--out_channels", "3",
            "--latent_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "8",
            *self.params(model),
        )

        self.assertTensors(actual, [expected])

    def test_vae_encode_tiled_interior(self):
        # 42x42 image with 24x24 tiles (18 stride): a 3x3 tile grid.
        # Only the interior bands are compared (the last band row and
        # column belong to the clipped border tiles).
        model = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        sample = torch.randn(1, 3, 42, 42)

        model.tile_sample_min_size = 24
        model.tile_latent_min_size = 24
        model.use_tiling = True

        expected = model._tiled_encode(sample)[:, :model.config.latent_channels]

        actual = self.cli(
            "AutoencoderKLFlux2_encode_tiled",

            "--sample", str(sample.tolist()),
            "--tile_size", "24",

            "--in_channels", "3",
            "--out_channels", "3",
            "--latent_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "8",
            *self.params(model),
        )

        self.assertTensors([actual[0][:, :, :36, :36]], [expected[:, :, :36, :36]])  # interior only

    def test_vae_decode_tiled_interior(self):
        # 14x14 latent with 8x8 latent tiles (6 stride): a 3x3 tile
        # grid. Only the interior bands are compared (the last band row
        # and column belong to the clipped border tiles).
        model = AutoencoderKLFlux2(
            in_channels=3,
            out_channels=3,
            latent_channels=4,
            down_block_types=("DownEncoderBlock2D",),
            up_block_types=("UpDecoderBlock2D",),
            block_out_channels=(8,),
            layers_per_block=1,
            norm_num_groups=8,
        )

        sample = torch.randn(1, model.config.latent_channels, 14, 14)

        model.tile_sample_min_size = 16
        model.tile_latent_min_size = 16
        model.use_tiling = True

        expected = model.tiled_decode(sample).sample

        actual = self.cli(
            "AutoencoderKLFlux2_decode_tiled",

            "--sample", str(sample.tolist()),
            "--tile_size", "16",

            "--in_channels", "3",
            "--out_channels", "3",
            "--latent_channels", "4",
            "--block_out_channels", "8",
            "--layers_per_block", "1",
            "--norm_num_groups", "8",
            *self.params(model),
        )

        self.assertTensors([actual[0][:, :, :12, :12]], [expected[:, :, :12, :12]])  # interior only
