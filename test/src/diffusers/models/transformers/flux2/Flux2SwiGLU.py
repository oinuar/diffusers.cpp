from utils import TestCase
import torch
from diffusers.models.transformers.transformer_flux2 import Flux2SwiGLU

class TestFlux2SwiGLU(TestCase):
    def test_1d(self):
        model = Flux2SwiGLU()
        x = torch.randn(4)

        expected = model(x)

        actual = self.cli(
            'Flux2SwiGLU',
            '--x', str(x.tolist()),
        )

        self.assertTensors(actual, [expected])

    def test_2d(self):
        model = Flux2SwiGLU()
        x = torch.randn(2, 12)

        expected = model(x)

        actual = self.cli(
            'Flux2SwiGLU',
            '--x', str(x.tolist()),
        )

        self.assertTensors(actual, [expected])
