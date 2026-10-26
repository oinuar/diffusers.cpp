import unittest
import subprocess
import os
import ast
import fnmatch
import json
import torch
import numpy as np
from gguf import (
    GGMLQuantizationType,
    quantize as gguf_quantize,
    dequantize as gguf_dequantize,
    quant_shape_to_byte_shape,
)


class TypeSpec:
    """
    Per-component quantization spec, a mirror of the TypeSpec of
    utils/convert-model/src/convert_hf_safetensors_to_gguf.py.

    default  -- a single quantization type applied to every tensor
    patterns -- per-tensor map: tensor name pattern -> quantization type

    An exact tensor name takes precedence over patterns; when several
    patterns match, the most specific one (the longest pattern) wins,
    ties in the order they appear.
    """
    def __init__(self, default=None, patterns=None):
        self.default = default
        self.patterns = patterns or {}

    @staticmethod
    def quantization_type(name):
        try:
            return GGMLQuantizationType[name]
        except KeyError:
            valid = ", ".join(t.name for t in GGMLQuantizationType)
            raise ValueError(f"Unknown quantization type {name!r}. Valid types: {valid}") from None

    @staticmethod
    def from_map(obj):
        return TypeSpec(
            patterns={
                pattern: TypeSpec.quantization_type(type_name)
                for pattern, type_name in obj.items()
            }
        )

    def resolve(self, name):
        """
        Resolve the quantization type of a tensor, or None to keep its
        original dtype.
        """
        if name in self.patterns:
            return self.patterns[name]

        for pattern in sorted(self.patterns, key=len, reverse=True):
            if fnmatch.fnmatchcase(name, pattern):
                return self.patterns[pattern]

        return self.default


class QuantSpec:
    """
    Tensor quantization spec with the same grammar as the --type option of
    utils/convert-model/src/convert_hf_safetensors_to_gguf.py:

      * None                        -> keep every tensor's original dtype
      * "Q8_0"                      -> one type for every tensor
      * '{"*": "Q8_0", "foo": "BF16"}' -> per-tensor JSON object, applied
        to every component
      * '{"transformer": {"*": "Q8_0"}, "text_encoder": "BF16"}'
                                      -> per-component JSON object: one
        entry per component, each a type name or a per-tensor object; the
        "*" entry is the default for components without an explicit entry
      * "@map.json"                 -> one of the above JSON objects read
        from a file

    A JSON object with only string values is a per-tensor map for all
    components; the per-component form is detected as soon as any value
    is an object.
    """
    def __init__(self, value=None, type_spec=None, components=None, fallback=None):
        self.value = value
        self.type_spec = type_spec
        self.components = components
        self.fallback = fallback

    @staticmethod
    def parse(value):
        if value is None:
            return QuantSpec()

        if isinstance(value, QuantSpec):
            return value

        text = value.strip()

        if not text:
            return QuantSpec()

        if text.startswith("@"):
            with open(text[1:]) as f:
                obj = json.load(f)
        elif text.startswith("{"):
            obj = json.loads(text)
        else:
            # a bare quantization type name applied to every tensor
            return QuantSpec(value, type_spec=TypeSpec(default=TypeSpec.quantization_type(text)))

        if not isinstance(obj, dict):
            raise ValueError(f"--quant JSON must be an object: {value!r}")

        if any(isinstance(item, dict) for item in obj.values()):
            components = {}
            fallback = None

            for name, item in obj.items():
                if isinstance(item, dict):
                    component_spec = TypeSpec.from_map(item)
                elif isinstance(item, str):
                    component_spec = TypeSpec(default=TypeSpec.quantization_type(item))
                else:
                    raise ValueError(f"--quant component {name!r} must be a quantization type name or an object")

                if name == "*":
                    if fallback is not None:
                        raise ValueError("--quant has multiple '*' component entries")
                    fallback = component_spec
                else:
                    components[name] = component_spec

            return QuantSpec(value, components=components, fallback=fallback)

        return QuantSpec(value, type_spec=TypeSpec.from_map(obj))

    def enabled(self):
        return self.type_spec is not None or bool(self.components) or self.fallback is not None

    def resolve(self, name, component=""):
        """
        Resolve the quantization type of a tensor, or None to keep its
        original dtype.

        name      -- the tensor name inside its component (dotted, without
                     the component prefix)
        component -- the component prefix ("" for a standalone model)
        """
        if self.components:
            spec = self.components.get(component, self.fallback) or TypeSpec()
        else:
            spec = self.type_spec or TypeSpec()

        return spec.resolve(name)


def quantize_to_bytes(tensor, qtype):
    """
    The raw bytes a GGUF file stores for a tensor of `qtype`, the same
    conversion utils/convert-model applies (quantize_tensor).
    """
    arr = tensor.detach().cpu().contiguous()

    # numpy cannot represent BF16; promote to float32 (lossless)
    if arr.dtype == torch.bfloat16:
        arr = arr.float()

    arr = arr.numpy()

    if qtype == GGMLQuantizationType.F32:
        return arr.astype(np.float32).tobytes()

    if qtype == GGMLQuantizationType.F64:
        return arr.astype(np.float64).tobytes()

    if qtype == GGMLQuantizationType.I8:
        return arr.astype(np.int8).tobytes()

    if qtype == GGMLQuantizationType.I16:
        return arr.astype(np.int16).tobytes()

    if qtype == GGMLQuantizationType.I32:
        return arr.astype(np.int32).tobytes()

    if qtype == GGMLQuantizationType.I64:
        return arr.astype(np.int64).tobytes()

    if qtype == GGMLQuantizationType.F16:
        return arr.astype(np.float16).tobytes()

    # BF16 and all quantized types (Q8_0, ...) via the gguf package, the
    # same one utils/convert-model uses.
    return gguf_quantize(arr, qtype).tobytes()


def dequantize_to_float32(raw, qtype, shape):
    """
    The float32 values the raw bytes of `qtype` represent, i.e. what the
    C++ side (ggml) computes from the same bytes.
    """
    if qtype == GGMLQuantizationType.F32:
        return torch.from_numpy(np.frombuffer(raw, np.float32).reshape(shape))

    if qtype == GGMLQuantizationType.F64:
        return torch.from_numpy(np.frombuffer(raw, np.float64).reshape(shape).astype(np.float32))

    if qtype in (
        GGMLQuantizationType.I8,
        GGMLQuantizationType.I16,
        GGMLQuantizationType.I32,
        GGMLQuantizationType.I64,
    ):
        dtype = {
            GGMLQuantizationType.I8: np.int8,
            GGMLQuantizationType.I16: np.int16,
            GGMLQuantizationType.I32: np.int32,
            GGMLQuantizationType.I64: np.int64,
        }[qtype]
        return torch.from_numpy(np.frombuffer(raw, dtype).reshape(shape).astype(np.float32))

    if qtype == GGMLQuantizationType.F16:
        return torch.from_numpy(np.frombuffer(raw, np.float16).reshape(shape).astype(np.float32))

    # BF16 and all quantized types via the gguf package
    data = np.frombuffer(raw, np.uint8).reshape(quant_shape_to_byte_shape(shape, qtype))
    return torch.from_numpy(gguf_dequantize(data, qtype))


class TestCase(unittest.TestCase):
    def run(self, result=None):
        result = super().run(result)

        for collection in (result.failures, result.errors):
            for i, (test, traceback) in enumerate(collection):
                if test is self:
                    path = self.repro_script_path()
                    
                    if self.write_repro_script():
                        collection[i] = (
                            test,
                            traceback
                            + f"\nRun this command to reproduce the failure:\n"
                            f"    uv run {path}\n",
                        )

        return result

    def write_repro_script(self):
        if not self._cli_command:
            return False

        os.makedirs(os.path.dirname(self.repro_script_path()), exist_ok=True)

        with open(self.repro_script_path(), 'w') as f:
            f.write(f"""
import os, sys, subprocess

# Never edit this line! We must keep test parameters as-is.
command = [{repr(self.cli_bin())}] + {repr(self._cli_command)}

# Add gdb by editing this line while keeping test parameters exactly same. For example:
#   command = ['gdb', '--args'] + command

result = subprocess.run(
    command,
    stdin=sys.stdin,
    stdout=sys.stdout,
    stderr=sys.stderr,
)

sys.exit(result.returncode)
""")
        return True

    def cli(self, *args: str) -> list:
        self._cli_command = [
            *args,
            '--runner-n_devices', str(self.n_devices()),
            '--runner-use_gpu', str(self.use_gpu()).lower(),
            *sum((["--runner-tensor_split", str(x)] for x in self.tensor_splits()), [])
        ]

        command_w_args = " ".join([self.cli_bin()] + list(map(lambda x: x if x.startswith("--") else f'"{x}"', self._cli_command)))

        result = subprocess.run([self.cli_bin()] + self._cli_command, capture_output=True, text=True, timeout=30)

        # If command failed, write a reproduction script.
        if result.returncode != 0:
            raise RuntimeError(f'{self.cli_bin()} failed (rc={result.returncode}):\n{result.stderr}')

        outputs = []
        for line in result.stdout.strip().split('\n'):
            value = ast.literal_eval(line)

            # TODO: add dtype to CLI output to make this properly
            try:
                outputs.append(torch.tensor(value, dtype=torch.float32))
                continue
            except ValueError:
                pass

            outputs.append(value)

        if self.verbose():
            print(result.stderr)

        return outputs

    def cli_bin(self):
        return os.environ['CLI']

    def repro_script_path(self):
        return os.path.join(os.environ['REPRO_DIR'], os.environ['TEST_ID'] + ".py")

    def n_devices(self):
        return int(os.environ.get('N_DEVICES', '1'))

    def tensor_splits(self):
        splits = os.environ.get('TENSOR_SPLITS', '')
        if splits == '':
            return []
        return splits.split(',')

    def use_gpu(self):
        return os.environ.get('USE_GPU', 'false') == 'true'

    def verbose(self):
        return os.environ.get('VERBOSE', 'false') == 'true'

    def params(self, model, path=None, prefix=""):
        args = []
        param_id = 0

        for name, tensor in list(model.named_parameters()) + list(model.named_buffers()):
            name = f"{prefix}-{name}" if prefix else name
            name = name.replace(".", "-")
            param_id += 1

            value = str(tensor.tolist())

            if path is not None:
                filename = os.path.join(path, f"{param_id}.{prefix}-params")
                with open(filename, "w") as f:
                    f.write(value)
                value = filename

            args.extend([f"--param-{name}", value])

        return args

    def quant_params(self, model, path=None, prefix="", spec=None):
        """
        Like params(), but every parameter is first resolved against the
        quantization spec (same grammar as the --type option of
        utils/convert-model) and, when a quantized type is resolved, the
        file holds the raw quantized bytes instead of a float32 list,
        exactly like a tensor in a GGUF file produced by convert-model.

        Returns (args, state):
          args  -- the --param-... CLI arguments; the caller must add
                  ["--quant", spec.value] once to the CLI command
          state -- the float32 weights the raw bytes represent
                  ({tensor name: tensor}), the values the C++ side (ggml)
                  computes from the same bytes; load them into the model
                  for the Python reference so both sides see identical
                  values
        """
        spec = QuantSpec.parse(spec)
        args = []
        state = {}
        param_id = 0

        for name, tensor, is_buffer in (
            [(n, t, False) for n, t in model.named_parameters()]
            + [(n, t, True) for n, t in model.named_buffers()]
        ):
            param_id += 1

            # Buffers are not quantized (the C++ side computes them
            # internally), they keep the regular float32 path
            qtype = None if is_buffer else spec.resolve(name, prefix)

            if qtype is None or qtype == GGMLQuantizationType.F32:
                # A regular float32 parameter, as in params()
                value = str(tensor.tolist())

                if path is not None:
                    filename = os.path.join(path, f"{param_id}.{prefix}-params")
                    with open(filename, "w") as f:
                        f.write(value)
                    value = filename
            else:
                raw = quantize_to_bytes(tensor, qtype)

                filename = os.path.join(path, f"{param_id}.{prefix}-{qtype.name}-quant")

                with open(filename, "wb") as f:
                    f.write(raw)

                value = filename
                state[name] = dequantize_to_float32(raw, qtype, tuple(tensor.shape))

            option = f"--param-{prefix}-{name}" if prefix else f"--param-{name}"
            option = option.replace(".", "-")

            args.extend([option, value])

        return args, state

    def apply_quantized_state(self, model, state):
        """
        Replace the model's weights with the float32 values a quantization
        produced (see quant_params), so the Python reference computes with
        the same values the C++ side gets from the quantized bytes.
        """
        parameters = dict(model.named_parameters())

        for name, value in state.items():
            parameters[name].data.copy_(value.to(parameters[name].device))

    def assertTensorsLess(self, actual: list, expected: list, max: float):
        diff = torch.stack(expected) - torch.stack(actual)

        actual_t = torch.stack(actual)
        expected_t = torch.stack(expected)
        diff_abs = (actual_t - expected_t).abs()

        k = min(10, diff_abs.numel())
        values, indices = torch.topk(diff_abs.flatten(), k)

        locations = [
            tuple(i.item() for i in torch.unravel_index(idx, diff_abs.shape))
            for idx in indices
        ]

        details = "\n".join(
            f"  - index={loc}, diff={val.item():.8g}, "
            f"actual={actual_t[loc].item():.8g}, "
            f"expected={expected_t[loc].item():.8g}"
            for i, (loc, val) in enumerate(zip(locations, values))
        )

        self.assertLess(
            diff_abs.max().item(), max,
            f"Max difference: {diff_abs.max().item():.8g} (limit={max})\n"
            f"Top differences:\n{details}" + (f"\n  ...and {diff_abs.numel() - k} more." if diff_abs.numel() > k else "")
        )

    def assertTensors(self, actual: list, expected: list, *args: str, **kwargs):
        finalKwargs = { 'rtol': 1e-4, 'atol': 1e-6, **kwargs }
        index = 0

        # GPU uses less accurate versions of operators in some cases, so make thresholds looser
        if self.use_gpu:
            finalKwargs['rtol'] = 2e-3
            finalKwargs['atol'] = 1e-3

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a.shape, e.shape, f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            self.assertEqual(a.dtype, e.dtype, f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            self.assertTrue(torch.allclose(a, e, **finalKwargs), f'\nActual: {str(a.tolist())}\nExpected: {str(e.tolist())}\nIndex: {index}')
            index += 1
