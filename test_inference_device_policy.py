from __future__ import annotations

import ast
import os
import types
import unittest
from pathlib import Path
from unittest import mock

import inference_device_policy as policy


ROOT = Path(__file__).resolve().parent
APP_FILES = ("Sentiment Analysis.py", "Sentiment Analysis (Stable last code).py")


class Device:
    def __init__(self, value):
        self.name = str(value)
        self.type = self.name.split(":", 1)[0]
        self.index = int(self.name.split(":", 1)[1]) if ":" in self.name else None

    def __str__(self):
        return self.name


def fake_torch(*, available=True, count=2, reserved=(10, 20), totals=(100, 200), fail=()):
    module = types.SimpleNamespace()
    module.device = Device
    probes = {}

    def empty(shape, *, device):
        index = int(str(device).split(":", 1)[1])
        if index in fail:
            raise RuntimeError("allocation failed")
        probe = types.SimpleNamespace(fill_=mock.Mock())
        probes[index] = probe
        return probe

    module.empty = mock.Mock(side_effect=empty)
    module.cuda = types.SimpleNamespace(
        is_available=mock.Mock(return_value=available),
        device_count=mock.Mock(return_value=count),
        synchronize=mock.Mock(),
        get_device_properties=mock.Mock(
            side_effect=lambda i: types.SimpleNamespace(total_memory=totals[i])
        ),
        memory_reserved=mock.Mock(side_effect=lambda i: reserved[i]),
    )
    return module, probes


class InferenceDevicePolicyTests(unittest.TestCase):
    def test_cpu_admission_aliases_preempt_all_cuda_calls(self):
        cases = (
            {"CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": "0"},
            {"TRAINING_CONTROL_CPU_ONLY": "yes", "CUDA_VISIBLE_DEVICES": "0"},
            {"OPF_ADP_DISABLE_GPU_ACCELERATORS": "on"},
            {"TRAINING_CONTROL_BACKEND": "cpu", "CUDA_VISIBLE_DEVICES": "0"},
            {"CUDA_VISIBLE_DEVICES": ""},
            {"CUDA_VISIBLE_DEVICES": " -1 "},
        )
        for env in cases:
            with self.subTest(env=env):
                torch, _ = fake_torch()
                self.assertEqual(policy.best_cuda_index(torch, env), -1)
                self.assertEqual(str(policy.resolve_torch_device(torch, env)), "cpu")
                torch.cuda.is_available.assert_not_called()
                torch.cuda.device_count.assert_not_called()
                torch.empty.assert_not_called()

    def test_conflicting_scheduler_admission_fails_before_probe(self):
        torch, _ = fake_torch()
        with self.assertRaisesRegex(RuntimeError, "conflicting CPU and GPU"):
            policy.best_cuda_index(
                torch, {"TRAINING_CONTROL_BACKEND": "gpu", "CPU_ONLY": "1"}
            )
        torch.cuda.is_available.assert_not_called()
        torch.empty.assert_not_called()

    def test_best_cuda_index_requires_real_operation_and_uses_local_free_memory(self):
        torch, probes = fake_torch(reserved=(50, 20), totals=(100, 200))
        env = {"CUDA_VISIBLE_DEVICES": "4,7"}
        self.assertEqual(policy.best_cuda_index(torch, env), 1)
        self.assertEqual(torch.empty.call_count, 2)
        probes[0].fill_.assert_called_once_with(1)
        probes[1].fill_.assert_called_once_with(1)
        self.assertEqual(
            [call.args[0] for call in torch.cuda.synchronize.call_args_list], [0, 1]
        )
        # Torch sees local masked indices; never reinterpret them as physical 4/7.
        self.assertEqual(
            [call.kwargs["device"] for call in torch.empty.call_args_list],
            ["cuda:0", "cuda:1"],
        )

    def test_failed_device_probe_is_skipped_not_selected_by_memory(self):
        torch, probes = fake_torch(fail=(1,), reserved=(90, 0), totals=(100, 1000))
        self.assertEqual(policy.best_cuda_index(torch, {}), 0)
        self.assertIn(0, probes)
        self.assertNotIn(1, probes)

    def test_gpu_admitted_child_cannot_silently_fall_back(self):
        torch, _ = fake_torch(available=False)
        with self.assertRaisesRegex(RuntimeError, "GPU-admitted inference worker"):
            policy.resolve_torch_device(
                torch,
                {"TRAINING_CONTROL_BACKEND": "gpu", "CUDA_VISIBLE_DEVICES": "0"},
            )
        torch.empty.assert_not_called()

    def test_standalone_unusable_cuda_falls_back_to_cpu(self):
        torch, _ = fake_torch(available=True, count=1, fail=(0,), reserved=(0,), totals=(1,))
        self.assertEqual(
            str(policy.resolve_torch_device(torch, {"CUDA_VISIBLE_DEVICES": "0"})),
            "cpu",
        )

    def test_cpu_admission_never_imports_optional_dataframe_gpu_stack(self):
        importer = mock.Mock(side_effect=AssertionError("accelerator import attempted"))
        status = policy.enable_optional_dataframe_acceleration(
            {"CPU_ONLY": "1", "CUDA_VISIBLE_DEVICES": "0"}, importer=importer
        )
        self.assertEqual(status["backend"], "pandas")
        self.assertFalse(status["requested"])
        self.assertFalse(status["enabled"])
        importer.assert_not_called()

    def test_usable_cupy_installs_cudf_pandas_before_application_import(self):
        probe = types.SimpleNamespace(fill=mock.Mock())
        cp = types.SimpleNamespace(
            uint8=object(),
            empty=mock.Mock(return_value=probe),
            cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(
                getDeviceCount=mock.Mock(return_value=1),
                deviceSynchronize=mock.Mock(),
            )),
        )
        cudf_pandas = types.SimpleNamespace(install=mock.Mock())
        modules = {"cupy": cp, "cudf.pandas": cudf_pandas}
        status = policy.enable_optional_dataframe_acceleration(
            {"CUDA_VISIBLE_DEVICES": "0"}, importer=lambda name: modules[name]
        )
        self.assertEqual(status["backend"], "cudf.pandas")
        self.assertTrue(status["enabled"])
        cp.empty.assert_called_once_with((1,), dtype=cp.uint8)
        probe.fill.assert_called_once_with(1)
        cp.cuda.runtime.deviceSynchronize.assert_called_once_with()
        cudf_pandas.install.assert_called_once_with()

    def test_unusable_optional_cupy_falls_back_without_importing_cudf(self):
        cp = types.SimpleNamespace(
            uint8=object(),
            empty=mock.Mock(side_effect=RuntimeError("driver mismatch")),
            cuda=types.SimpleNamespace(runtime=types.SimpleNamespace(
                getDeviceCount=mock.Mock(return_value=1),
                deviceSynchronize=mock.Mock(),
            )),
        )
        importer = mock.Mock(side_effect=lambda name: cp if name == "cupy" else (_ for _ in ()).throw(AssertionError("cudf imported")))
        status = policy.enable_optional_dataframe_acceleration(
            {"CUDA_VISIBLE_DEVICES": "0"}, importer=importer
        )
        self.assertEqual(status["backend"], "pandas")
        self.assertEqual(status["reason"], "cupy_unusable")
        self.assertEqual(importer.call_args_list, [mock.call("cupy")])

    def test_retained_apps_bootstrap_dataframe_acceleration_before_pandas(self):
        for name in APP_FILES:
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("enable_optional_dataframe_acceleration,", source, name)
            bootstrap = source.index("DATAFRAME_ACCELERATION = enable_optional_dataframe_acceleration()")
            pandas_import = source.index("import pandas as pd")
            self.assertLess(bootstrap, pandas_import, name)
    def test_cpu_admission_never_calls_spacy_gpu_preference(self):
        spacy = types.SimpleNamespace(prefer_gpu=mock.Mock(side_effect=AssertionError("spaCy GPU touched")))
        status = policy.configure_optional_spacy_gpu(
            spacy, {"TRAINING_CONTROL_BACKEND": "cpu", "CUDA_VISIBLE_DEVICES": "0"}
        )
        self.assertEqual(status["backend"], "cpu")
        self.assertFalse(status["requested"])
        spacy.prefer_gpu.assert_not_called()

    def test_spacy_gpu_is_preferred_when_available_but_optional(self):
        spacy = types.SimpleNamespace(prefer_gpu=mock.Mock(return_value=True))
        status = policy.configure_optional_spacy_gpu(spacy, {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertTrue(status["enabled"])
        self.assertEqual(status["backend"], "gpu")
        spacy.prefer_gpu.assert_called_once_with()
        spacy = types.SimpleNamespace(prefer_gpu=mock.Mock(return_value=False))
        status = policy.configure_optional_spacy_gpu(spacy, {"CUDA_VISIBLE_DEVICES": "0"})
        self.assertFalse(status["enabled"])
        self.assertEqual(status["backend"], "cpu")

    def test_retained_apps_prefer_spacy_gpu_before_loading_model(self):
        for name in APP_FILES:
            source = (ROOT / name).read_text(encoding="utf-8")
            configure = source.index("SPACY_ACCELERATION = configure_optional_spacy_gpu(spacy)")
            model_load = source.index('nlp = spacy.load("en_core_web_sm")')
            self.assertLess(configure, model_load, name)
    def test_transformers_device_convention(self):
        self.assertEqual(policy.transformers_pipeline_device(Device("cpu")), -1)
        self.assertEqual(policy.transformers_pipeline_device(Device("cuda")), 0)
        self.assertEqual(policy.transformers_pipeline_device(Device("cuda:3")), 3)

    def test_gpu_admitted_runtime_failures_are_not_swallowed_by_retained_apps(self):
        required_messages = (
            'Error processing sub-batch: {e}',
            "Error in 'addEmotions': {e}",
            'Error processing with model {model}: {e}',
            'Error initializing device: {e}',
            'Error initializing DataProcessor or FileHandler: {e}',
            'Error during file creation: {e}',
        )
        for name in APP_FILES:
            source = (ROOT / name).read_text(encoding="utf-8")
            self.assertIn("gpu_admission_requested,", source, name)
            for message in required_messages:
                index = source.index(message)
                guard = source[max(0, index - 180):index]
                self.assertIn("if gpu_admission_requested():", guard, (name, message))
                self.assertIn("raise", guard, (name, message))

    def test_both_retained_apps_use_one_resolved_device_without_direct_cuda_probe(self):
        for name in APP_FILES:
            source = (ROOT / name).read_text(encoding="utf-8")
            tree = ast.parse(source)
            self.assertIn("from inference_device_policy import", source, name)
            self.assertNotIn("torch.cuda.", source, name)
            self.assertIn("device = resolve_torch_device(torch)", source, name)
            self.assertIn(
                "device=transformers_pipeline_device(self.device)", source, name
            )
            calls = [
                node for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "resolve_torch_device"
            ]
            self.assertEqual(len(calls), 1, name)


if __name__ == "__main__":
    unittest.main()
