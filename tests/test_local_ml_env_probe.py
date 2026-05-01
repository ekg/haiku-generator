"""Tests for the local ML hardware probe."""

import os
import sys


sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from probe_local_ml_env import (  # noqa: E402
    LibraryInfo,
    OnnxInfo,
    TorchInfo,
    classify_environment,
    format_report,
    probe_environment,
    report_to_jsonable,
)


def libraries(llama_cpp: bool = False) -> dict[str, LibraryInfo]:
    return {
        "numpy": LibraryInfo("numpy", "numpy", False),
        "torch": LibraryInfo("torch", "torch", False),
        "onnxruntime": LibraryInfo("onnxruntime", "onnxruntime", False),
        "llama_cpp": LibraryInfo("llama_cpp", "llama-cpp-python", llama_cpp),
    }


def test_classifies_rocm_pytorch_when_torch_has_hip_device():
    classification = classify_environment(
        torch_info=TorchInfo(
            installed=True,
            version="2.4.0",
            hip_version="6.1",
            cuda_available=True,
            device_count=1,
            devices=("AMD Radeon",),
        ),
        onnx_info=OnnxInfo(installed=False),
        gpu_info={"amd_gpu_visible": True, "gpu_devices": ("AMD GPU",)},
        rocm_info={"available": True},
        vulkan_info={"available": False},
        libraries=libraries(),
    )

    assert classification == "rocm-pytorch"


def test_classifies_onnx_rocm_without_torch_rocm():
    classification = classify_environment(
        torch_info=TorchInfo(installed=False),
        onnx_info=OnnxInfo(
            installed=True,
            version="1.18.0",
            providers=("ROCMExecutionProvider", "CPUExecutionProvider"),
        ),
        gpu_info={"amd_gpu_visible": True, "gpu_devices": ("AMD GPU",)},
        rocm_info={"available": True},
        vulkan_info={"available": False},
        libraries=libraries(),
    )

    assert classification == "onnx-rocm"


def test_classifies_llama_vulkan_when_vulkan_and_llama_are_visible():
    classification = classify_environment(
        torch_info=TorchInfo(installed=False),
        onnx_info=OnnxInfo(installed=False),
        gpu_info={"amd_gpu_visible": True, "gpu_devices": ("AMD GPU",)},
        rocm_info={"available": False},
        vulkan_info={"available": True},
        libraries=libraries(llama_cpp=True),
    )

    assert classification == "llama-vulkan"


def test_cpu_fallback_is_normal_and_actionable():
    report = probe_environment()
    rendered = format_report(report)
    jsonable = report_to_jsonable(report)

    assert report.baseline_ready
    assert report.cpu_training_required
    assert report.classification in {
        "cpu",
        "rocm-pytorch",
        "onnx-rocm",
        "llama-vulkan",
        "unknown",
    }
    assert "CPU n-gram training requirement" in rendered
    assert "Required baseline dependencies" in rendered
    assert "Optional acceleration dependencies" in rendered
    assert jsonable["cpu_training_required"] is True
