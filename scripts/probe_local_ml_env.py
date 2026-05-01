#!/usr/bin/env python3
"""Probe local CPU/GPU capability for the local haiku model path.

The probe is intentionally read-only: it imports optional Python libraries when
present and runs common discovery commands if they already exist, but it never
installs packages or changes system configuration. Missing GPU support is
reported as the normal CPU fallback.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import importlib.metadata
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from typing import Any, Sequence


BASELINE_DEPENDENCIES = ("python>=3.10", "standard-library")
OPTIONAL_DEPENDENCIES = (
    "torch with ROCm/HIP for the GRU/LSTM follow-up",
    "onnxruntime-rocm for ONNX ROCm experiments",
    "llama.cpp/llama-cpp-python with Vulkan for llama-vulkan experiments",
)
ML_PACKAGES = (
    ("numpy", "numpy"),
    ("torch", "torch"),
    ("onnxruntime", "onnxruntime"),
    ("llama_cpp", "llama-cpp-python"),
)
GPU_ENV_VARS = (
    "HIP_VISIBLE_DEVICES",
    "ROCR_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
    "HSA_OVERRIDE_GFX_VERSION",
    "PYTORCH_ROCM_ARCH",
)


@dataclass(frozen=True)
class CommandResult:
    """Result of a bounded local command probe."""

    command: str
    available: bool
    returncode: int | None = None
    output: str = ""
    error: str = ""


@dataclass(frozen=True)
class LibraryInfo:
    """Import/package status for a relevant Python library."""

    module: str
    package: str
    installed: bool
    version: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class TorchInfo:
    """PyTorch acceleration details, if torch is installed."""

    installed: bool
    version: str | None = None
    hip_version: str | None = None
    cuda_available: bool | None = None
    device_count: int | None = None
    devices: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class OnnxInfo:
    """ONNX Runtime provider details, if onnxruntime is installed."""

    installed: bool
    version: str | None = None
    providers: tuple[str, ...] = ()
    error: str | None = None


@dataclass(frozen=True)
class HardwareReport:
    """Structured capability report consumed by CLI output and tests."""

    classification: str
    baseline_ready: bool
    cpu_training_required: bool
    os: dict[str, str]
    python: dict[str, str]
    cpu: dict[str, str | int | None]
    memory: dict[str, int | None]
    gpu: dict[str, Any]
    rocm: dict[str, Any]
    vulkan: dict[str, Any]
    libraries: dict[str, LibraryInfo]
    torch: TorchInfo
    onnxruntime: OnnxInfo
    environment: dict[str, str]
    required_dependencies: tuple[str, ...] = BASELINE_DEPENDENCIES
    optional_acceleration_dependencies: tuple[str, ...] = OPTIONAL_DEPENDENCIES
    next_steps: tuple[str, ...] = ()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Print local CPU/GPU capability for local haiku training."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit the full report as JSON instead of the human summary",
    )
    args = parser.parse_args(argv)

    report = probe_environment()
    if args.json:
        print(json.dumps(report_to_jsonable(report), indent=2, sort_keys=True))
    else:
        print(format_report(report))
    return 0


def probe_environment() -> HardwareReport:
    """Collect a non-invasive hardware and optional-ML capability report."""

    command_results = _probe_commands()
    libraries = _probe_libraries()
    torch_info = _probe_torch(libraries["torch"].installed)
    onnx_info = _probe_onnxruntime(libraries["onnxruntime"].installed)
    gpu_info = _gpu_visibility(command_results)
    rocm_info = _rocm_status(command_results, gpu_info)
    vulkan_info = _vulkan_status(command_results)
    classification = classify_environment(
        torch_info=torch_info,
        onnx_info=onnx_info,
        gpu_info=gpu_info,
        rocm_info=rocm_info,
        vulkan_info=vulkan_info,
        libraries=libraries,
    )

    report = HardwareReport(
        classification=classification,
        baseline_ready=_baseline_ready(),
        cpu_training_required=True,
        os=_os_info(),
        python=_python_info(),
        cpu=_cpu_info(),
        memory=_memory_info(),
        gpu=gpu_info,
        rocm=rocm_info,
        vulkan=vulkan_info,
        libraries=libraries,
        torch=torch_info,
        onnxruntime=onnx_info,
        environment={name: os.environ[name] for name in GPU_ENV_VARS if name in os.environ},
        next_steps=(),
    )
    return _with_next_steps(report)


def classify_environment(
    *,
    torch_info: TorchInfo,
    onnx_info: OnnxInfo,
    gpu_info: dict[str, Any],
    rocm_info: dict[str, Any],
    vulkan_info: dict[str, Any],
    libraries: dict[str, LibraryInfo],
) -> str:
    """Classify the best supported local acceleration path."""

    gpu_visible = bool(gpu_info.get("amd_gpu_visible") or gpu_info.get("gpu_devices"))
    rocm_visible = bool(rocm_info.get("available") or torch_info.hip_version)
    if (
        torch_info.installed
        and torch_info.hip_version
        and torch_info.cuda_available
        and (torch_info.device_count or 0) > 0
    ):
        return "rocm-pytorch"
    if (
        onnx_info.installed
        and "ROCMExecutionProvider" in onnx_info.providers
        and (gpu_visible or rocm_visible)
    ):
        return "onnx-rocm"
    if (
        vulkan_info.get("available")
        and gpu_visible
        and libraries.get(
            "llama_cpp", LibraryInfo("llama_cpp", "llama-cpp-python", False)
        ).installed
    ):
        return "llama-vulkan"
    if _baseline_ready():
        return "cpu"
    return "unknown"


def format_report(report: HardwareReport) -> str:
    """Render a concise human-readable report."""

    lines = [
        "Local haiku ML environment probe",
        "================================",
        f"Classification: {report.classification}",
        f"CPU n-gram baseline ready: {'yes' if report.baseline_ready else 'no'}",
        "CPU n-gram training requirement: must work even when GPU acceleration is absent",
        "",
        "System",
        f"- OS: {report.os['system']} {report.os['release']} ({report.os['machine']})",
        f"- Python: {report.python['version']} at {report.python['executable']}",
        f"- CPU: {report.cpu['processor'] or 'unknown'}; logical cores={report.cpu['logical_cores']}",
        f"- RAM: {_format_bytes(report.memory.get('total_bytes'))}",
        f"- Available RAM: {_format_bytes(report.memory.get('available_bytes'))}",
        "",
        "Required baseline dependencies",
        *[f"- {dependency}" for dependency in report.required_dependencies],
        "",
        "Optional acceleration dependencies",
        *[f"- {dependency}" for dependency in report.optional_acceleration_dependencies],
        "",
        "Python ML libraries",
    ]

    for info in report.libraries.values():
        status = f"installed ({info.version})" if info.installed else "not installed"
        if info.error:
            status = f"{status}; probe error: {info.error}"
        lines.append(f"- {info.package}: {status}")

    lines.extend(
        [
            "",
            "GPU and ROCm visibility",
            f"- AMD GPU visible: {'yes' if report.gpu.get('amd_gpu_visible') else 'no'}",
            f"- GPU devices: {', '.join(report.gpu.get('gpu_devices') or ()) or 'none detected'}",
            f"- /dev/kfd: {'present' if report.gpu.get('dev_kfd') else 'absent'}",
            f"- /dev/dri: {'present' if report.gpu.get('dev_dri') else 'absent'}",
            f"- ROCm command visibility: {'yes' if report.rocm.get('available') else 'no'}",
            f"- HIP version: {report.rocm.get('hip_version') or report.torch.hip_version or 'unknown'}",
            f"- PyTorch ROCm usable: {_torch_rocm_summary(report.torch)}",
            f"- ONNX Runtime providers: {', '.join(report.onnxruntime.providers) or 'unknown/not installed'}",
            f"- Vulkan visible: {'yes' if report.vulkan.get('available') else 'no'}",
        ]
    )

    if report.environment:
        lines.append("")
        lines.append("Relevant environment variables")
        for key, value in report.environment.items():
            lines.append(f"- {key}={value}")

    lines.append("")
    lines.append("Actionable next steps")
    lines.extend(f"- {step}" for step in report.next_steps)
    return "\n".join(lines)


def report_to_jsonable(report: HardwareReport) -> dict[str, Any]:
    """Convert nested dataclasses to JSON-safe dictionaries."""

    return asdict(report)


def _probe_libraries() -> dict[str, LibraryInfo]:
    libraries: dict[str, LibraryInfo] = {}
    for module, package in ML_PACKAGES:
        installed = importlib.util.find_spec(module) is not None
        version = None
        error = None
        if installed:
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = "unknown"
            except Exception as exc:  # pragma: no cover - defensive probe guard
                error = str(exc)
        libraries[module] = LibraryInfo(module, package, installed, version, error)
    return libraries


def _probe_torch(installed: bool) -> TorchInfo:
    if not installed:
        return TorchInfo(installed=False)

    try:
        import torch  # type: ignore[import-not-found]

        device_count = torch.cuda.device_count()
        devices: list[str] = []
        for index in range(device_count):
            try:
                devices.append(torch.cuda.get_device_name(index))
            except Exception as exc:  # pragma: no cover - hardware-specific
                devices.append(f"device {index} (name probe failed: {exc})")
        return TorchInfo(
            installed=True,
            version=getattr(torch, "__version__", "unknown"),
            hip_version=getattr(torch.version, "hip", None),
            cuda_available=torch.cuda.is_available(),
            device_count=device_count,
            devices=tuple(devices),
        )
    except Exception as exc:
        return TorchInfo(installed=True, error=str(exc))


def _probe_onnxruntime(installed: bool) -> OnnxInfo:
    if not installed:
        return OnnxInfo(installed=False)

    try:
        import onnxruntime  # type: ignore[import-not-found]

        return OnnxInfo(
            installed=True,
            version=getattr(onnxruntime, "__version__", "unknown"),
            providers=tuple(onnxruntime.get_available_providers()),
        )
    except Exception as exc:
        return OnnxInfo(installed=True, error=str(exc))


def _probe_commands() -> dict[str, CommandResult]:
    return {
        "lspci": _run_command(("lspci",)),
        "rocminfo": _run_command(("rocminfo",)),
        "rocm-smi": _run_command(("rocm-smi", "--showproductname")),
        "hipconfig": _run_command(("hipconfig", "--version")),
        "vulkaninfo": _run_command(("vulkaninfo", "--summary")),
    }


def _run_command(command: Sequence[str], timeout_seconds: int = 5) -> CommandResult:
    executable = shutil.which(command[0])
    if executable is None:
        return CommandResult(" ".join(command), available=False)

    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except Exception as exc:
        return CommandResult(" ".join(command), available=True, error=str(exc))

    output = "\n".join(part for part in (completed.stdout, completed.stderr) if part).strip()
    return CommandResult(
        " ".join(command),
        available=True,
        returncode=completed.returncode,
        output=_trim_output(output),
    )


def _gpu_visibility(command_results: dict[str, CommandResult]) -> dict[str, Any]:
    devices: list[str] = []
    lspci = command_results["lspci"]
    if lspci.available and lspci.output:
        for line in lspci.output.splitlines():
            lowered = line.lower()
            if any(kind in lowered for kind in ("vga", "3d controller", "display")):
                devices.append(line.strip())

    rocm_smi = command_results["rocm-smi"]
    if rocm_smi.available and rocm_smi.output:
        for line in rocm_smi.output.splitlines():
            if "card" in line.lower() or "gpu" in line.lower() or "product" in line.lower():
                candidate = line.strip(" |")
                if candidate and candidate not in devices:
                    devices.append(candidate)

    amd_visible = any(("amd" in device.lower() or "ati" in device.lower()) for device in devices)
    return {
        "amd_gpu_visible": amd_visible,
        "gpu_devices": tuple(devices),
        "dev_kfd": os.path.exists("/dev/kfd"),
        "dev_dri": os.path.isdir("/dev/dri"),
    }


def _rocm_status(
    command_results: dict[str, CommandResult], gpu_info: dict[str, Any]
) -> dict[str, Any]:
    rocminfo = command_results["rocminfo"]
    rocm_smi = command_results["rocm-smi"]
    hipconfig = command_results["hipconfig"]
    hip_version = None
    if hipconfig.available and hipconfig.output:
        hip_version = hipconfig.output.splitlines()[0].strip()

    return {
        "available": bool(
            (rocminfo.available and rocminfo.returncode == 0)
            or (rocm_smi.available and rocm_smi.returncode == 0)
        ),
        "hip_version": hip_version,
        "rocminfo": asdict(rocminfo),
        "rocm_smi": asdict(rocm_smi),
        "has_kernel_device": bool(gpu_info.get("dev_kfd")),
    }


def _vulkan_status(command_results: dict[str, CommandResult]) -> dict[str, Any]:
    vulkaninfo = command_results["vulkaninfo"]
    devices: list[str] = []
    if vulkaninfo.available and vulkaninfo.output:
        for line in vulkaninfo.output.splitlines():
            normalized = line.lower().replace(" ", "")
            if "devicename" in normalized or "gpuid" in normalized:
                devices.append(line.strip())
    return {
        "available": bool(vulkaninfo.available and vulkaninfo.returncode == 0),
        "devices": tuple(devices),
        "vulkaninfo": asdict(vulkaninfo),
    }


def _with_next_steps(report: HardwareReport) -> HardwareReport:
    steps: list[str] = [
        "Use CPU for the first n-gram baseline; GPU support is optional and must not block training.",
    ]
    if report.classification == "rocm-pytorch":
        steps.append(
            "PyTorch reports HIP devices; the GRU/LSTM follow-up is worth testing with torch ROCm."
        )
    elif report.classification == "onnx-rocm":
        steps.append(
            "ONNX Runtime exposes ROCMExecutionProvider; keep PyTorch GRU/LSTM on CPU unless torch also reports HIP devices."
        )
    elif report.classification == "llama-vulkan":
        steps.append(
            "Vulkan and llama-cpp-python are visible; this is useful for llama.cpp experiments, not the CPU n-gram baseline."
        )
    elif report.classification == "cpu":
        if not report.gpu.get("amd_gpu_visible"):
            steps.append("No AMD GPU was detected by the available probes.")
        if not report.rocm.get("available"):
            steps.append(
                "ROCm tools/runtime were not visible; install/configure ROCm manually before trying PyTorch ROCm."
            )
        if not report.torch.installed:
            steps.append("PyTorch is not installed; defer it until the GRU/LSTM follow-up needs it.")
        elif not report.torch.hip_version:
            steps.append(
                "Installed PyTorch does not report HIP support; use a ROCm PyTorch build for AMD GPU experiments."
            )
        elif not report.torch.cuda_available:
            steps.append(
                "PyTorch has a HIP build but reports no usable device; check ROCm device permissions and GPU support."
            )
    else:
        steps.append("Baseline requirements are incomplete; use a supported Python 3 interpreter before local training.")

    memory_total = report.memory.get("total_bytes")
    if memory_total is not None and memory_total < 8 * 1024**3:
        steps.append("System RAM is below 8 GiB; keep n-gram runs small and defer neural training.")
    return replace(report, next_steps=tuple(steps))


def _baseline_ready() -> bool:
    return sys.version_info >= (3, 10)


def _os_info() -> dict[str, str]:
    return {
        "system": platform.system() or "unknown",
        "release": platform.release() or "unknown",
        "version": platform.version() or "unknown",
        "machine": platform.machine() or "unknown",
    }


def _python_info() -> dict[str, str]:
    return {
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "executable": sys.executable,
    }


def _cpu_info() -> dict[str, str | int | None]:
    processor = platform.processor()
    if not processor and os.path.exists("/proc/cpuinfo"):
        try:
            with open("/proc/cpuinfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    if line.lower().startswith("model name"):
                        processor = line.split(":", 1)[1].strip()
                        break
        except OSError:
            processor = ""
    return {
        "processor": processor or None,
        "machine": platform.machine() or None,
        "logical_cores": os.cpu_count(),
    }


def _memory_info() -> dict[str, int | None]:
    total = None
    available = None
    if os.path.exists("/proc/meminfo"):
        try:
            values: dict[str, int] = {}
            with open("/proc/meminfo", "r", encoding="utf-8") as handle:
                for line in handle:
                    key, raw_value = line.split(":", 1)
                    parts = raw_value.strip().split()
                    if parts and parts[0].isdigit():
                        values[key] = int(parts[0]) * 1024
            total = values.get("MemTotal")
            available = values.get("MemAvailable")
        except (OSError, ValueError):
            pass
    return {"total_bytes": total, "available_bytes": available}


def _trim_output(output: str, limit: int = 6000) -> str:
    if len(output) <= limit:
        return output
    return output[:limit].rstrip() + "\n...[trimmed]"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "unknown"
    gib = value / 1024**3
    return f"{gib:.1f} GiB"


def _torch_rocm_summary(torch_info: TorchInfo) -> str:
    if not torch_info.installed:
        return "no; torch not installed"
    if torch_info.error:
        return f"unknown; torch probe failed: {torch_info.error}"
    if not torch_info.hip_version:
        return "no; installed torch does not report HIP"
    if not torch_info.cuda_available or not torch_info.device_count:
        return "no; HIP build present but no usable device"
    return f"yes; {torch_info.device_count} device(s): {', '.join(torch_info.devices) or 'unnamed'}"


if __name__ == "__main__":
    raise SystemExit(main())
