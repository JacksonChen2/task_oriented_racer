"""Frozen, local-only Qwen global guidance with non-blocking replacement."""

from __future__ import annotations

import json
import os
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .schemas import GlobalGuidance
from .single_gpu_pause import SingleGpuPauseController


@dataclass(frozen=True)
class QwenConfig:
    model_path: str
    backend: str = "transformers"
    device: str = "auto"
    dtype: str = "bfloat16"
    local_files_only: bool = True
    max_new_tokens: int = 1024
    normalize_omega: bool = True
    dequantize_fp8: bool = True
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    prompt_variant: str = "legacy_full_schema"
    vllm_quantization: str = "fp8"
    vllm_linear_backend: str = "cutlass"
    vllm_gpu_memory_utilization: float = 0.25
    vllm_max_model_len: int = 8192
    vllm_max_num_seqs: int = 1
    vllm_enforce_eager: bool = False
    vllm_enable_prefix_caching: bool = False
    vllm_disable_deep_gemm: bool = True
    vllm_disable_flashinfer_sampler: bool = True
    vllm_seed: int = 0

    def validate(self) -> "QwenConfig":
        path = Path(self.model_path).expanduser().resolve()
        if not path.is_dir() or not (path / "config.json").is_file():
            raise FileNotFoundError(f"local Qwen checkpoint is invalid: {path}")
        if self.load_in_8bit and self.load_in_4bit:
            raise ValueError("8-bit and 4-bit loading are mutually exclusive")
        backend = self.backend.lower()
        if backend not in {"transformers", "vllm"}:
            raise ValueError("Qwen backend must be transformers or vllm")
        if self.dtype.lower() not in {
            "bfloat16",
            "float16",
            "fp16",
            "float32",
        }:
            raise ValueError(f"unsupported Qwen dtype: {self.dtype}")
        if backend == "vllm":
            if self.device.lower() not in {"auto", "cuda"}:
                raise ValueError("vLLM Qwen backend requires device auto or cuda")
            if self.dequantize_fp8:
                raise ValueError(
                    "dequantize_fp8 must be false when Qwen backend is vllm"
                )
            if self.load_in_8bit or self.load_in_4bit:
                raise ValueError(
                    "Transformers load_in_8bit/load_in_4bit flags cannot be "
                    "used with vLLM"
                )
            if self.vllm_quantization.lower() not in {"auto", "fp8"}:
                raise ValueError("vLLM quantization must be auto or fp8")
            if not 0.0 < self.vllm_gpu_memory_utilization <= 1.0:
                raise ValueError(
                    "vLLM gpu_memory_utilization must be in (0, 1]"
                )
            if self.vllm_max_model_len <= 0 or self.vllm_max_num_seqs <= 0:
                raise ValueError(
                    "vLLM max_model_len and max_num_seqs must be positive"
                )
            if not self.vllm_linear_backend.strip():
                raise ValueError("vLLM linear_backend must not be empty")
        if self.prompt_variant not in {
            "legacy_full_schema",
            "compact_nonzero_schema",
        }:
            raise ValueError(
                "prompt_variant must be legacy_full_schema or "
                "compact_nonzero_schema"
            )
        return self


def qwen_config_from_mapping(
    value: dict[str, Any], *, seed: int
) -> QwenConfig:
    """Build one validated model configuration for either process topology."""

    vllm = value.get("vllm", {})
    # Checkpoints are intentionally external to Git.  Both names are accepted
    # so the same cloned framework can be moved between machines without
    # editing every experiment YAML.  The generic name wins when both exist.
    model_path = (
        os.environ.get("RACER_QWEN_MODEL_PATH", "").strip()
        or os.environ.get("RACER_QWEN8_MODEL_PATH", "").strip()
        or str(value["model_path"])
    )
    return QwenConfig(
        model_path=model_path,
        backend=str(value.get("backend", "transformers")),
        device=str(value.get("device", "auto")),
        dtype=str(value.get("dtype", "bfloat16")),
        local_files_only=bool(value.get("local_files_only", True)),
        max_new_tokens=int(value.get("max_new_tokens", 1024)),
        normalize_omega=bool(value.get("normalize_omega", True)),
        dequantize_fp8=bool(value.get("dequantize_fp8", True)),
        load_in_8bit=bool(value.get("load_in_8bit", False)),
        load_in_4bit=bool(value.get("load_in_4bit", False)),
        prompt_variant=str(value.get("prompt_variant", "legacy_full_schema")),
        vllm_quantization=str(vllm.get("quantization", "fp8")),
        vllm_linear_backend=str(vllm.get("linear_backend", "cutlass")),
        vllm_gpu_memory_utilization=float(
            vllm.get("gpu_memory_utilization", 0.25)
        ),
        vllm_max_model_len=int(vllm.get("max_model_len", 8192)),
        vllm_max_num_seqs=int(vllm.get("max_num_seqs", 1)),
        vllm_enforce_eager=bool(vllm.get("enforce_eager", False)),
        vllm_enable_prefix_caching=bool(
            vllm.get("enable_prefix_caching", False)
        ),
        vllm_disable_deep_gemm=bool(
            vllm.get("disable_deep_gemm", True)
        ),
        vllm_disable_flashinfer_sampler=bool(
            vllm.get("disable_flashinfer_sampler", True)
        ),
        vllm_seed=int(vllm.get("seed", seed)),
    )


def _numbered_entries(section: str) -> list[tuple[int, str]]:
    """Parse whitespace-separated ``id:value`` entries without ignoring junk."""

    cleaned = section.strip()
    pattern = re.compile(
        r"(\d+)\s*[:：]\s*(.*?)(?=\s+\d+\s*[:：]|$)", re.DOTALL
    )
    entries: list[tuple[int, str]] = []
    cursor = 0
    for match in pattern.finditer(cleaned):
        if cleaned[cursor : match.start()].strip():
            raise ValueError("Qwen guidance contains invalid entry text")
        entries.append((int(match.group(1)), match.group(2).strip()))
        cursor = match.end()
    if cleaned[cursor:].strip():
        raise ValueError("Qwen guidance contains invalid trailing text")
    return entries


def parse_guidance(
    text: str, n_uavs: int, normalize_omega: bool = True
) -> GlobalGuidance:
    match = re.fullmatch(
        r"\s*UAV\s+links\s*[:：]\s*(.*?)\s*"
        r"BS\s+priority\s*[:：]\s*(.*?)\s*",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if match is None:
        raise ValueError(
            "Qwen output must contain exactly UAV links and BS priority"
        )

    expected_targets = min(4, max(0, n_uavs - 1))
    dependency = np.zeros((n_uavs, n_uavs), dtype=np.float32)
    seen_sources: set[int] = set()
    for source_id, targets_text in _numbered_entries(match.group(1)):
        if not 1 <= source_id <= n_uavs:
            raise ValueError("UAV links source ID is outside the valid range")
        if source_id in seen_sources:
            raise ValueError("UAV links contains a duplicate source ID")
        seen_sources.add(source_id)
        raw_targets = [value.strip() for value in targets_text.split(",")]
        if expected_targets == 0 and raw_targets == [""]:
            raw_targets = []
        if len(raw_targets) != expected_targets or any(
            not value.isdigit() for value in raw_targets
        ):
            raise ValueError(
                f"UAV {source_id} must list exactly {expected_targets} target IDs"
            )
        targets = [int(value) for value in raw_targets]
        if any(not 1 <= target <= n_uavs for target in targets):
            raise ValueError("UAV links target ID is outside the valid range")
        if source_id in targets:
            raise ValueError("UAV links cannot contain self-links")
        if len(set(targets)) != len(targets):
            raise ValueError("UAV links cannot repeat a target for one source")
        for rank, target_id in enumerate(targets):
            dependency[source_id - 1, target_id - 1] = (
                expected_targets - rank
            ) / max(expected_targets, 1)
    if seen_sources != set(range(1, n_uavs + 1)):
        raise ValueError("UAV links must contain every source UAV exactly once")

    priorities = np.zeros(n_uavs, dtype=np.float32)
    seen_priority_ids: set[int] = set()
    for uav_id, priority_text in _numbered_entries(match.group(2)):
        if not 1 <= uav_id <= n_uavs:
            raise ValueError("BS priority UAV ID is outside the valid range")
        if uav_id in seen_priority_ids:
            raise ValueError("BS priority contains a duplicate UAV ID")
        if not priority_text.isdigit():
            raise ValueError("BS priority must be an integer from 0 to 4")
        priority = int(priority_text)
        if not 0 <= priority <= 4:
            raise ValueError("BS priority must be an integer from 0 to 4")
        seen_priority_ids.add(uav_id)
        priorities[uav_id - 1] = priority / 4.0
    if seen_priority_ids != set(range(1, n_uavs + 1)):
        raise ValueError("BS priority must contain every UAV exactly once")

    return GlobalGuidance.validated(
        dependency,
        priorities,
        n_uavs,
        normalize_omega,
    )


class QwenGlobalAgent:
    def __init__(
        self,
        config: QwenConfig,
        n_uavs: int,
        max_relay_recipients_per_uav: int | None = None,
    ) -> None:
        self.config = config.validate()
        self.n_uavs = n_uavs
        if max_relay_recipients_per_uav is not None and not (
            0 <= max_relay_recipients_per_uav <= max(0, n_uavs - 1)
        ):
            raise ValueError("invalid Qwen relay fanout limit")
        self.max_relay_recipients_per_uav = (
            max_relay_recipients_per_uav
        )
        self.tokenizer: Any = None
        self.model: Any = None
        self._vllm_sampling_params: Any = None
        self.parse_failures = 0
        self.inference_latency_s = 0.0

    def load(self) -> None:
        if self.model is not None:
            return
        if self.config.local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        if self.config.backend.lower() == "vllm":
            self._load_vllm()
        else:
            self._load_transformers()

    def _load_transformers(self) -> None:
        import torch
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            FineGrainedFP8Config,
        )

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "fp16": torch.float16,
            "float32": torch.float32,
        }.get(self.config.dtype.lower())
        if dtype is None:
            raise ValueError(f"unsupported Qwen dtype: {self.config.dtype}")
        kwargs: dict[str, Any] = {
            "local_files_only": self.config.local_files_only,
            "torch_dtype": dtype,
        }
        checkpoint_config = json.loads(
            (Path(self.config.model_path) / "config.json").read_text(
                encoding="utf-8"
            )
        )
        quantization = checkpoint_config.get("quantization_config", {})
        if (
            self.config.dequantize_fp8
            and quantization.get("quant_method") == "fp8"
        ):
            # The local checkpoint is FP8, but native Transformers FP8 kernels
            # are distributed separately. Dequantizing locally to the selected
            # BF16/FP16 dtype keeps loading offline and needs no kernel download.
            kwargs["quantization_config"] = FineGrainedFP8Config(
                dequantize=True
            )
        if self.config.load_in_8bit:
            kwargs["load_in_8bit"] = True
        if self.config.load_in_4bit:
            kwargs["load_in_4bit"] = True
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_path,
            local_files_only=self.config.local_files_only,
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_path, **kwargs
        )
        target_device = self.config.device
        if target_device == "auto":
            target_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(target_device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

    def _load_vllm(self) -> None:
        # vLLM 0.27.1 otherwise auto-selects DeepGEMM and FlashInfer sampling
        # paths that mis-handle the RTX PRO 6000 Blackwell (SM120) on this
        # machine. CUTLASS FP8 plus native sampling is verified locally.
        if self.config.vllm_disable_deep_gemm:
            os.environ["VLLM_USE_DEEP_GEMM"] = "0"
        if self.config.vllm_disable_flashinfer_sampler:
            os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"

        try:
            from vllm import LLM, SamplingParams
        except ImportError as error:
            raise RuntimeError(
                "Qwen backend vllm requires the optional vLLM dependency"
            ) from error

        checkpoint_config = json.loads(
            (Path(self.config.model_path) / "config.json").read_text(
                encoding="utf-8"
            )
        )
        checkpoint_quantization = (
            checkpoint_config.get("quantization_config") or {}
        ).get("quant_method")
        quantization = self.config.vllm_quantization.lower()
        if quantization == "fp8" and checkpoint_quantization != "fp8":
            raise ValueError(
                "vLLM FP8 backend requires a checkpoint whose "
                "quantization_config.quant_method is fp8"
            )

        kwargs: dict[str, Any] = {
            "model": self.config.model_path,
            "tokenizer": self.config.model_path,
            "dtype": {
                "fp16": "float16",
            }.get(self.config.dtype.lower(), self.config.dtype.lower()),
            "trust_remote_code": False,
            "gpu_memory_utilization": self.config.vllm_gpu_memory_utilization,
            "max_model_len": self.config.vllm_max_model_len,
            "max_num_seqs": self.config.vllm_max_num_seqs,
            "enforce_eager": self.config.vllm_enforce_eager,
            "enable_prefix_caching": self.config.vllm_enable_prefix_caching,
            "seed": self.config.vllm_seed,
        }
        if quantization != "auto":
            kwargs["quantization"] = quantization
        linear_backend = self.config.vllm_linear_backend.strip().lower()
        if linear_backend != "auto":
            kwargs["linear_backend"] = linear_backend

        self.model = LLM(**kwargs)
        self.tokenizer = self.model.get_tokenizer()
        self._vllm_sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.config.max_new_tokens,
            skip_special_tokens=True,
        )

    def _prompt(self, large_state: dict[str, Any]) -> str:
        n = self.n_uavs
        compact_prompt = self.config.prompt_variant == "compact_nonzero_schema"
        top_k = min(4, max(0, n - 1))
        links_example = " ".join(
            f"{source + 1}:"
            + ",".join(
                str((source + offset) % n + 1)
                for offset in range(1, top_k + 1)
            )
            for source in range(n)
        )
        priority_example = " ".join(
            f"{uav + 1}:{4 - uav % 5}" for uav in range(n)
        )
        schema = (
            f"UAV links: {links_example}\n"
            f"BS priority: {priority_example}"
        )
        fanout_instruction = ""
        if self.max_relay_recipients_per_uav is not None:
            fanout_instruction = (
                f"每个信息源 UAV 每个通信时隙最多可经 BS 转发给 "
                f"{self.max_relay_recipients_per_uav} 个其他 UAV；选择全局 top 链路时，"
                f"同一 source 最多出现 {self.max_relay_recipients_per_uav} 次。\n"
            )
        return (
            "你是多无人机基站辅助通信的全局任务分析器。输入是一个慢时间尺度状态："
            "所有地图与任务进度字段都只表示 BS 已成功接收并缓存的信息。"
            "bs_map_summary 是 BS 地图的覆盖、已知体素和前沿摘要；"
            "bs_exploration_progress 是 BS 地图覆盖率及最近增量；"
            "uav_bs_missing_bytes 表示各 UAV 当前拥有而 BS 缺失的字节数；"
            "bs_uav_missing_bytes 表示 BS 当前拥有而各 UAV 缺失的字节数；"
            "racer_states、positions 和 trajectory_summary 是 BS 最近获知的每机状态；"
            "coarse_communication_summary 是每机 BS 上下行可靠性、中断率及 UAV 邻居链路摘要；"
            "trajectory_summary 只含目标、长度和"
            "预计时间；region_summary 只含 BS 地图生成的质心、归一化大小、未知率和前沿统计；"
            "peer_aoi_summary 是按接收 UAV 压缩的 peer AoI，"
            "bs_aoi_s 是 BS 对每机信息的年龄。\n"
            f"UAV links 部分必须为每架源 UAV 输出 top-{top_k} 个目标 UAV 编号；"
            "排序依据是该源 UAV 的信息利用基站转发给各接收方 UAV 的任务重要性，"
            "目标 UAV 编号使用逗号分隔，并按该重要性从高到低排列；"
            "每个源 UAV 必须恰好出现一次，"
            f"目标编号范围为 1 到 {n}，不能包含源 UAV 自身或重复目标。"
            f"BS priority 部分必须为 {n} 架 UAV 分别输出一个 0 到 4 的整数，"
            "4 表示最高 BS 上传优先级，0 表示最低；每个 UAV 必须恰好出现一次。"
            + fanout_instruction
            + (
                "当 uav_bs_missing_bytes、bs_uav_missing_bytes 或未知区域非零时，"
                "BS priority 不允许全部为 0。"
                if compact_prompt and n > 1
                else ""
            )
            + (
                "严格只输出下面两行纯文本，不要输出 JSON、Markdown 或解释。格式示例：\n"
            )
            + schema
            + "\n当前状态：\n"
            + json.dumps(
                large_state, ensure_ascii=False, separators=(",", ":"), allow_nan=False
            )
        )

    def infer(self, large_state: dict[str, Any]) -> GlobalGuidance:
        self.load()
        prompt = self._prompt(large_state)
        messages = [{"role": "user", "content": prompt}]
        try:
            rendered = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            rendered = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        if self.config.backend.lower() == "vllm":
            started = time.perf_counter()
            generated = self.model.generate(
                [rendered], self._vllm_sampling_params, use_tqdm=False
            )
            self.inference_latency_s = time.perf_counter() - started
            text = generated[0].outputs[0].text
        else:
            import torch

            inputs = self.tokenizer(rendered, return_tensors="pt")
            model_device = next(self.model.parameters()).device
            inputs = {
                name: value.to(model_device) for name, value in inputs.items()
            }
            started = time.perf_counter()
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    do_sample=False,
                    use_cache=True,
                )
            self.inference_latency_s = time.perf_counter() - started
            suffix = generated[0, inputs["input_ids"].shape[1] :]
            text = self.tokenizer.decode(suffix, skip_special_tokens=True)
        try:
            return parse_guidance(
                text, self.n_uavs, normalize_omega=self.config.normalize_omega
            )
        except (TypeError, ValueError):
            self.parse_failures += 1
            raise

    def close(self) -> None:
        """Release a vLLM EngineCore instead of leaving it to atexit."""

        if self.config.backend.lower() == "vllm" and self.model is not None:
            llm_engine = getattr(self.model, "llm_engine", None)
            engine_core = getattr(llm_engine, "engine_core", None)
            shutdown = getattr(engine_core, "shutdown", None)
            if callable(shutdown):
                shutdown()
        self.model = None
        self.tokenizer = None
        self._vllm_sampling_params = None


class GuidanceManager:
    """Keep the last valid guidance and optionally update it in a worker."""

    def __init__(
        self,
        agent: QwenGlobalAgent | None,
        n_uavs: int,
        asynchronous: bool = False,
        pause_controller: SingleGpuPauseController | None = None,
    ) -> None:
        self.agent = agent
        self.asynchronous = asynchronous
        self.pause_controller = pause_controller
        self._lock = threading.Lock()
        self._guidance = GlobalGuidance.neutral(n_uavs)
        self._worker: threading.Thread | None = None
        self.epoch = 0
        self.failures = 0
        self.latency_s = 0.0

    @property
    def current(self) -> GlobalGuidance:
        with self._lock:
            return GlobalGuidance(
                self._guidance.task_dependency.copy(),
                self._guidance.semantic_importance.copy(),
            )

    @property
    def enabled(self) -> bool:
        """Whether this manager has a real large-model inference agent."""

        return self.agent is not None

    @property
    def single_gpu_pause_count(self) -> int:
        return (
            self.pause_controller.pause_count
            if self.pause_controller is not None
            else 0
        )

    @property
    def single_gpu_pause_ack_wait_s(self) -> float:
        return (
            self.pause_controller.pause_wait_s
            if self.pause_controller is not None
            else 0.0
        )

    def _run(self, state: dict[str, Any]) -> None:
        if self.agent is None:
            return
        try:
            if self.pause_controller is None:
                guidance = self.agent.infer(state)
            else:
                with self.pause_controller.paused():
                    guidance = self.agent.infer(state)
        except Exception:
            with self._lock:
                self.failures += 1
            return
        with self._lock:
            self._guidance = guidance
            self.epoch += 1
            self.latency_s = self.agent.inference_latency_s

    def request_update(self, state: dict[str, Any]) -> bool:
        if self.agent is None:
            return False
        if not self.asynchronous:
            self._run(state)
            return True
        if self._worker is not None and self._worker.is_alive():
            return False
        self._worker = threading.Thread(
            target=self._run, args=(state,), daemon=True, name="qwen-global-agent"
        )
        self._worker.start()
        return True

    def close(self) -> None:
        """Let an in-flight local generation finish before Python teardown."""

        if self.pause_controller is not None:
            self.pause_controller.cancel_pending()
        worker = self._worker
        if worker is not None and worker.is_alive():
            worker.join()
        close_agent = getattr(self.agent, "close", None)
        if callable(close_agent):
            close_agent()
