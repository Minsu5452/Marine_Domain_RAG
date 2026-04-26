"""EXAONE-3.5-2.4B-Instruct LLM 래퍼.

권장 백엔드: llama-cpp-python + GGUF (CPU). Mac 에서 가장 안정적이고 메모리 효율적.
fallback: transformers (HuggingFace AutoModel). MPS+SBERT 동시 로드 시 SIGSEGV 가능.
키가 없거나 로드 실패 시 mock LLM 으로 fallback.
"""

from __future__ import annotations

import logging
import os
from typing import Iterable

logger = logging.getLogger(__name__)


class MockLLM:
    """LLM 다운로드/로드가 불가할 때 fallback. 컨텍스트만 echo."""

    def __init__(self, name: str = "mock") -> None:
        self.name = name

    def generate(self, system: str, user: str, *, max_new_tokens: int = 512,
                 temperature: float = 0.2) -> str:
        return ("[MOCK LLM 응답]\n"
                "사용자 질문: " + user[:200] + "\n"
                "(실제 LLM 미적용 — 컨텍스트 기반 retrieve 결과만 노출됩니다)")


class ExaoneGGUFLLM:
    """llama-cpp-python 기반 GGUF inference. CPU 안정 + 빠름.

    HF repo: LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF
    파일: Q4_K_M (1.5GB) 추천 — Mac M-series CPU 에서 토큰 수십개/초.
    """

    def __init__(self, repo_id: str = "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
                 filename: str = "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
                 n_ctx: int = 4096, n_threads: int | None = None) -> None:
        # llama-cpp + sentence-transformers(BLAS) 가 같은 프로세스에서 충돌 (libomp 중복) 하지 않도록
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        os.environ.setdefault("OMP_NUM_THREADS", "1")

        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = hf_hub_download(repo_id=repo_id, filename=filename)
        self.llm = Llama(
            model_path=path,
            n_ctx=n_ctx,
            n_threads=n_threads or max(os.cpu_count() // 2, 1),
            verbose=False,
        )
        self.name = f"{repo_id}::{filename}"

    def generate(self, system: str, user: str, *, max_new_tokens: int = 256,
                 temperature: float = 0.2) -> str:
        out = self.llm.create_chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return out["choices"][0]["message"]["content"].strip()


class ExaoneLLM:
    def __init__(self, model_name: str, *, device: str = "auto", dtype: str = "float16") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        if device == "auto":
            if torch.backends.mps.is_available():
                device = "mps"
            elif torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        torch_dtype = {"float16": torch.float16,
                       "bfloat16": torch.bfloat16,
                       "float32": torch.float32}.get(dtype, torch.float16)
        if device == "cpu":
            torch_dtype = torch.float32

        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype,
            device_map=device,
            trust_remote_code=True,
        )
        self.model.eval()
        self.name = model_name

    def generate(self, system: str, user: str, *, max_new_tokens: int = 512,
                 temperature: float = 0.2) -> str:
        import torch

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        try:
            chat = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
            )
            if isinstance(chat, torch.Tensor):
                input_ids = chat.to(self.device)
            elif hasattr(chat, "input_ids"):
                input_ids = chat["input_ids"].to(self.device)
            else:
                input_ids = torch.tensor(chat, device=self.device)
                if input_ids.dim() == 1:
                    input_ids = input_ids.unsqueeze(0)
        except Exception:  # noqa: BLE001
            prompt = f"[SYSTEM]\n{system}\n[USER]\n{user}\n[ASSISTANT]\n"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)

        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            out = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=self.tokenizer.eos_token_id,
            )
        gen = out[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()


def load_llm(provider: str, model_name: str, *, device: str = "auto",
             dtype: str = "float16", gguf_repo_id: str | None = None,
             gguf_filename: str | None = None):
    if provider == "mock":
        return MockLLM()
    if provider == "exaone_gguf":
        try:
            return ExaoneGGUFLLM(
                repo_id=gguf_repo_id or "LGAI-EXAONE/EXAONE-3.5-2.4B-Instruct-GGUF",
                filename=gguf_filename or "EXAONE-3.5-2.4B-Instruct-Q4_K_M.gguf",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("EXAONE GGUF 로드 실패 (%s) — MockLLM fallback", e)
            return MockLLM()
    if provider == "exaone":
        try:
            return ExaoneLLM(model_name, device=device, dtype=dtype)
        except Exception as e:  # noqa: BLE001
            logger.warning("EXAONE 로드 실패 (%s) — MockLLM fallback", e)
            return MockLLM()
    logger.warning("unknown llm provider %s — MockLLM fallback", provider)
    return MockLLM()
