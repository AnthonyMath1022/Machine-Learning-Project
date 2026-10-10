"""Local Qwen GGUF inference through Ollama, with shared evidence validation."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

import requests
from pydantic import ValidationError

from src.models.config import OLLAMA_BASE_URL, OLLAMA_CONTEXT, OLLAMA_MODEL_ID
from src.models.model import AgentOutputError, JobMatchAgent, SYSTEM_PROMPT


# Qwen3-8B through Ollama measured about 4.0-4.2 UTF-8 bytes per token for English
# prompts and more for Chinese. This only screens out clearly oversized input.
BYTES_PER_TOKEN = 4


class OllamaJobMatchAgent(JobMatchAgent):
    """Use quantized weights without importing PyTorch or dequantizing a GGUF.

    Inherits job summaries, requirement coverage and exact-quote checks from
    JobMatchAgent. Ollama must already be running with the selected model pulled.
    """

    def __init__(
        self, model_id: str = OLLAMA_MODEL_ID, *, base_url: str = OLLAMA_BASE_URL,
        context_size: int = OLLAMA_CONTEXT, max_new_tokens: int = 2048,
        timeout: float = 300,
    ):
        if isinstance(context_size, bool) or not isinstance(context_size, int) or context_size < 1024:
            raise ValueError("context_size must be an integer of at least 1024 tokens.")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int) or max_new_tokens < 1:
            raise ValueError("max_new_tokens must be a positive integer.")
        if max_new_tokens >= context_size - 256:
            raise ValueError("Leave space in context_size for the input and chat template.")
        if isinstance(timeout, bool) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number of seconds.")
        parsed = urlsplit(base_url)
        if (parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
                or parsed.username or parsed.password or parsed.path not in {"", "/"}
                or parsed.query or parsed.fragment):
            raise ValueError("Ollama base_url must be a local HTTP address without a path or credentials.")
        super().__init__(model_id, max_new_tokens=max_new_tokens,
                         max_input_tokens=context_size - max_new_tokens - 256)
        self.base_url = base_url.rstrip("/")
        self.context_size = context_size
        self.timeout = timeout
        self._loaded = False

    def _request(self, endpoint: str, payload: dict) -> dict:
        # Ignore proxy configuration for local resume data and reject redirects.
        with requests.Session() as session:
            session.trust_env = False
            try:
                with session.post(self.base_url + endpoint, json=payload,
                                  timeout=(5, self.timeout), allow_redirects=False) as response:
                    if 300 <= response.status_code < 400:
                        raise RuntimeError("The local Ollama endpoint returned an unexpected redirect.")
                    response.raise_for_status()
                    data = response.json()
            except requests.ConnectionError as error:
                raise RuntimeError("Cannot connect to Ollama. Start Ollama, then run AI matching again.") from error
            except requests.Timeout as error:
                raise RuntimeError("Ollama inference timed out. Shorten the documents and retry.") from error
            except requests.HTTPError as error:
                if error.response is not None and error.response.status_code == 404:
                    raise RuntimeError(f"Ollama model is unavailable. Run: ollama pull {self.model_id}") from error
                status = error.response.status_code if error.response is not None else "unknown"
                if status == 400 and re.search(r"exceeds?\b.*\bcontext", error.response.text, re.IGNORECASE | re.DOTALL):
                    # Ollama counted the prompt, so the model is loaded and must be unloaded later.
                    self._loaded = True
                    needed = re.search(r'n_prompt_tokens\\*"\s*:\s*(\d+)', error.response.text)
                    detail = f"need {int(needed.group(1)):,} tokens, more than" if needed else "do not fit in"
                    raise ValueError(
                        f"Documents {detail} the {self.context_size:,}-token context; shorten them before analysis."
                    ) from error
                raise RuntimeError(f"Ollama request failed (HTTP {status}). Check the local Ollama log.") from error
            except ValueError as error:
                raise AgentOutputError("Ollama returned an invalid JSON response.") from error
        if not isinstance(data, dict):
            raise AgentOutputError("Ollama returned an invalid response object.")
        if data.get("error"):
            raise RuntimeError("Ollama could not complete inference: " + str(data["error"]))
        return data

    def _generate(self, task: str, documents: dict, schema):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                task + "\nJSON schema:\n" + json.dumps(schema.model_json_schema())
                + "\nDocuments (JSON data):\n" + json.dumps(documents, ensure_ascii=False)
            )},
        ]
        # Ollama has no public tokenize endpoint, so estimate before sending and let
        # Ollama enforce the exact limit: with truncate/shift disabled it rejects an
        # oversized prompt instead of silently dropping part of the documents.
        estimate = math.ceil(sum(len(message["content"].encode("utf-8")) for message in messages) / BYTES_PER_TOKEN)
        if estimate > self.max_input_tokens:
            raise ValueError(
                f"Documents are too long for the {self.context_size:,}-token context (about {estimate:,} "
                f"input tokens, {self.max_input_tokens:,} available); shorten them before analysis."
            )
        data = self._request("/api/chat", {
            "model": self.model_id, "messages": messages,
            "stream": False, "think": False, "format": schema.model_json_schema(),
            "keep_alive": "5m", "truncate": False, "shift": False,
            "options": {"num_ctx": self.context_size, "num_predict": self.max_new_tokens,
                        "temperature": 0, "seed": 42},
        })
        self._loaded = True
        if data.get("done") is not True or data.get("done_reason") != "stop":
            raise AgentOutputError("Ollama generation was incomplete or reached its output limit; shorten the documents and retry.")
        used = (data.get("prompt_eval_count"), data.get("eval_count"))
        if any(isinstance(count, bool) or not isinstance(count, int) for count in used):
            raise AgentOutputError("Ollama did not report token usage, so the complete documents cannot be confirmed as read.")
        # An Ollama that ignores shift=False discards prompt tokens once the context fills.
        if sum(used) > self.context_size:
            raise ValueError(f"Documents and the answer exceed the {self.context_size:,}-token context; shorten them before analysis.")
        message = data.get("message")
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            raise AgentOutputError("Ollama returned no final JSON answer.")
        try:
            return schema.model_validate_json(content)
        except ValidationError as error:
            raise AgentOutputError("Ollama did not return a complete valid assessment JSON object.") from error

    def unload(self):
        """Free Ollama's GPU allocations before OCR or a backend/model switch."""
        if self._loaded:
            self._request("/api/generate", {"model": self.model_id, "keep_alive": 0, "stream": False})
            self._loaded = False


def main() -> int:
    from src.resume.parser import extract_resume_text

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, required=True, help="UTF-8 job description (.txt)")
    parser.add_argument("--resume", type=Path, help="PDF, DOCX or UTF-8 TXT resume")
    parser.add_argument("--ocr", action="store_true", help="Use the separate OCR environment for scanned PDFs")
    parser.add_argument("--model-id", default=OLLAMA_MODEL_ID)
    args = parser.parse_args()
    agent = OllamaJobMatchAgent(args.model_id)
    try:
        job = args.job.read_text(encoding="utf-8-sig")
        resume = extract_resume_text(args.resume, ocr=args.ocr) if args.resume else None
        result = agent.assess_resume(job, resume) if resume else agent.summarize_job(job)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(1, f"Analysis failed: {error}\n")
    print(result.model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
