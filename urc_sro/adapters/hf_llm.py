from __future__ import annotations
from typing import List, Tuple, Optional

from ..types import Document
from ..llm_interfaces import LLM

class HFGenerativeLLM(LLM):
    """
    Thin wrapper around Hugging Face Transformers (default: mistralai/Mistral-7B-Instruct-v0.3).
    Implements the LLM Protocol used by SRO.
    """

    def __init__(
        self,
        model_name: str = "mistralai/Mistral-7B-Instruct-v0.3",
        max_new_tokens: int = 256,
        temperature: float = 0.2,
    ) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Device map 'auto' = place on available GPU if present
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
        )

        # Ensure pad token id
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    # --- helpers ---
    def _format_context(self, context: List[Document], max_chars: int = 2000) -> str:
        buf, total = [], 0
        for d in context:
            chunk = f"[{d.id}] {d.text}"
            if total + len(chunk) > max_chars:
                break
            buf.append(chunk)
            total += len(chunk)
        return "\n".join(buf)

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        full_prompt = (system + "\n" if system else "") + prompt
        inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=self.temperature > 0,
            temperature=self.temperature,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        text = self.tokenizer.decode(out[0], skip_special_tokens=True)
        # Try to return only the model completion (strip the prompt prefix if present)
        return text[len(full_prompt):].strip() if text.startswith(full_prompt) else text.strip()

    def generate_answer_with_steps(self, query: str, context: List[Document]) -> Tuple[str, List[str]]:
        ctx = self._format_context(context)
        system = "You answer using ONLY the provided context. Cite document ids in brackets when relevant."
        prompt = f"Context:\n{ctx}\n\nQuestion: {query}\nWrite a concise, well-supported answer (3–5 sentences)."
        answer = self.generate(prompt, system=system)
        steps = [
            "Identify relevant passages from context",
            "Compose a concise answer grounded in cited passages",
            "Validate key claims against the context",
        ]
        return answer, steps

    def generate_next_step(self, query: str, context: List[Document], prior_steps: List[str]) -> str:
        return "Propose the next retrieval or verification step based on uncovered gaps."

    def self_refine_step(self, step: str, feedback: str) -> str:
        return f"{step} (refined: {feedback})"

    def self_refine_answer(self, answer: str, feedback: str) -> str:
        return f"{answer} [Refined: {feedback}]"


# ------------------------------
# Minimal dependency-free mock LLM for tests / mock pipeline
# ------------------------------
from typing import List, Tuple, Optional
from ..types import Document
from ..llm_interfaces import LLM

class TrivialLLM(LLM):
    """
    A tiny, no-deps LLM adapter for fast orchestration tests.
    Produces a short draft from context snippets and simple steps.
    """

    def generate(self, prompt: str, system: Optional[str] = None) -> str:
        pre = f"{system.strip()} " if system else ""
        return (pre + prompt.strip()).strip()

    def generate_answer_with_steps(self, query: str, context: List[Document]) -> Tuple[str, List[str]]:
        src_ids = [d.id for d in context[:3]]
        snippet = " ".join(_first_sentence(d.text) for d in context[:2]).strip()
        steps = [
            "Identify relevant passages from retrieved context.",
            "Synthesize a concise answer grounded in cited passages."
        ]
        answer = (
            f"Based on sources {src_ids}, here is a concise answer to your query: {query.strip()}. "
            f"{snippet or 'No detailed evidence snippets available.'}"
        ).strip()
        return answer, steps

    def generate_next_step(self, query: str, context: List[Document], prior_steps: List[str]) -> str:
        return "Validate that each statement is supported by at least one retrieved passage."

    def self_refine_step(self, step: str, feedback: str) -> str:
        return f"{step} (refined: {feedback})"

    def self_refine_answer(self, answer: str, feedback: str) -> str:
        return (answer + f" [Refined: {feedback}]").strip()

def _first_sentence(text: str) -> str:
    s = text.strip().split(".")
    return (s[0] + ".") if s and s[0] else ""
