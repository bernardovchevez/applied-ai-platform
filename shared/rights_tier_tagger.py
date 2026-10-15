"""LoRA rights-tier tagger for ingest (adapter + data plane)."""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
FINETUNE_DIR = LAB_ROOT / "platform" / "finetune"
DEFAULT_ADAPTER = FINETUNE_DIR / "checkpoints" / "flan-t5-lora-rights-tier" / "adapter"

_tagger: "RightsTierTagger | None" = None


class RightsTierTagger:
    """Lazy-loaded flan-t5 LoRA classifier for snippet → tier labels."""

    def __init__(self, adapter_dir: Path | None = None) -> None:
        self.adapter_dir = adapter_dir or DEFAULT_ADAPTER
        self._model = None
        self._tokenizer = None
        self._device: str | None = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        if not self.adapter_dir.exists():
            raise FileNotFoundError(
                f"LoRA adapter not found: {self.adapter_dir}\n"
                "Run: python platform/finetune/train_classifier.py"
            )
        if str(FINETUNE_DIR) not in sys.path:
            sys.path.insert(0, str(FINETUNE_DIR))
        from inference import load_finetuned, predict_batch  # noqa: WPS433

        self._predict_batch = predict_batch
        self._model, self._tokenizer, self._device = load_finetuned(self.adapter_dir)

    def tag_snippets(self, snippets: list[str]) -> list[str]:
        self._ensure_loaded()
        labels, _elapsed = self._predict_batch(
            snippets, self._model, self._tokenizer, self._device
        )
        return labels

    def tag_snippet(self, snippet: str) -> str:
        return self.tag_snippets([snippet])[0]

    def document_tier(
        self,
        chunk_labels: list[str],
        *,
        warehouse_tier: str | None,
        doc_type: str | None,
    ) -> str | None:
        """Prefer warehouse tier for rights_clearance; else majority LoRA vote."""
        if warehouse_tier and doc_type == "rights_clearance":
            return warehouse_tier
        if not chunk_labels:
            return warehouse_tier
        return Counter(chunk_labels).most_common(1)[0][0]


def get_tagger(adapter_dir: Path | None = None) -> RightsTierTagger:
    global _tagger
    if _tagger is None or (adapter_dir and adapter_dir != _tagger.adapter_dir):
        _tagger = RightsTierTagger(adapter_dir)
    return _tagger
