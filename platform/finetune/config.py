"""Fine-tune constants for the rights tier classifier."""

MODEL_ID = "google/flan-t5-base"
LABELS = ("cleared", "pending", "denied", "unknown")
TASK_PREFIX = "classify rights tier:"
MAX_INPUT_LENGTH = 256
MAX_LABEL_LENGTH = 8

# LoRA defaults (flan-t5 seq2seq)
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ("q", "v")
