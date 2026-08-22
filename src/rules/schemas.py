# src/rules/schemas.py
from pydantic import BaseModel, Field, field_validator, model_validator

from src.config import settings

TRAIT_KEYS = ("shape", "apex", "base", "margin")

# The database stores the rule base, but config stays the vocabulary: a rule
# may only name a class one of the models can actually predict.
VOCAB: dict[str, list[str]] = {
    "shape": settings.SHAPE_CLASSES,
    "apex": settings.APEX_CLASSES,
    "base": settings.BASE_CLASSES,
    "margin": settings.MARGIN_CLASSES,
}


def _canonical(trait_key: str, value: str) -> str:
    """Map a submitted class to the exact casing config uses, or raise."""
    allowed = VOCAB[trait_key]
    match = next((c for c in allowed if c.lower() == value.strip().lower()), None)
    if match is None:
        raise ValueError(
            f"'{value}' is not a known {trait_key} class "
            f"(allowed: {', '.join(allowed)})"
        )
    return match


class RuleGroupIn(BaseModel):
    code: str = Field(min_length=1, max_length=20)
    name: str = Field(min_length=1, max_length=255)
    shape: str
    apex: str
    base: str
    margin: str
    is_active: bool = True

    @field_validator("code", "name")
    @classmethod
    def _strip(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def _check_vocab(self):
        for key in TRAIT_KEYS:
            setattr(self, key, _canonical(key, getattr(self, key)))
        return self

    def to_row(self) -> dict:
        return self.model_dump()


class RuleTestRequest(BaseModel):
    """One hand-entered prediction, for the admin page's test bench."""
    traits: dict[str, str] = Field(default_factory=dict)
    probs: dict[str, float] = Field(default_factory=dict)
    conf_th: float = Field(default=0.6, ge=0.0, le=1.0)
    top_n: int = Field(default=3, ge=1, le=20)
