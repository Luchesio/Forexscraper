"""
Typed schemas for the structured sections of the analysis.

These mirror the JSON that Gemini is now asked to emit for the five
machine-readable sections. Validating here means the frontend never has to
guess at the model's intent with regex — it receives clean, typed data.

The narrative sections (MACRO_NARRATIVE, NEWS_IMPACT) remain free-text prose
and are passed through as plain strings.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator, model_validator

Direction               = Literal["bullish", "bearish", "neutral"]
BiasStrength            = Literal["strong", "moderate", "weak"]
StrengthLevel           = Literal["very-strong", "strong", "neutral", "weak", "very-weak"]
Stance                  = Literal["hawkish", "dovish", "neutral"]
FcAlignment             = Literal["STRONG", "MODERATE", "MIXED"]
AlignmentStatus         = Literal["CONFIRMED", "CONFLICTED", "NEUTRAL"]
RiskMode                = Literal["ON", "OFF", "NEUTRAL"]
MarketStructure         = Literal["CLEAN", "CHOPPY"]
ReactionQuality         = Literal["STRONG", "MODERATE", "WEAK"]
ConfirmationReliability = Literal["HIGH", "MODERATE", "LOW"]


class SessionFlow(BaseModel):
    asia:    str = ""
    london:  str = ""
    newYork: str = ""


class MarketContext(BaseModel):
    """Global trade environment + risk sentiment (was TRADE_ENVIRONMENT)."""
    marketStructure:         MarketStructure
    reactionQuality:         ReactionQuality
    confirmationReliability: ConfirmationReliability
    riskMode:                RiskMode = "NEUTRAL"
    riskDrivers:             str = ""


# Maps the 1-8 relative ranking onto the five strength labels, symmetrically:
#   1 → very-strong | 2-3 → strong | 4-5 → neutral | 6-7 → weak | 8 → very-weak
_STRENGTH_BY_RANK: dict[int, StrengthLevel] = {
    1: "very-strong",
    2: "strong",
    3: "strong",
    4: "neutral",
    5: "neutral",
    6: "weak",
    7: "weak",
    8: "very-weak",
}


class CurrencyScore(BaseModel):
    """One entry per currency (was FUNDAMENTAL_CONFIDENCE, now also carries
    the strength/stance/driver/outlook the frontend used to keyword-guess)."""
    currency:       str
    score:          float = Field(ge=0.0, le=10.0)
    alignment:      FcAlignment
    interpretation: str = ""
    strength:       StrengthLevel
    rank:           Optional[int] = None   # 1 = strongest … 8 = weakest (relative ranking)
    stance:         Stance = "neutral"
    driver:         str = ""
    outlook:        str = ""

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("rank")
    @classmethod
    def _rank_range(cls, v):
        # Keep a valid 1-8 rank; null out anything else rather than dropping the currency.
        if v is None:
            return None
        return v if 1 <= v <= 8 else None

    @model_validator(mode="after")
    def _strength_from_rank(self):
        # rank is the single source of truth for the label: when present, strength
        # is derived from it so the two can never contradict. The model's own
        # strength is used only as a fallback when rank is missing/invalid.
        if self.rank is not None:
            self.strength = _STRENGTH_BY_RANK[self.rank]
        return self


class HtfAlignmentItem(BaseModel):
    symbol:    str
    daily:     Direction
    intraday:  Direction
    alignment: AlignmentStatus

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


class InstrumentBias(BaseModel):
    symbol:       str
    direction:    Direction
    strength:     BiasStrength
    drivers:      str = ""
    invalidation: str = ""

    @field_validator("symbol")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()


ImpactLevel = Literal["HIGH", "MEDIUM", "LOW"]


class EconomicEvent(BaseModel):
    """A scheduled/just-released economic event the model found in the news.

    `hoursUntil` is the model's best estimate of hours until release (null if
    unknown or already released). The frontend derives the "< 2h → alert"
    urgency flag from it — that display rule stays deterministic in the UI
    rather than being guessed at."""
    currency:   str
    name:       str
    timing:     str = ""
    impact:     ImpactLevel = "MEDIUM"
    hoursUntil: Optional[float] = None

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("impact", mode="before")
    @classmethod
    def _impact_upper(cls, v):
        return v.strip().upper() if isinstance(v, str) else v