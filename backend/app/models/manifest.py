from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ImbalanceStrategy(str, Enum):
    """Supported class imbalance handling strategies."""
    NONE = "none"
    CLASS_WEIGHT = "class_weight"
    SMOTE_ENN = "smote_enn"


class ThresholdsConfig(BaseModel):
    """Decision thresholds satisfying 0 < review < high < 1."""
    model_config = ConfigDict(extra="forbid")

    high: float = Field(..., gt=0.0, lt=1.0, description="High risk threshold")
    review: float = Field(..., gt=0.0, lt=1.0, description="Analyst review threshold")

    @model_validator(mode="after")
    def validate_thresholds(self) -> "ThresholdsConfig":
        if not (0.0 < self.review < self.high < 1.0):
            raise ValueError(
                f"Invalid thresholds: must satisfy 0 < review < high < 1. "
                f"Got review={self.review}, high={self.high}."
            )
        return self


class ManifestSchema(BaseModel):
    """Strict schema for manifest.json validating ML bundle contract."""
    model_config = ConfigDict(extra="forbid")

    model_version: str = Field(..., min_length=1, description="Model release version")
    imbalance_strategy: ImbalanceStrategy = Field(..., description="Imbalance handling strategy")
    feature_names: list[str] = Field(..., min_length=1, description="Ordered feature column names")
    thresholds: ThresholdsConfig = Field(..., description="Risk threshold definitions")
    shap_top_k: int = Field(..., ge=1, description="Number of top SHAP features to report")
    stability_min_overlap: float = Field(..., ge=0.0, le=1.0, description="Minimum overlap ratio for stability")
    n_replicates: int = Field(..., ge=2, description="Number of random seed replicate models (>= 2)")
