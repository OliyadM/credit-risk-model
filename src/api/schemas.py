"""Request and response bodies for the scoring service."""

from pydantic import BaseModel, Field, field_validator


class PredictRequest(BaseModel):
    """A customer id and the feature row the artifact was trained on."""

    customer_id: str = Field(min_length=1, max_length=128)
    features: dict[str, float]

    @field_validator("customer_id")
    @classmethod
    def strip_customer_id(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("customer_id is empty")
        return cleaned


class PredictResponse(BaseModel):
    """Score for one customer. The probability is a rank score, not a PD."""

    customer_id: str
    risk_probability: float = Field(ge=0.0, le=1.0)
    is_high_risk: int = Field(ge=0, le=1)
    threshold: float
    model_name: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_name: str
    model_version: str
    n_features: int


class FeatureSchemaResponse(BaseModel):
    model_name: str
    model_version: str
    threshold: float
    feature_columns: list[str]
