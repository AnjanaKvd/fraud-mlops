"""
schemas.py — Pydantic models (schemas) for the Fraud Detection API.

What are Pydantic schemas?
--------------------------
Pydantic schemas are Python classes that define the *shape* and *validation
rules* for data flowing in and out of the API.  FastAPI uses them to:

  • Automatically parse and validate JSON request bodies.
  • Serialize Python objects into JSON responses.
  • Drive the interactive Swagger / ReDoc docs generated at /docs and /redoc.

Two schemas are defined here:
  1. TransactionInput  – the JSON body a caller sends when requesting a
                         fraud prediction.
  2. PredictionOutput  – the JSON the API returns after scoring the
                         transaction.
"""

import uuid
from datetime import datetime, timezone

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# TransactionInput
# ---------------------------------------------------------------------------


class TransactionInput(BaseModel):
    """
    Represents a single credit-card transaction to be scored for fraud.

    Fields
    ------
    V1 … V28 : float
        Principal-component features produced by PCA on the original
        transaction data (exact semantics are confidential per the dataset
        provider).  All 28 components are required.
    Amount : float
        Transaction amount in the card's billing currency.
        Must be >= 0 — a negative charge is not a valid input.
    """

    # ------------------------------------------------------------------ #
    # PCA features  V1 – V28                                              #
    # Each field is required (no default → use Ellipsis `...`).          #
    # The `description` text appears as a tooltip in the Swagger UI.     #
    # ------------------------------------------------------------------ #
    V1: float = Field(..., description="PCA feature 1")
    V2: float = Field(..., description="PCA feature 2")
    V3: float = Field(..., description="PCA feature 3")
    V4: float = Field(..., description="PCA feature 4")
    V5: float = Field(..., description="PCA feature 5")
    V6: float = Field(..., description="PCA feature 6")
    V7: float = Field(..., description="PCA feature 7")
    V8: float = Field(..., description="PCA feature 8")
    V9: float = Field(..., description="PCA feature 9")
    V10: float = Field(..., description="PCA feature 10")
    V11: float = Field(..., description="PCA feature 11")
    V12: float = Field(..., description="PCA feature 12")
    V13: float = Field(..., description="PCA feature 13")
    V14: float = Field(..., description="PCA feature 14")
    V15: float = Field(..., description="PCA feature 15")
    V16: float = Field(..., description="PCA feature 16")
    V17: float = Field(..., description="PCA feature 17")
    V18: float = Field(..., description="PCA feature 18")
    V19: float = Field(..., description="PCA feature 19")
    V20: float = Field(..., description="PCA feature 20")
    V21: float = Field(..., description="PCA feature 21")
    V22: float = Field(..., description="PCA feature 22")
    V23: float = Field(..., description="PCA feature 23")
    V24: float = Field(..., description="PCA feature 24")
    V25: float = Field(..., description="PCA feature 25")
    V26: float = Field(..., description="PCA feature 26")
    V27: float = Field(..., description="PCA feature 27")
    V28: float = Field(..., description="PCA feature 28")

    # ------------------------------------------------------------------ #
    # Transaction amount                                                   #
    # `ge=0` adds a JSON-Schema constraint visible in the Swagger UI AND  #
    # raises a ValidationError before our custom validator even runs.     #
    # ------------------------------------------------------------------ #
    Amount: float = Field(
        ...,
        ge=0,
        description="Transaction amount in the card's billing currency (>= 0).",
    )

    # ------------------------------------------------------------------ #
    # Field validator — extra runtime guard on Amount                     #
    # ------------------------------------------------------------------ #
    @field_validator("Amount")
    @classmethod
    def amount_must_be_non_negative(cls, v: float) -> float:
        """Ensure Amount is >= 0 and return it unchanged.

        Although `ge=0` in Field() already rejects negative values at the
        schema level, an explicit validator:
          • Provides a human-readable error message.
          • Gives a single extensible location for future business-logic
            checks (e.g. maximum transaction ceiling, currency rounding).
        """
        if v < 0:
            raise ValueError(f"Amount must be a non-negative number, received {v}.")
        return v

    # ------------------------------------------------------------------ #
    # model_config — Swagger / OpenAPI example                            #
    # `json_schema_extra` surfaces under the "Example Value" tab in the  #
    # Swagger UI (/docs), giving API consumers a ready-to-use payload.   #
    # ------------------------------------------------------------------ #
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "summary": "Typical legitimate transaction",
                    "value": {
                        "V1": -1.3598071336738,
                        "V2": -0.0727811733098497,
                        "V3": 2.53634673796914,
                        "V4": 1.37815522427443,
                        "V5": -0.338320769942518,
                        "V6": 0.462387777762292,
                        "V7": 0.239598554061257,
                        "V8": 0.0986979012610507,
                        "V9": 0.363786969611213,
                        "V10": 0.0907941719789316,
                        "V11": -0.551599533260813,
                        "V12": -0.617800855762348,
                        "V13": -0.991389847235408,
                        "V14": -0.311169353699879,
                        "V15": 1.46817697209427,
                        "V16": -0.470400525259478,
                        "V17": 0.207971241929242,
                        "V18": 0.0257905801985591,
                        "V19": 0.403992960255733,
                        "V20": 0.251412098239705,
                        "V21": -0.018306777944153,
                        "V22": 0.277837575558899,
                        "V23": -0.110473910188767,
                        "V24": 0.0669280749146731,
                        "V25": 0.128539358273528,
                        "V26": -0.189114843888824,
                        "V27": 0.133558376740387,
                        "V28": -0.0210530534538215,
                        "Amount": 149.62,
                    },
                }
            ]
        }
    }


# ---------------------------------------------------------------------------
# PredictionOutput
# ---------------------------------------------------------------------------


class PredictionOutput(BaseModel):
    """
    The API's response after evaluating a transaction for fraud.

    FastAPI automatically serialises this model to JSON before sending the
    HTTP response, so all field types must be JSON-serialisable.

    Fields
    ------
    fraud_probability : float
        Model's estimated probability that the transaction is fraudulent,
        constrained to [0.0, 1.0].
    is_fraud : bool
        Hard binary classification derived from the probability threshold
        (e.g. True when fraud_probability >= 0.5).
    model_version : str
        Semantic-version string or MLflow run-ID of the model that produced
        this prediction — essential for reproducibility and audit trails.
    prediction_id : str
        A UUID-4 string that uniquely identifies this prediction request,
        enabling downstream tracing, logging, and de-duplication.
    timestamp : str
        ISO-8601 UTC timestamp recording when the prediction was generated.
    """

    fraud_probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability that the transaction is fraudulent, in [0, 1].",
    )
    is_fraud: bool = Field(
        ...,
        description="True if the transaction is classified as fraudulent.",
    )
    model_version: str = Field(
        ...,
        description="Version / MLflow run-ID of the model that scored this transaction.",
    )
    # `default_factory` is called once per response object so every prediction
    # gets its own unique ID without any caller input required.
    prediction_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique UUID-4 identifier for this prediction request.",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO-8601 UTC timestamp of when this prediction was generated.",
    )
