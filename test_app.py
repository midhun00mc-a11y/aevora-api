from pathlib import Path

import pytest
from fastapi import HTTPException
from pydantic import ValidationError
from google import genai

import app as api
from app import FitRequest, GeminiSuggestion, recommend_fit
import memory_store


@pytest.fixture(autouse=True)
def isolated_memory_db(request, monkeypatch):
    test_db_dir = Path(__file__).parent / ".pytest-state"
    test_db_dir.mkdir(exist_ok=True)
    db_path = test_db_dir / f"{request.node.name}.sqlite3"
    if db_path.exists():
        db_path.unlink()
    monkeypatch.setenv("AEVORA_DB_PATH", str(db_path))
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    yield
    if db_path.exists():
        db_path.unlink()


def payload(**overrides):
    data = {"product_id": "tee-001", "usual_size": "M", "fit_preference": "regular"}
    data.update(overrides)
    return FitRequest(**data)


def test_recommendation_uses_measurement_and_fit_preference():
    result = recommend_fit(payload(fit_preference="relaxed", chest_cm=94))
    assert result.recommended_size == "L"
    assert result.confidence == "medium"
    assert "ease" in result.explanation
    assert result.availability["XL"] is False
    assert result.fallback_used is True
    assert result.fallback_reason == "missing_api_key"


def test_recommendation_changes_with_usual_size_when_measurements_missing():
    small = recommend_fit(payload(usual_size="S"))
    large = recommend_fit(payload(usual_size="L"))
    assert small.recommended_size == "S"
    assert large.recommended_size == "L"
    assert small.confidence == "low"


def test_unknown_product_returns_404():
    with pytest.raises(HTTPException) as error:
        recommend_fit(payload(product_id="missing"))
    assert error.value.status_code == 404


def test_recommendation_endpoint_returns_consistent_json():
    schema = api.app.openapi()
    assert "/v1/fit/recommend" in schema["paths"]
    assert "post" in schema["paths"]["/v1/fit/recommend"]
    result = recommend_fit(payload())
    body = result.model_dump()
    assert body["recommended_size"] == "M"
    assert body["fallback_used"] is True
    assert body["customer_id"] == "demo-customer-001"
    assert body["memory"]["feedback_used"] is False


def test_invalid_preference_and_measurement_return_422():
    with pytest.raises(ValidationError):
        payload(fit_preference="baggy")
    with pytest.raises(ValidationError):
        payload(chest_cm=-10)


def test_size_with_no_stock_is_not_recommended():
    result = recommend_fit(payload(usual_size="XL", chest_cm=110))
    assert result.recommended_size == "L"
    assert result.availability["XL"] is False


def test_valid_gemini_size_is_used_and_catalog_fields_stay_backend_owned(monkeypatch):
    monkeypatch.setattr(api, "_gemini_suggestion", lambda *args: GeminiSuggestion(
        recommended_size="M", style_note="You prefer a classic everyday feel."
    ))
    result = recommend_fit(payload(chest_cm=94))
    assert result.recommended_size == "M"
    assert result.recommendation_source == "gemini"
    assert result.fallback_used is False
    assert result.availability["XL"] is False
    assert result.garment_measurements["M"].chest_cm == 102
    assert "classic everyday feel" in result.explanation


def test_unavailable_gemini_size_uses_deterministic_fallback(monkeypatch):
    monkeypatch.setattr(api, "_gemini_suggestion", lambda *args: GeminiSuggestion(
        recommended_size="XL", style_note="A relaxed style."
    ))
    result = recommend_fit(payload(fit_preference="relaxed", chest_cm=94))
    assert result.recommended_size == "L"
    assert result.recommendation_source == "rules"
    assert result.fallback_used is True
    assert result.fallback_reason == "unavailable_model_size"


def test_gemini_failure_uses_fallback_without_exposing_error(monkeypatch):
    def fail(*args):
        raise RuntimeError("network down and secret should not leak")
    monkeypatch.setattr(api, "_gemini_suggestion", fail)
    result = recommend_fit(payload())
    assert result.fallback_used is True
    assert result.fallback_reason == "gemini_error"
    assert "secret" not in result.model_dump_json()


def test_saved_preferences_are_retrieved_for_later_recommendation():
    memory_store.save_preferences("demo-1", "L", "relaxed")
    result = recommend_fit(FitRequest(product_id="tee-001", customer_id="demo-1"))
    assert result.recommended_size == "L"
    assert result.memory.preferences_used is True
    assert result.memory.feedback_used is False


def test_customer_memory_is_isolated_and_missing_preferences_are_not_assumed():
    memory_store.save_preferences("customer-a", "S", "slim")
    with pytest.raises(HTTPException) as error:
        recommend_fit(FitRequest(product_id="tee-001", customer_id="customer-b"))
    assert error.value.status_code == 422


def test_feedback_is_used_only_when_it_exists_and_changes_next_recommendation():
    before = recommend_fit(payload(fit_preference="relaxed", chest_cm=84, usual_size="M"))
    assert before.recommended_size == "M"
    assert before.memory.feedback_used is False

    response = api.submit_fit_feedback("demo-customer-001", api.FitFeedbackRequest(
        product_id="tee-001", recommended_size="M", sentiment="too_tight"
    ))
    assert response["saved"] is True
    after = recommend_fit(payload(fit_preference="relaxed", chest_cm=84, usual_size="M"))
    assert after.recommended_size == "L"
    assert after.memory.feedback_count == 1
    assert after.memory.feedback_used is True
    assert after.memory.feedback_adjustment_cm == 4


def test_preferences_can_be_updated_and_deleted_through_api():
    saved = api.update_preferences("demo-2", api.CustomerPreferenceUpdate(
        usual_size="L", fit_preference="slim"
    ))
    assert saved.saved is True
    assert saved.usual_size == "L"

    deleted = api.remove_preferences("demo-2")
    assert deleted["deleted"] is True
    assert api.read_preferences("demo-2").saved is False


def test_official_gemini_sdk_receives_verified_context_and_structured_schema(monkeypatch):
    observed = {}

    class FakeInteractions:
        def create(self, **kwargs):
            observed.update(kwargs)
            return type("Interaction", (), {"output_text": GeminiSuggestion(
                recommended_size="M", style_note="You prefer a relaxed style."
            ).model_dump_json()})()

    class FakeClient:
        def __init__(self, api_key, http_options):
            observed["api_key"] = api_key
            observed["timeout"] = http_options.timeout
            self.interactions = FakeInteractions()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-a-real-secret")
    monkeypatch.setattr(genai, "Client", FakeClient)
    suggestion = api._gemini_suggestion(payload(fit_preference="relaxed", chest_cm=94, height_cm=177), api.PRODUCTS["tee-001"], "too_tight")
    assert suggestion.recommended_size == "M"
    assert observed["model"] == "gemini-3.8-flash"
    response_format = observed["response_format"]
    assert response_format[0]["mime_type"] == "application/json"
    assert response_format[0]["schema"] == GeminiSuggestion.model_json_schema()
    assert observed["timeout"] == 10_000
    assert "size_chart" in observed["input"]
    assert "fit_preference" in observed["input"]
    assert "too_tight" in observed["input"]
    assert "available_sizes" in observed["input"]
    assert "chest_cm\": 94" not in observed["input"]
    assert "height_cm\": 177" not in observed["input"]
