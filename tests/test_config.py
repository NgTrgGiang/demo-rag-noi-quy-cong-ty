"""Test cấu hình provider trong config.py."""

import pytest

import config


def test_models_co_du_3_provider():
    """MODELS phải khai báo đủ 3 provider, mỗi cái có embedding + chat."""
    for provider in ("openai", "gemini", "ollama"):
        assert provider in config.MODELS
        assert "embedding" in config.MODELS[provider]
        assert "chat" in config.MODELS[provider]


def test_lay_dung_ten_model_openai(monkeypatch):
    """Với provider openai, _model() trả về đúng tên model mặc định."""
    monkeypatch.setattr(config, "PROVIDER", "openai")
    assert config._model("chat") == "gpt-4o-mini"
    assert config._model("embedding") == "text-embedding-3-small"


def test_provider_khong_hop_le_thi_bao_loi(monkeypatch):
    """Provider lạ phải raise ValueError (tránh chạy nhầm cấu hình sai)."""
    monkeypatch.setattr(config, "PROVIDER", "provider-khong-ton-tai")
    with pytest.raises(ValueError):
        config._model("chat")
