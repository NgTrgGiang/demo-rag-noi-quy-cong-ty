"""Test cấu hình provider/model (Settings) trong config.py."""

import pytest

import config


def test_models_co_du_3_provider():
    """MODELS + MODEL_CHOICES phải khai báo đủ 3 provider, mỗi cái có embedding + chat."""
    for provider in ("openai", "gemini", "ollama"):
        assert provider in config.MODELS
        assert set(config.MODEL_CHOICES[provider]) == {"chat", "embedding"}


def test_settings_mac_dinh_dung_model_cua_provider():
    """Settings mặc định (không override) -> lấy đúng model mặc định của provider."""
    s = config.Settings(provider="openai")
    assert s.chat_model_name() == "gpt-4o-mini"
    assert s.emb_model() == "text-embedding-3-small"


def test_settings_cho_phep_override_model():
    """Người dùng chọn model khác -> Settings phải dùng model đó."""
    s = config.Settings(
        provider="openai", chat_model="gpt-4o", embedding_model="text-embedding-3-large"
    )
    assert s.chat_model_name() == "gpt-4o"
    assert s.emb_model() == "text-embedding-3-large"


def test_settings_provider_khong_hop_le_thi_bao_loi():
    """Provider lạ phải raise ValueError (tránh chạy nhầm cấu hình sai)."""
    with pytest.raises(ValueError):
        config.Settings(provider="provider-khong-ton-tai").chat_model_name()
