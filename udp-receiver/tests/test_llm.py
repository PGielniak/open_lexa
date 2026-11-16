import pytest

from open_lexa.llm import LLMConfigError, create_llm, create_llm_from_file, validate_settings


def test_llm_configuration_and_legacy_fallback():
    assert create_llm({}) is None
    assert create_llm({'LLM_INTENT_MODE': 'off'}) is None
    settings = validate_settings({'mode': 'fallback', 'model': 'm', 'api_key': 'key'})
    assert settings['mode'] == 'fallback'
    assert create_llm_from_file(settings, client=object()).model == 'm'


@pytest.mark.parametrize('settings', [
    {'mode': 'invalid'}, {'mode': 'always', 'model': 'm'},
    {'mode': 'always', 'api_key': 'key'},
    {'mode': 'always', 'model': 'm', 'api_key': 'k', 'url': 'ftp://bad'},
])
def test_invalid_llm_settings(settings):
    with pytest.raises(LLMConfigError):
        validate_settings(settings)
