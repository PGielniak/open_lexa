import json
import os
import queue
import threading
from types import SimpleNamespace

import pytest

from open_lexa.commands import Decision
from open_lexa.llm import (LLMConfigError, build_dictionary, create_llm,
                           create_llm_from_file, parse_answer, settings_from_env,
                           validate_settings)
from open_lexa.llm import SETTINGS_PATH as LLM_SETTINGS_PATH
from open_lexa.nlu import NLU, parse_lexicon
from open_lexa.protocol import Recording
from open_lexa.server import HealthState, worker
from open_lexa.store import DirStore
from open_lexa.watch import JsonWatcher

from test_commands_archive import MemoryArchive, run_worker
from test_server import FakeTranscriber

LEXICON_DATA = {
    'version': 3,
    'intents': [
        {'name': 'play_music', 'actionable': True, 'entities': ['search_query'],
         'phrases': ['włącz {search_query}', 'zagraj muzykę']},
        {'name': 'set_volume', 'actionable': True, 'entities': ['volume'],
         'phrases': ['ustaw głośność na {volume}']},
        {'name': 'noise', 'actionable': False, 'entities': [], 'phrases': []},
    ],
    'entities': {
        'volume': {'kind': 'number', 'values': {}},
        'search_query': {'kind': 'text', 'values': {}},
        'genre': {'kind': 'enum', 'values': {'rock': ['rocka'], 'jazz': []}},
    },
    'examples': [],
}

BASE_ENV = {'LLM_API_KEY': 'sk-test', 'LLM_MODEL': 'gpt-test'}


class FakeChatClient:
    """Minimal OpenAI-shaped client: chat.completions.create() only."""

    def __init__(self, content=None, error=None):
        self.error = error
        self.content = content
        self.kwargs = None
        self.closed = False
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.kwargs = kwargs
        if self.error is not None:
            raise self.error
        message = SimpleNamespace(content=self.content)
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    def close(self):
        self.closed = True


def answer(**fields):
    return json.dumps({'intent': None, 'entities': {}, 'confidence': 0.9, **fields},
                      ensure_ascii=False)


def make_llm(content=None, error=None):
    client = FakeChatClient(content, error)
    return create_llm(BASE_ENV, client=client), client


def lexicon():
    return parse_lexicon(LEXICON_DATA)


# --- configuration ----------------------------------------------------------------------

def test_llm_disabled_without_credentials_or_explicit_off():
    assert create_llm({}) is None
    assert create_llm({'LLM_INTENT_MODE': 'off', **BASE_ENV}) is None


def test_default_mode_is_always_once_configured():
    llm = create_llm(BASE_ENV, client=FakeChatClient(answer(intent='noise')))
    assert llm is not None and llm.mode == 'always'


@pytest.mark.parametrize('env, message', [
    ({'LLM_INTENT_MODE': 'sometimes'}, 'LLM_INTENT_MODE'),
    ({'LLM_INTENT_MODE': 'always', 'LLM_MODEL': 'm'}, 'LLM_API_KEY'),
    ({'LLM_INTENT_MODE': 'always', 'LLM_API_KEY': 'k'}, 'LLM_MODEL'),
    ({'LLM_URL': 'ftp://x', **BASE_ENV}, 'LLM_URL'),
    ({'LLM_TIMEOUT': 'abc', **BASE_ENV}, 'LLM_TIMEOUT'),
])
def test_invalid_llm_settings_raise(env, message):
    with pytest.raises(LLMConfigError, match=message):
        create_llm(env, client=FakeChatClient())


def test_settings_url_is_normalised():
    settings = validate_settings(settings_from_env(
        {'LLM_URL': 'https://llm.example/v1/chat/completions/', **BASE_ENV}))
    assert settings['url'] == 'https://llm.example/v1'


# --- dictionary ----------------------------------------------------------------------

def test_lexicon_keeps_original_phrases_for_the_prompt():
    parsed = lexicon()
    assert parsed.phrases['play_music'] == ('włącz {search_query}', 'zagraj muzykę')


def test_build_dictionary_uses_lexicon_and_adds_builtin_intents():
    dictionary = build_dictionary(lexicon())
    names = {intent['name'] for intent in dictionary['intents']}
    assert names == {'noise', 'not_for_me', 'play_music', 'set_volume'}
    play = next(i for i in dictionary['intents'] if i['name'] == 'play_music')
    assert play['phrases'] == ['włącz {search_query}', 'zagraj muzykę']
    assert play['entities'] == ['search_query']
    assert {'name': 'genre', 'kind': 'enum', 'values': ['jazz', 'rock']} in dictionary['entity_types']


def test_build_dictionary_without_lexicon_offers_rule_commands():
    dictionary = build_dictionary(None)
    names = {intent['name'] for intent in dictionary['intents']}
    assert {'lights_on', 'volume_up', 'noise', 'not_for_me'} <= names


# --- answer validation ------------------------------------------------------------------

def test_parse_answer_accepts_only_dictionary_labels():
    dictionary = build_dictionary(lexicon())
    decision = parse_answer(
        answer(intent='set_volume', entities={'volume': 'pięć', 'genre': 'rock',
                                              'bogus_type': 'x'}), dictionary)
    assert decision == Decision('set_volume', 'set_volume', intent='set_volume',
                                entities=(('volume', '5'),), score=90.0, source='llm')


@pytest.mark.parametrize('content', [
    answer(intent='make_coffee'),          # not in the dictionary
    answer(intent='unknown'),              # model gave up
    'to nie jest JSON',
    '[1, 2, 3]',
])
def test_parse_answer_rejects_invalid_or_unknown(content):
    assert parse_answer(content, build_dictionary(lexicon())) is None


def test_parse_answer_strips_fences_and_accepts_list_entities():
    dictionary = build_dictionary(lexicon())
    fenced = '```json\n' + answer(intent='play_music',
                                  entities=[{'type': 'search_query', 'value': 'Chill'}]) + '\n```'
    decision = parse_answer(fenced, dictionary)
    assert decision.entities == (('search_query', 'Chill'),)


def test_parse_answer_non_actionable_and_missing_confidence():
    dictionary = build_dictionary(lexicon())
    noise = parse_answer('{"intent": "noise"}', dictionary)
    assert noise is not None and noise.command is None and noise.intent == 'noise'
    assert noise.score is None


# --- client behaviour -------------------------------------------------------------------

def test_recognize_builds_json_prompt_and_returns_decision():
    llm, client = make_llm(answer(intent='play_music', entities={'search_query': 'Chill'}))
    decision = llm.recognize('włącz chill', lexicon())
    assert decision.intent == 'play_music' and decision.source == 'llm'
    assert client.kwargs['response_format'] == {'type': 'json_object'}
    assert client.kwargs['temperature'] == 0
    payload = json.loads(client.kwargs['messages'][1]['content'])
    assert payload['transcript'] == 'włącz chill'
    assert any(i['name'] == 'play_music' for i in payload['dictionary']['intents'])


def test_recognize_swallows_network_and_garbage_failures():
    llm, client = make_llm(error=RuntimeError('connection reset'))
    assert llm.recognize('cokolwiek', lexicon()) is None
    assert llm.stats['failures'] == 1 and llm.stats['calls'] == 1
    client.content, client.error = 'garbage', None
    assert llm.recognize('cokolwiek', lexicon()) is None
    assert llm.stats['failures'] == 2 and llm.stats['calls'] == 2
    llm.close()
    assert client.closed


def test_dictionary_is_cached_per_lexicon_version():
    llm, _ = make_llm()
    parsed = lexicon()
    first = llm.dictionary(parsed)
    assert llm.dictionary(parsed) is first
    assert llm.dictionary(None) is not first


# --- NLU cascade -------------------------------------------------------------------

class StubLLM:
    def __init__(self, decision):
        self.decision = decision
        self.texts = []

    def recognize(self, text, lexicon):
        self.texts.append(text)
        return self.decision

    def close(self):
        pass


MATCHED = Decision('play_music', 'play_music', intent='play_music',
                   entities=(('search_query', 'chill'),), score=95.0, source='llm')


def nlu_with(mode, decision=None):
    stub = StubLLM(decision)
    nlu = NLU(llm=stub, mode=mode)
    nlu.update(LEXICON_DATA)
    return nlu, stub


def test_nlu_off_never_calls_the_llm():
    nlu, stub = nlu_with('off', MATCHED)
    assert nlu.decide('jaka jest pogoda').command is None
    assert stub.texts == []


def test_always_mode_lets_llm_decide_every_transcript():
    nlu, stub = nlu_with('always', MATCHED)
    assert nlu.decide('ustaw głośność na 20').intent == 'play_music'  # overrides the lexicon
    assert nlu.decide('').intent is None                              # empty text: no call
    assert stub.texts == ['ustaw głośność na 20']


def test_always_mode_keeps_base_decision_when_llm_fails():
    nlu, stub = nlu_with('always', None)
    decision = nlu.decide('ustaw głośność na 20')
    assert decision.source == 'canonical' and stub.texts == ['ustaw głośność na 20']


def test_fallback_mode_skips_llm_when_lexicon_or_rules_match():
    nlu, stub = nlu_with('fallback', MATCHED)
    assert nlu.decide('ustaw głośność na 20').source == 'canonical'
    assert nlu.decide('puść radio').intent == 'play_radio'  # regex rules
    assert stub.texts == []


def test_fallback_mode_asks_llm_when_nothing_is_recognised():
    nlu, stub = nlu_with('fallback', MATCHED)
    assert nlu.decide('jaka jest pogoda').intent == 'play_music'
    assert stub.texts == ['jaka jest pogoda']


# --- worker pipeline -------------------------------------------------------------------

def test_worker_archives_llm_decision_with_entities():
    archive = MemoryArchive()
    nlu, stub = nlu_with('always', MATCHED)
    state = run_worker(FakeTranscriber(['Włącz chill']),
                       handle=nlu.decide, nlu=nlu, archive=archive)
    meta = archive.saved[0][2]
    assert state.archived == 1 and state.failures == 0
    assert meta['text'] == 'Włącz chill'
    assert meta['command'] == 'play_music' and meta['intent'] == 'play_music'
    assert meta['source'] == 'llm' and meta['score'] == 95.0
    assert meta['entities'] == {'search_query': 'chill'}


# --- published admin settings (_config/llm.json) -----------------------------------------

def test_create_llm_from_file_prefers_the_published_settings():
    llm = create_llm_from_file({'mode': 'fallback', 'url': 'https://llm.example/v1',
                                'model': 'published-model', 'api_key': 'sk-file'}, env={})
    assert llm is not None and llm.mode == 'fallback' and llm.model == 'published-model'
    assert str(llm.client.base_url).startswith('https://llm.example/v1')
    assert create_llm_from_file({'mode': 'off'}, env=BASE_ENV) is None
    with pytest.raises(LLMConfigError, match='LLM_API_KEY'):
        create_llm_from_file({'mode': 'always', 'model': 'm'}, env={})


def test_missing_file_falls_back_to_the_environment():
    llm = create_llm_from_file(None, env=BASE_ENV, client=FakeChatClient())
    assert llm is not None and llm.model == 'gpt-test'


def clear_llm_env(monkeypatch):
    for key in [name for name in os.environ if name.startswith('LLM_')]:
        monkeypatch.delenv(key)


def test_worker_switches_llm_from_the_published_file(tmp_path, monkeypatch, caplog):
    clear_llm_env(monkeypatch)
    store = DirStore(tmp_path)
    watcher = JsonWatcher(store, LLM_SETTINGS_PATH)
    nlu = NLU()
    transcriber = FakeTranscriber(['Włącz światło.'] * 4)
    state = HealthState(transcriber.name)
    jobs = queue.Queue()
    thread = threading.Thread(target=worker, args=(jobs, transcriber, 'pl', None, state),
                              kwargs={'handle': nlu.decide, 'nlu': nlu, 'llm_config': watcher},
                              daemon=True)
    thread.start()
    published = json.dumps({'mode': 'fallback', 'url': 'https://llm.example/v1',
                            'model': 'published-model', 'api_key': 'sk-file'}).encode()
    try:
        # No file yet and no environment key: the rules pipeline alone.
        jobs.put(Recording(('127.0.0.1', 1234), 1, bytes(640)))
        jobs.join()
        assert nlu.llm is None and state.llm == {}
        # An invalid file keeps the current (absent) client and only warns.
        store.write_atomic(LLM_SETTINGS_PATH, b'{"mode": "sometimes"}')
        watcher.poll()
        jobs.put(Recording(('127.0.0.1', 1234), 2, bytes(640)))
        jobs.join()
        assert nlu.llm is None and 'New LLM settings rejected' in caplog.text
        # The admin's file overrides the environment; the rules still answer first.
        store.write_atomic(LLM_SETTINGS_PATH, published)
        watcher.poll()
        jobs.put(Recording(('127.0.0.1', 1234), 3, bytes(640)))
        jobs.join()
        assert nlu.llm is not None and nlu.llm.model == 'published-model'
        assert nlu.mode == 'fallback' and nlu.decide('włącz światło').intent == 'lights_on'
        assert state.llm['model'] == 'published-model' and state.llm['calls'] == 0
        # Removing the file returns to the (here empty) environment.
        (tmp_path / LLM_SETTINGS_PATH).unlink()
        watcher.poll()
        jobs.put(Recording(('127.0.0.1', 1234), 4, bytes(640)))
        jobs.join()
        assert nlu.llm is None and nlu.mode == 'off' and state.llm == {}
    finally:
        jobs.put(None)
        thread.join(timeout=5)
    assert not thread.is_alive()
