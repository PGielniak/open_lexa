import base64
import errno
import io
import json
import wave

import pytest
from fastapi.testclient import TestClient

from open_lexa_admin import db
import time

from open_lexa_admin.app import Worker, create_app, parse_range
from open_lexa.store import DirStore, SmbStore
from open_lexa.audio import wav_bytes
from open_lexa.stt import TranscriptResult
from open_lexa.text import compact, normalize, phonetic

AUTH = {'Authorization': 'Basic ' + base64.b64encode(b'admin:correct-horse-battery').decode()}


def put_recording(root, stem, text, day='2026-11-02', wav=True, **meta):
    folder = root / day
    folder.mkdir(parents=True, exist_ok=True)
    if wav:
        (folder / f'{stem}.wav').write_bytes(wav_bytes(bytes(3200)))
    (folder / f'{stem}.json').write_text(json.dumps({
        'created': f'{day}T10:00:00.000+00:00', 'session': stem[-8:], 'duration_seconds': 0.1,
        'text': text, 'language': 'pl', 'backend': 'hailo', 'command': None,
        'reply': 'sorry ale nie rozumiem', **meta}, ensure_ascii=False))


@pytest.fixture
def conn():
    connection = db.open_db(':memory:')
    yield connection
    connection.close()


def test_phonetic_normalisation_merges_sound_alike_spellings():
    assert normalize('  Włącz, ŚWIATŁO!  ') == 'włącz światło'
    assert phonetic('Włącz muzykę ze Spotify') == 'wlacz muzyke ze spotify'
    assert phonetic('Rzeka, chór, żółw') == phonetic('żeka hur zułw')
    assert compact('ze spotify') == compact('zespotify')


def test_migration_and_seed_are_idempotent(conn):
    db.migrate(conn)
    db.seed(conn)
    conn.execute("UPDATE intents SET description = 'edited' WHERE name = 'stop'")
    db.seed(conn)
    assert conn.execute('PRAGMA user_version').fetchone()[0] == len(db.MIGRATIONS)
    assert conn.execute("SELECT description FROM intents WHERE name = 'stop'").fetchone()[0] == 'edited'
    assert conn.execute("SELECT count(*) FROM intents WHERE name = 'play_spotify'").fetchone()[0] == 1
    tree = {i['name']: (i['parent'], i['depth']) for i in db.intents(conn)}
    assert tree['play_spotify'] == ('play_music', 1)
    assert db.intent_entity_map(conn)['play_spotify'] == ['search_query']
    # Names used by the current receiver rules stay meaningful as predicted_intent.
    assert {'lights_on', 'lights_off', 'play_radio', 'volume_up', 'volume_down'} <= set(tree)


def test_ingest_adds_complete_pairs_once(conn, tmp_path):
    put_recording(tmp_path, '1-0000000a', 'Włącz światło.', command='lights_on')
    put_recording(tmp_path, '2-0000000b', 'bez wav', wav=False)
    put_recording(tmp_path, '3-0000000c', None, error='stt down')
    (tmp_path / '_lexicon').mkdir()
    (tmp_path / '_lexicon' / 'x.json').write_text('{}')
    store = DirStore(tmp_path)

    assert db.ingest(conn, store) == 2
    assert db.ingest(conn, store) == 0
    assert db.ingest(conn, store, full=False) == 0
    rows = {r['stem']: r for r in db.list_recordings(conn)}
    assert set(rows) == {'1-0000000a', '3-0000000c'}
    assert rows['1-0000000a']['audio_path'] == '2026-11-02/1-0000000a.wav'
    assert rows['1-0000000a']['predicted_intent'] == 'lights_on'
    assert rows['1-0000000a']['corrected_transcript'] == 'Włącz światło.'
    assert rows['3-0000000c']['stt_error'] == 'stt down'
    assert db.counts(conn) == {'new': 2, 'labeled': 0, 'skipped': 0}

    put_recording(tmp_path, '4-0000000d', 'nowe', day='2026-11-03')
    assert db.ingest(conn, store, full=False) == 1


def test_ingest_uses_gpt_as_corrected_and_keeps_both_transcripts(conn, tmp_path):
    put_recording(
        tmp_path, '1-0000000a', 'włącz światło', backend='dual', model='gpt-transcribe',
        transcripts={
            'local': {'backend': 'http', 'model': 'whisper-1', 'text': 'włącz światło',
                      'latency_ms': 40},
            'cloud': {'backend': 'openai', 'model': 'gpt-transcribe', 'text': 'Włącz światło.',
                      'latency_ms': 900},
        },
        corrected_text='Włącz światło.', corrected_source='cloud')
    db.ingest(conn, DirStore(tmp_path))
    row = db.list_recordings(conn)[0]

    assert row['original_transcript'] == 'włącz światło'
    assert row['corrected_transcript'] == 'Włącz światło.'
    assert row['corrected_source'] == 'cloud'
    assert row['stt_backend'] == 'dual' and row['stt_model'] == 'gpt-transcribe'
    transcripts = db.transcript_map(row['transcripts'])
    assert transcripts['local']['model'] == 'whisper-1'
    assert transcripts['cloud']['text'] == 'Włącz światło.'


def test_export_keeps_both_transcripts_in_the_dataset(conn, tmp_path):
    put_recording(
        tmp_path, '1-0000000a', 'włącz światło', backend='dual',
        transcripts={
            'local': {'backend': 'http', 'model': 'whisper-1', 'text': 'włącz światło'},
            'cloud': {'backend': 'openai', 'model': 'gpt-transcribe', 'text': 'Włącz światło.'},
        },
        corrected_text='Włącz światło.', corrected_source='cloud')
    db.ingest(conn, DirStore(tmp_path))
    rid = db.list_recordings(conn)[0]['id']
    db.save_label(conn, rid, corrected_transcript='Włącz światło.', intent='lights_on', entities=[])
    _, dataset = db.export(conn)

    assert dataset[0]['local_transcript'] == 'włącz światło'
    assert dataset[0]['cloud_transcript'] == 'Włącz światło.'
    assert dataset[0]['corrected_source'] == 'cloud'


def test_detail_shows_both_transcripts_and_prefills_the_gpt_one(tmp_path):
    archive = tmp_path / 'archive'
    put_recording(
        archive, '1-0000000a', 'włącz światło', backend='dual',
        transcripts={
            'local': {'backend': 'http', 'model': 'whisper-1', 'text': 'włącz światło'},
            'cloud': {'backend': 'openai', 'model': 'gpt-transcribe', 'text': 'Włącz światło.'},
        },
        corrected_text='Włącz światło.', corrected_source='cloud')
    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=DirStore(archive),
                     username='admin', password='correct-horse-battery', background=False)
    with TestClient(app) as test_client:
        app.state.worker.ingest(full=True)
        page = test_client.get('/recordings/1', headers=AUTH).text
    assert 'Lokalny Whisper' in page and 'GPT transcribe' in page
    assert 'włącz światło' in page and 'Włącz światło.' in page
    assert 'auto „poprawna”' in page
    assert '<textarea id="corrected" name="corrected_transcript">Włącz światło.</textarea>' in page


def test_save_label_spans_learned_synonyms_and_warnings(conn, tmp_path):
    put_recording(tmp_path, '1-0000000a', 'Plej dram end bejs on spotyfaj')
    db.ingest(conn, DirStore(tmp_path))
    rid = db.list_recordings(conn)[0]['id']

    assert db.suggest_fragment('Plej dram end bejs on spotyfaj', 'drum and bass') == 'dram end bejs'
    warnings = db.save_label(conn, rid, corrected_transcript='Play drum and bass on spotify', intent='play_spotify',
                             entities=[{'type': 'search_query', 'value': 'drum and bass',
                                        'original_fragment': 'dram end bejs'},
                                       {'type': 'search_query', 'value': 'Pendulum', 'original_fragment': ''}])
    assert warnings and 'Pendulum' in warnings[0]
    entities = db.recording_entities(conn, rid)
    assert (entities[0]['value'], entities[0]['span_start'], entities[0]['span_end']) == ('drum and bass', 5, 18)
    assert entities[1]['span_start'] is None
    synonyms = {v['value']: [(s['synonym'], s['source']) for s in v['synonyms']] for v in db.dictionary(conn)}
    assert synonyms['drum and bass'] == [('dram end bejs', 'learned')]
    assert synonyms['Pendulum'] == []

    # Correcting the fragment replaces the learned synonym instead of piling up wrong ones.
    db.save_label(conn, rid, corrected_transcript='Play drum and bass on spotify', intent='play_spotify',
                  entities=[{'type': 'search_query', 'value': 'Drum and Bass', 'original_fragment': 'dram end bejs on'}])
    synonyms = {v['value']: [s['synonym'] for s in v['synonyms']] for v in db.dictionary(conn)}
    assert synonyms['drum and bass'] == ['dram end bejs on']
    assert db.get_recording(conn, rid)['status'] == 'labeled'

    with pytest.raises(ValueError):
        db.save_label(conn, rid, corrected_transcript='x', intent=None, entities=[])


def test_export_contains_examples_dictionary_and_dataset(conn, tmp_path):
    put_recording(tmp_path, '1-0000000a', 'Wąc muzyk, rozpotify')
    put_recording(tmp_path, '2-0000000b', 'To')
    db.ingest(conn, DirStore(tmp_path))
    ids = {r['stem']: r['id'] for r in db.list_recordings(conn)}
    db.save_label(conn, ids['1-0000000a'], corrected_transcript='Włącz muzykę ze Spotify', intent='play_spotify',
                  entities=[])
    db.save_label(conn, ids['2-0000000b'], corrected_transcript='', intent='noise', entities=[])

    lexicon, dataset = db.export(conn)
    examples = {e['text']: e for e in lexicon['examples']}
    assert examples['Wąc muzyk, rozpotify']['intent'] == 'play_spotify'
    assert examples['Wąc muzyk, rozpotify']['phonetic'] == 'wac muzyk rozpotify'
    intents = {i['name']: i for i in lexicon['intents']}
    assert intents['play_spotify']['parent'] == 'play_music'
    assert intents['noise']['actionable'] is False
    assert 'puść {search_query} na spotify' in intents['play_spotify']['phrases']
    assert 'erem ef' in lexicon['entities']['station']['values']['RMF24']
    assert [row['intent'] for row in dataset] == ['play_spotify', 'noise']
    assert dataset[0]['parent_intent'] == 'play_music'
    assert dataset[0]['audio_path'] == '2026-11-02/1-0000000a.wav'


@pytest.fixture
def client(tmp_path):
    archive = tmp_path / 'archive'
    put_recording(archive, '1-0000000a', 'Plej dram end bejs on spotyfaj')
    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=DirStore(archive), username='admin',
                     password='correct-horse-battery', background=False)
    with TestClient(app) as test_client:
        app.state.worker.ingest(full=True)
        yield test_client, app, archive


def test_auth_required_except_healthz(client):
    test_client, _, _ = client
    assert test_client.get('/healthz').status_code == 200
    assert test_client.get('/recordings').status_code == 401
    wrong = {'Authorization': 'Basic ' + base64.b64encode(b'admin:nope').decode()}
    assert test_client.get('/recordings', headers=wrong).status_code == 401
    page = test_client.get('/recordings?status=new', headers=AUTH)
    assert page.status_code == 200 and 'Plej dram end bejs' in page.text


def test_audio_supports_range_requests(client):
    test_client, _, _ = client
    full = test_client.get('/recordings/1/audio', headers=AUTH)
    assert full.status_code == 200 and full.content[:4] == b'RIFF'
    assert full.headers['accept-ranges'] == 'bytes'
    part = test_client.get('/recordings/1/audio', headers={**AUTH, 'Range': 'bytes=0-1'})
    assert part.status_code == 206 and part.content == b'RI'
    assert part.headers['content-range'] == f'bytes 0-1/{len(full.content)}'
    assert test_client.get('/recordings/99/audio', headers=AUTH).status_code == 404
    assert parse_range('bytes=-4', 10) == (6, 9)
    assert parse_range(None, 10) is None


def test_labeling_flow_exports_to_archive(client):
    test_client, app, archive = client
    detail = test_client.get('/recordings/1', headers=AUTH)
    assert detail.status_code == 200 and 'play_spotify' in detail.text
    assert test_client.get('/recordings/1/suggest?value=drum and bass', headers=AUTH).json() == {
        'fragment': 'dram end bejs'}
    response = test_client.post('/recordings/1', headers=AUTH, follow_redirects=False, data={
        'corrected_transcript': 'Play drum and bass on spotify', 'intent': 'play_spotify', 'action': 'save',
        'entity_type': ['search_query'], 'entity_value': ['drum and bass'], 'entity_fragment': ['dram end bejs']})
    assert response.status_code == 303 and 'saved=1' in response.headers['location']
    app.state.worker.export()
    lexicon = json.loads((archive / '_lexicon' / 'lexicon.json').read_text())
    assert lexicon['examples'][0]['entities'][0]['value'] == 'drum and bass'
    rows = (archive / '_dataset' / 'labels.jsonl').read_text().splitlines()
    assert json.loads(rows[0])['corrected_transcript'] == 'Play drum and bass on spotify'
    assert 'dram end bejs' in test_client.get('/dictionary', headers=AUTH).text


def test_dictionary_edits(client):
    test_client, _, _ = client
    post = lambda **data: test_client.post('/dictionary', headers=AUTH, data=data, follow_redirects=False)
    assert post(action='add_value', entity_type='search_query', value='Pendulum', synonyms='pendalam, pendulam'
                ).status_code == 303
    assert post(action='add_intent', name='play podcast', parent='play_music', description='Podcast',
                entity_types='search_query').status_code == 303
    assert post(action='add_phrase', intent='play_podcast', phrase='puść podcast {search_query}').status_code == 303
    page = test_client.get('/dictionary', headers=AUTH).text
    assert 'pendalam' in page and 'play_podcast' in page and 'puść podcast {search_query}' in page
    assert post(action='bogus').status_code == 400
    assert post(action='add_intent', name='stop').status_code == 409


class FakeSmb:
    def __init__(self):
        self.files, self.calls = {}, []

    def register_session(self, server, **kwargs):
        self.calls.append(('register', server))

    def _check(self, kwargs):
        assert kwargs['port'] == 4455 and 'connection_cache' in kwargs

    def listdir(self, path, **kwargs):
        self._check(kwargs)
        if path.endswith('missing'):
            raise OSError(errno.ENOENT, 'not found')
        return ['b', 'a']

    def makedirs(self, path, exist_ok=False, **kwargs):
        self._check(kwargs)

    def open_file(self, path, mode='r', **kwargs):
        self._check(kwargs)
        files = self.files

        class Handle:
            def __enter__(self): return self
            def __exit__(self, *exc): return False
            def write(self, data): files[path] = data
            def read(self): return files[path]
        return Handle()

    def rename_over(self, src, dst_in_share, **kwargs):
        self._check(kwargs)
        self.files['\\\\nas\\open-lexa\\' + dst_in_share] = self.files.pop(src)

    def reset_connection_cache(self, **kwargs):
        pass


def test_smb_store_paths_atomic_write_and_missing_dirs():
    client = FakeSmb()
    store = SmbStore('nas', 'open-lexa', '', port=4455, client=client)
    store.write_atomic('_lexicon/lexicon.json', b'{}')
    store.write_atomic('_lexicon/lexicon.json', b'{"v": 2}')
    assert client.files == {r'\\nas\open-lexa\_lexicon\lexicon.json': b'{"v": 2}'}
    assert store.read('_lexicon/lexicon.json') == b'{"v": 2}'
    assert store.list_dir('') == ['a', 'b']
    assert store.list_dir('missing') == []
    with pytest.raises(ValueError):
        store.read('../networkstorage/secret')


def test_failed_export_is_retried(tmp_path):
    class FlakyStore(DirStore):
        failures = 2

        def write_atomic(self, rel, data):
            if self.failures:
                self.failures -= 1
                raise OSError('STATUS_ACCESS_DENIED')
            super().write_atomic(rel, data)

    db.open_db(tmp_path / 'labels.sqlite3').close()
    store = FlakyStore(tmp_path / 'archive')
    worker = Worker(str(tmp_path / 'labels.sqlite3'), store, interval=0.1, debounce=0.01)
    worker.start()
    try:
        deadline = time.monotonic() + 5
        while worker.last_export is None and time.monotonic() < deadline:
            time.sleep(0.02)
    finally:
        worker.stop()
    assert worker.last_export is not None
    assert worker.last_error is None
    assert (tmp_path / 'archive' / '_lexicon' / 'lexicon.json').exists()


def test_settings_round_trip_keeps_the_api_key_out_of_html(client):
    test_client, app, archive = client
    page = test_client.get('/settings', headers=AUTH).text
    assert 'http://open-lexa-whisper:8080/v1' in page and 'nie zostały jeszcze zapisane' in page
    post = lambda **data: test_client.post('/settings', headers=AUTH, data=data, follow_redirects=False)
    response = post(mode='cloud', cloud_url='https://api.example.com/v1', cloud_model='whisper-1',
                    cloud_api_key='sk-secret-value-1234', language='pl')
    assert response.headers['location'] == '/settings?saved=1'
    page = test_client.get('/settings', headers=AUTH).text
    assert 'sk-secret-value-1234' not in page and '…1234' in page
    # An empty key keeps the stored one; the disabled local fieldset keeps its values.
    post(mode='local', local_url='http://whisper.lan:8080/v1', language='pl')
    stored = db.get_setting(db.connect(app.state.worker.db_path), 'stt')
    assert stored['mode'] == 'local' and stored['cloud_api_key'] == 'sk-secret-value-1234'
    assert stored['cloud_url'] == 'https://api.example.com/v1'
    assert 'error=' in post(mode='cloud', cloud_url='ftp://nope', language='pl').headers['location']
    assert 'error=' in post(mode='cloud', clear_api_key='1', language='pl').headers['location']
    post(mode='cloud', cloud_url='https://api.mistral.ai/v1/audio/transcriptions/', cloud_model='voxtral-mini-latest',
         local_url='http://whisper.lan:8080/v1', language='pl')
    app.state.worker.export()
    published = json.loads((archive / '_config' / 'stt.json').read_text())
    assert published['local_url'] == 'http://whisper.lan:8080/v1'
    assert published['cloud_url'] == 'https://api.mistral.ai/v1' and published['cloud_api_key'] == 'sk-secret-value-1234'


def test_settings_dual_round_trip_publishes_both_endpoints(client):
    test_client, app, archive = client
    page = test_client.get('/settings', headers=AUTH).text
    assert 'value="dual" checked' in page
    post = lambda **data: test_client.post('/settings', headers=AUTH, data=data, follow_redirects=False)
    response = post(mode='dual', local_url='http://whisper.lan:8080/v1',
                    local_model='whisper-small', cloud_url='https://api.openai.com/v1',
                    cloud_model='gpt-transcribe', cloud_api_key='sk-dual-test-1234', language='pl')
    assert response.headers['location'] == '/settings?saved=1'
    stored = db.get_setting(db.connect(app.state.worker.db_path), 'stt')
    assert stored['mode'] == 'dual' and stored['local_model'] == 'whisper-small'
    app.state.worker.export()
    published = json.loads((archive / '_config' / 'stt.json').read_text())
    assert published['mode'] == 'dual' and published['cloud_api_key'] == 'sk-dual-test-1234'
    # Dual needs both endpoints; a missing key is rejected like the cloud mode.
    error = post(mode='dual', local_url='http://whisper.lan:8080/v1',
                 cloud_url='https://api.openai.com/v1', clear_api_key='1', language='pl')
    assert 'error=' in error.headers['location']


class FakeTranscriber:
    name = 'fake'

    def __init__(self, text):
        self.text, self.calls, self.closed = text, [], False

    def transcribe(self, pcm, language):
        self.calls.append((len(pcm), language))
        if isinstance(self.text, Exception):
            raise self.text
        return self.text

    def close(self):
        self.closed = True


class FakeDualTranscriber:
    name = 'dual'

    def __init__(self, local='lokalna wersja', cloud='Włącz światło w kuchni.'):
        self.local, self.cloud = local, cloud
        self.calls, self.closed = [], False

    def transcribe_all(self, pcm, language):
        self.calls.append((len(pcm), language))
        return [
            TranscriptResult(self.local, 'http', 'whisper-1', role='local', latency_ms=30),
            TranscriptResult(self.cloud, 'openai', 'gpt-transcribe', role='cloud', latency_ms=700),
        ]

    def close(self):
        self.closed = True


@pytest.fixture
def recorder(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    built = []

    def factory(settings):
        built.append((settings, FakeTranscriber(factory.text)))
        return built[-1][1]

    factory.text = 'Włącz światło w kuchni'
    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=DirStore(archive), username='admin',
                     password='correct-horse-battery', background=False, transcriber_factory=factory)
    with TestClient(app) as test_client:
        yield test_client, factory, built, archive


def upload(test_client, data, origin='upload'):
    return test_client.post('/record', headers=AUTH, data={'origin': origin},
                            files={'audio': ('a.wav', data, 'audio/wav')})


def test_upload_in_dual_mode_prefills_gpt_transcription(tmp_path):
    archive = tmp_path / 'archive'
    archive.mkdir()
    built = []

    def factory(settings):
        built.append(FakeDualTranscriber())
        return built[-1]

    app = create_app(db_path=str(tmp_path / 'labels.sqlite3'), store=DirStore(archive),
                     username='admin', password='correct-horse-battery', background=False,
                     transcriber_factory=factory)
    with TestClient(app) as test_client:
        test_client.post('/settings', headers=AUTH, data={
            'mode': 'dual', 'local_url': 'http://whisper.lan:8080/v1',
            'local_model': 'whisper-1', 'cloud_url': 'https://api.openai.com/v1',
            'cloud_model': 'gpt-transcribe', 'cloud_api_key': 'sk-dual-1234', 'language': 'pl'})
        result = upload(test_client, wav_bytes(bytes(32000)), origin='browser').json()

    assert result['text'] == 'Włącz światło w kuchni.' and result['error'] is None
    assert len(built) == 1 and built[0].calls == [(32000, 'pl')] and built[0].closed
    meta = json.loads(sorted(archive.glob('*/*.json'))[-1].read_text())
    assert meta['text'] == 'Włącz światło w kuchni.'
    assert meta['corrected_text'] == 'Włącz światło w kuchni.' and meta['corrected_source'] == 'cloud'
    assert meta['transcripts']['local']['text'] == 'lokalna wersja'
    assert meta['transcripts']['cloud']['model'] == 'gpt-transcribe'


def test_upload_rejects_wrong_format(recorder):
    test_client, _, built, _ = recorder
    assert test_client.get('/record', headers=AUTH).status_code == 200
    stereo = io.BytesIO()
    with wave.open(stereo, 'wb') as out:
        out.setnchannels(2), out.setsampwidth(2), out.setframerate(44100), out.writeframes(bytes(400))
    response = upload(test_client, stereo.getvalue())
    assert response.status_code == 400 and '44100 Hz' in response.json()['detail']
    assert upload(test_client, b'not a wav').status_code == 400
    assert upload(test_client, wav_bytes(bytes(3200)), origin='evil').status_code == 400
    assert built == []


def test_upload_is_archived_transcribed_and_opened_for_labeling(recorder):
    test_client, factory, built, archive = recorder
    without_settings = upload(test_client, wav_bytes(bytes(3200)))
    assert without_settings.status_code == 200 and 'brak ustawień STT' in without_settings.json()['error']
    test_client.post('/settings', headers=AUTH, data={'mode': 'local', 'local_url': 'http://w:8080/v1',
                                                       'language': 'pl'})
    result = upload(test_client, wav_bytes(bytes(32000)), origin='browser').json()
    assert result['text'] == 'Włącz światło w kuchni' and result['error'] is None
    settings, transcriber = built[-1]
    assert settings['local_url'] == 'http://w:8080/v1' and transcriber.calls == [(32000, 'pl')]
    assert transcriber.closed
    detail = test_client.get(result['url'], headers=AUTH)
    assert detail.status_code == 200 and 'mikrofon w przeglądarce' in detail.text
    sidecars = sorted(archive.glob('*/*.json'))
    assert len(sidecars) == 2 and all(p.with_suffix('.wav').exists() for p in sidecars)
    meta = json.loads(sidecars[-1].read_text())
    assert meta['origin'] == 'browser' and meta['command'] == 'lights_on' and meta['duration_seconds'] == 1.0
    assert test_client.get(result['url'] + '/audio', headers=AUTH).content == wav_bytes(bytes(32000))

    factory.text = RuntimeError('whisper down')
    failed = upload(test_client, wav_bytes(bytes(3200))).json()
    assert failed['error'] == 'whisper down' and failed['text'] is None
    too_long = upload(test_client, wav_bytes(bytes(16000 * 2 * 13)))
    assert too_long.status_code == 400 and '12 s' in too_long.json()['detail']
