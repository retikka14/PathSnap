"""FastAPI tests with temporary trip files and a stub classifier (no dataset required)."""
import numpy as np
import pytest
from fastapi.testclient import TestClient
from src import api


@pytest.fixture
def client(tmp_path, monkeypatch):
    data = tmp_path / 'Student folder'
    folder = data / 'student_test' / '00000007'
    folder.mkdir(parents=True)
    (folder / '00000007.track').write_text('77.0000 12.0000 0\n77.0001 12.0000 1\n77.0002 12.0000 2\n')
    (folder / '00000007.nodes').write_text('77.0000 12.0000\n77.0001 12.0000\n77.0002 12.0000\n77.0003 12.0000\n')
    (folder / '00000007.arcs').write_text('0 1\n1 2\n2 3\n')
    artifact = tmp_path / 'model.joblib'
    artifact.write_bytes(b'stub')
    monkeypatch.setattr(api, 'DATA', data)
    monkeypatch.setattr(api, 'ARTIFACT', artifact)
    monkeypatch.setenv('PATHSNAP_EVAL_SPLIT', 'student_test')

    class StubModel:
        def predict_proba(self, features):
            n = len(features)
            return np.tile(np.array([[0.1, 0.9]]), (n, 1))

    monkeypatch.setattr(api, 'load', lambda _: {'model': StubModel(), 'k': 2})
    return TestClient(api.app)


def test_health(client):
    r = client.get('/health')
    assert r.status_code == 200
    assert r.json()['status'] == 'ok'
    assert r.json()['model_ready'] is True


def test_trips(client):
    r = client.get('/trips')
    assert r.status_code == 200
    assert r.json()['trips'] == ['00000007']


def test_rejects_invalid_trip_id(client):
    r = client.get('/predict/abc')
    assert r.status_code == 400


def test_missing_trip(client):
    r = client.get('/predict/99999999')
    assert r.status_code == 404


def test_predict_response_contract(client):
    r = client.get('/predict/00000007')
    assert r.status_code == 200, r.text
    body = r.json()
    n = body['observations']
    assert n == 3
    assert len(body['gps']) == n
    assert len(body['raw_arc_ids']) == n
    assert len(body['decoded_arc_ids']) == n
    assert len(body['predicted_segments']) == n
    assert all(len(segment) == 2 for segment in body['predicted_segments'])


def test_missing_model_returns_503(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api, 'ARTIFACT', tmp_path / 'missing.joblib')
    r = client.get('/predict/00000007')
    assert r.status_code == 503
