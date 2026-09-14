from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)

def _case_id():
    r = client.post('/chat/message', json={'message': 'Synthetic safety test message.'})
    assert r.status_code == 200
    return r.json()['case_id']

def test_case_list_detail_confirm_and_audit():
    case_id = _case_id()
    assert any(c['case_id'] == case_id for c in client.get('/cases').json()['cases'])
    assert client.get(f'/case/{case_id}').status_code == 200
    action = client.post(f'/case/{case_id}/confirm', json={'operator_id': 'operator-test', 'action': 'CONFIRM'})
    assert action.status_code == 200
    detail = client.get(f'/case/{case_id}').json()
    assert detail['operator_confirmation_state'] == 'CONFIRM'
    assert len(detail['operator_actions']) == 1

def test_override_requires_reason_and_unknown_case_is_safe():
    case_id = _case_id()
    assert client.post(f'/case/{case_id}/confirm', json={'operator_id': 'operator-test', 'action': 'OVERRIDE'}).status_code == 400
    assert client.get('/case/CASE-NOPE').status_code == 404
    assert client.post('/case/CASE-NOPE/confirm', json={'operator_id':'x','action':'CONFIRM'}).status_code == 404
