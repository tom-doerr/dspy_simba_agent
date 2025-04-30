import dspy
import coding_agent as ca

class DummyLM:
    pass

def test_configure_lm(monkeypatch):
    # Capture calls
    calls = {}
    def fake_LM(name):
        calls['lm_name'] = name
        return DummyLM()
    def fake_configure(lm):
        calls['configured'] = isinstance(lm, DummyLM)

    monkeypatch.setattr(dspy, "LM", fake_LM)
    monkeypatch.setattr(dspy, "configure", fake_configure)

    lm = ca.configure_lm("my_lm")
    assert isinstance(lm, DummyLM)
    assert calls['lm_name'] == "my_lm"
    assert calls['configured'] is True
