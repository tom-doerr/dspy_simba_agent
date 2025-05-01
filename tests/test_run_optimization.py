import coding_agent as ca
import coding_agent.optimization as ca_opt

class DummyPrediction:
    def __init__(self, completion):
        self.completion = completion

class DummyCoder:
    def __init__(self):
        self.forward_called = False

    def __call__(self, prompt):
        self.forward_called = True
        return DummyPrediction("XYZ")

class DummyExample:
    def __init__(self):
        self.problem = {'prompt': 'PP', 'task_id': 'T' }

def test_run_pre_optimization(capsys, monkeypatch):
    # Mock human_eval_metric within the optimization module namespace
    monkeypatch.setattr(ca_opt, "human_eval_metric", lambda gold, pred, trace=None, timeout=10: 0.3)
    coder = DummyCoder()
    example = DummyExample()

    # Pass timeout
    result = ca.run_pre_optimization(coder, example, timeout=10)
    out = capsys.readouterr().out
    assert "```python" in out
    # printed prompt + completion
    assert "PPXYZ" in out
    # result is the prediction
    assert hasattr(result, "completion") and result.completion == "XYZ"
    assert coder.forward_called

def test_run_post_optimization(capsys, monkeypatch):
    # Mock human_eval_metric within the optimization module namespace
    monkeypatch.setattr(ca_opt, "human_eval_metric", lambda gold, pred, trace=None, timeout=10: 0.7)
    coder = DummyCoder()
    example = DummyExample()

    # should not return anything, only print
    # Pass timeout
    ret = ca.run_post_optimization(coder, example, timeout=10)
    out = capsys.readouterr().out
    assert ret is None
    assert "```python" in out
    assert "PPXYZ" in out
