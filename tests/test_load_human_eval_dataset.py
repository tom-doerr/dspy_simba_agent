import datasets
import pytest
import coding_agent as ca

def test_load_dataset_success(monkeypatch):
    fake = {'test': [{'prompt': 'p1'}, {'prompt': 'p2'}]}
    monkeypatch.setattr(datasets, "load_dataset", lambda name: fake)
    problems = ca.load_human_eval_dataset()
    assert isinstance(problems, list)
    assert problems == fake['test']

def test_load_dataset_missing_split(monkeypatch):
    fake = {'train': []}
    monkeypatch.setattr(datasets, "load_dataset", lambda name: fake)
    with pytest.raises(ValueError) as ei:
        ca.load_human_eval_dataset()
    assert "Expected 'test' split not found" in str(ei.value)

def test_load_dataset_exception_propagates(monkeypatch):
    def bad_load(name):
        raise RuntimeError("no go")
    monkeypatch.setattr(datasets, "load_dataset", bad_load)
    with pytest.raises(RuntimeError):
        ca.load_human_eval_dataset()
