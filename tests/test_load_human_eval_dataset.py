import datasets
import pytest
from coding_agent.data import load_human_eval_dataset

def test_load_dataset_success(monkeypatch):
    fake = [{'prompt': 'p1'}, {'prompt': 'p2'}] # Simulate the structure returned for the 'test' split
    # Mock load_dataset to accept name and kwargs, return fake only for 'test' split
    monkeypatch.setattr(datasets, "load_dataset", lambda name, split='test', **kwargs: fake if split=='test' else [])
    problems = load_human_eval_dataset()
    assert isinstance(problems, list)
    assert problems == fake

def test_load_dataset_missing_split(monkeypatch):
    fake = {'train': []} # Simulate missing 'test' split
    # Update mock to accept 'name' and 'split' kwargs
    def mock_load(name, split='test', **kwargs):
        if name == "openai_humaneval" and split == 'test':
            # Simulate datasets raising KeyError if split doesn't exist in the loaded data
            raise KeyError(f"Split {split} not found in dataset {name}.")
        return fake.get(split, []) # Return empty list for other splits
    monkeypatch.setattr(datasets, "load_dataset", mock_load)
    # Expect the KeyError from the mock, which should be caught and logged by our function
    # The function might re-raise or just log, let's check the actual function behavior
    # Assuming it logs and raises a specific error or a generic Exception
    with pytest.raises(Exception) as ei:
        load_human_eval_dataset()
    # Check if the error message indicates failure to load or mentions the split
    assert "Failed to load HumanEval dataset" in str(ei.value) or "not found" in str(ei.value).lower()

def test_load_dataset_exception_propagates(monkeypatch):
    # Update mock to accept 'name' and 'split' kwargs
    def bad_load(name, split='test', **kwargs):
        raise RuntimeError("no go")
    monkeypatch.setattr(datasets, "load_dataset", bad_load)
    with pytest.raises(RuntimeError):
        load_human_eval_dataset()
