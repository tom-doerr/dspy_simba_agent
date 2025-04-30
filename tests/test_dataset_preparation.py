import dspy
import coding_agent as ca

class DummyExample:
    def __init__(self, prompt, problem):
        self.prompt = prompt
        self.problem = problem
        self.with_inputs_called = False

    def with_inputs(self, field):
        self.with_inputs_called = True
        self.field = field
        return self

def test_prepare_dspy_dataset(monkeypatch):
    problems = [
        {'prompt': 'aaa', 'foo': 1},
        {'prompt': 'bbb', 'bar': 2},
    ]
    calls = []
    def fake_Example(prompt, problem):
        calls.append((prompt, problem))
        return DummyExample(prompt, problem)
    monkeypatch.setattr(dspy, "Example", fake_Example)

    ds = ca.prepare_dspy_dataset(problems)
    assert len(ds) == 2
    for idx, ex in enumerate(ds):
        assert isinstance(ex, DummyExample)
        assert ex.prompt == problems[idx]['prompt']
        assert ex.problem is problems[idx]
        assert ex.with_inputs_called
        assert ex.field == 'prompt'

def test_get_devset_smaller_than_subset():
    data = list(range(3))
    # subset size is 5, data len=3
    dev = ca.get_devset(data, optimization_subset_size=5)
    assert dev is data  # returns original list

def test_get_devset_equal_or_larger_than_subset():
    data = list(range(10))
    dev = ca.get_devset(data, optimization_subset_size=4)
    assert dev == data[:4]
