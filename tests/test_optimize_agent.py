import dspy.teleprompt
from coding_agent.optimization import optimize_agent

class DummyOptimizer:
    def __init__(self, metric, max_steps, max_demos):
        self.metric = metric
        self.max_steps = max_steps
        self.max_demos = max_demos
        self.compile_called = False

    def compile(self, coder, trainset, seed):
        self.compile_called = True
        return "OPTIMIZED"

def test_optimize_agent(monkeypatch):
    # stub SIMBA in dspy.teleprompt
    monkeypatch.setattr(dspy.teleprompt, "SIMBA", DummyOptimizer)
    coder = object()
    trainset = [1,2,3]
    # Use imported function directly, pass metric positionally (as a callable), add timeout
    dummy_metric = lambda gold, pred, trace=None: 0.5 # Dummy metric function
    optimized = optimize_agent(coder, trainset, dummy_metric, max_steps=2, max_demos=3, seed=99, timeout=10)
    assert optimized == "OPTIMIZED"
