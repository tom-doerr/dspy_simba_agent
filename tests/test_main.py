import coding_agent as ca

def test_main_load_failure(monkeypatch):
    # configure_lm will throw -> main should catch and return quietly
    monkeypatch.setattr(ca, "configure_logging", lambda: None)
    monkeypatch.setattr(ca, "configure_lm", lambda name: (_ for _ in ()).throw(RuntimeError("bad")))
    # should not raise
    ca.main()

def test_main_empty_devset(monkeypatch):
    monkeypatch.setattr(ca, "configure_logging", lambda: None)
    monkeypatch.setattr(ca, "configure_lm", lambda name: None)
    monkeypatch.setattr(ca, "load_human_eval_dataset", lambda: [{'prompt':'x','task_id':'T'}])
    monkeypatch.setattr(ca, "prepare_dspy_dataset", lambda probs: [])
    # should not raise
    ca.main()

def test_main_full_flow(monkeypatch):
    calls = []
    monkeypatch.setattr(ca, "configure_logging", lambda: None)
    monkeypatch.setattr(ca, "configure_lm", lambda name: None)
    # stub dataset load
    dummy = [{'prompt': 'P', 'task_id': 'TID'}]
    monkeypatch.setattr(ca, "load_human_eval_dataset", lambda: dummy)
    # prepare returns one Example-like object
    class Ex:
        def __init__(self, p):
            self.problem = {'prompt': p, 'task_id': 'TID'}
    example = Ex('P')
    monkeypatch.setattr(ca, "prepare_dspy_dataset", lambda probs: [example])
    monkeypatch.setattr(ca, "get_devset", lambda ds, optimization_subset_size: [example])
    monkeypatch.setattr(ca, "run_pre_optimization", lambda coder, ex: calls.append(("pre", ex)))
    monkeypatch.setattr(ca, "optimize_agent", lambda coder, ts, metric, max_steps, max_demos, seed: "OPT")
    monkeypatch.setattr(ca, "run_post_optimization", lambda optimized, ex: calls.append(("post", ex)))
    ca.main()
    assert calls == [("pre", example), ("post", example)]
