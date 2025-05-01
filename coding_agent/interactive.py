import cmd
import coding_agent

class InteractiveShell(cmd.Cmd):
    intro = "Welcome to the Coding Agent interactive shell. Type help or ? to list commands."
    prompt = "coding-agent> "

    def __init__(self):
        super().__init__()
        # stateful attributes
        self.lm = None
        self.problems = []
        self.dataset = []
        self.devset = []
        self.coder = None
        self.optimized = None

    def do_configure_lm(self, arg):
        "Configure language model: configure_lm <lm_name>"
        name = arg.strip()
        if not name:
            print("Usage: configure_lm <lm_name>")
            return
        try:
            self.lm = coding_agent.configure_lm(name)
            print(f"LM configured: {name}")
        except Exception as e:
            print(f"Error configuring LM: {e}")

    def do_load(self, arg):
        "Load HumanEval dataset: load"
        try:
            self.problems = coding_agent.load_human_eval_dataset()
            print(f"Loaded {len(self.problems)} problems.")
        except Exception as e:
            print(f"Error loading dataset: {e}")

    def do_prepare(self, arg):
        "Prepare DSPy dataset from loaded problems: prepare"
        if not self.problems:
            print("No problems loaded. Use 'load' first.")
            return
        self.dataset = coding_agent.prepare_dspy_dataset(self.problems)
        print(f"Prepared DSPy dataset with {len(self.dataset)} examples.")

    def do_devset(self, arg):
        "Get devset: devset <subset_size>"
        parts = arg.split()
        if len(parts) != 1:
            print("Usage: devset <subset_size>")
            return
        try:
            size = int(parts[0])
        except ValueError:
            print("Subset size must be an integer.")
            return
        self.devset = coding_agent.get_devset(self.dataset, size)
        print(f"Devset size: {len(self.devset)}.")

    def do_init_coder(self, arg):
        "Initialize the SimpleCoder: init_coder"
        self.coder = coding_agent.SimpleCoder()
        print("Initialized SimpleCoder.")

    def do_pre(self, arg):
        "Run pre-optimization on the first example: pre"
        if self.coder is None:
            print("Coder not initialized. Use 'init_coder'.")
            return
        if not self.devset:
            print("Devset is empty. Use 'devset'.")
            return
        ex = self.devset[0]
        try:
            result = coding_agent.run_pre_optimization(self.coder, ex)
            self.last_pre = result
            print("Pre-optimization run completed.")
        except Exception as e:
            print(f"Error during pre-optimization: {e}")

    def do_optimize(self, arg):
        "Optimize agent: optimize <max_steps> <max_demos> <seed> <metric>"
        parts = arg.split()
        if len(parts) != 4:
            print("Usage: optimize <max_steps> <max_demos> <seed> <metric>")
            return
        try:
            max_steps = int(parts[0])
            max_demos = int(parts[1])
            seed = int(parts[2])
            metric_name = parts[3]
        except ValueError:
            print("max_steps, max_demos, and seed must be integers.")
            return
        metric_func = getattr(coding_agent, metric_name, None)
        if metric_func is None:
            print(f"Metric '{metric_name}' not found.")
            return
        if self.coder is None or not self.devset:
            print("Ensure coder initialized and devset prepared.")
            return
        try:
            self.optimized = coding_agent.optimize_agent(
                self.coder, self.devset, metric_func, max_steps, max_demos, seed
            )
            print("Optimization completed.")
        except Exception as e:
            print(f"Error during optimization: {e}")

    def do_post(self, arg):
        "Run post-optimization on the first example: post"
        if self.optimized is None:
            print("No optimized agent. Use 'optimize'.")
            return
        if not self.devset:
            print("Devset is empty.")
            return
        ex = self.devset[0]
        try:
            coding_agent.run_post_optimization(self.optimized, ex)
            print("Post-optimization run completed.")
        except Exception as e:
            print(f"Error during post-optimization: {e}")

    def do_exit(self, arg):
        "Exit the interactive shell"
        print("Exiting interactive shell.")
        return True

    def do_EOF(self, arg):
        return self.do_exit(arg)

def main():
    InteractiveShell().cmdloop()

if __name__ == "__main__":
    main()
