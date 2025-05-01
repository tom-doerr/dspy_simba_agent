import dspy

class CodingSignature(dspy.Signature):
    """DSPy Signature for the coding task.

    Takes a HumanEval prompt (code signature + docstring) and expects
    the Python function body as completion.
    """
    prompt: str = dspy.InputField(
        desc="The prompt from the HumanEval dataset (signature+docstring)."
    )
    completion: str = dspy.OutputField(
        desc="The generated Python code completion (function body)."
    )

class SimpleCoder(dspy.Module):
    """A simple DSPy Module that uses a Predictor based on CodingSignature."""
    def __init__(self) -> None:
        super().__init__()
        self.generate_code = dspy.Predict(CodingSignature)

    def forward(self, prompt: str) -> dspy.Prediction:
        """Runs the prediction."""
        return self.generate_code(prompt=prompt)
