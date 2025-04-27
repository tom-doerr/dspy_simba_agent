# Minimal test file to check if importing dspy causes issues with pytest collection

print("Attempting to import dspy...")
try:
    import dspy
    print("dspy imported successfully.")
except Exception as e:
    print(f"Error importing dspy: {e}")
    # Re-raise to ensure pytest sees the import error if it happens here
    raise

# Define a minimal class inheriting from dspy.Module
print("Defining MinimalDSPyModule...")
try:
    class MinimalDSPyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            # Add a dummy predictor if needed for structure
            self.dummy_predictor = dspy.Predict("question -> answer")

        def forward(self, question):
            return self.dummy_predictor(question=question)
    print("MinimalDSPyModule defined successfully.")
except Exception as e:
    print(f"Error defining MinimalDSPyModule: {e}")
    # Optional: Raise or skip depending on whether definition is critical for the test
    # For this diagnosis, let's allow the test to proceed if definition fails, but print clearly
    pass # Allow collection test to proceed even if definition fails

def test_dspy_was_imported():
    """A simple test that only runs if the module was loaded."""
    assert 'dspy' in globals(), "dspy should be imported globally in this module"
    print("Test function executed.")

def test_minimal_class_defined():
    """Checks if the minimal dspy class could be defined."""
    assert 'MinimalDSPyModule' in globals(), "MinimalDSPyModule class should be defined globally"
    print("MinimalDSPyModule definition check passed.")
