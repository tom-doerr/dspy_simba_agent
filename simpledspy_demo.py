# simpledspy_demo.py
from simpledspy import pipe

print("--- Basic Text Processing ---")
# The README example implies automatic cleaning, but doesn't specify how.
# Assuming it's a placeholder or relies on a configured LM.
# Let's try a simple transformation task instead.
input_text = "Translate to French: Hello world"
translation = pipe(input_text, description="Translate the input text to French")
print(f"Input: '{input_text}'")
print(f"Output (Translation): '{translation}'")
print("-" * 25)


print("--- Multiple Outputs ---")
# Example from README: Extracting name and age
input_string = "John Doe, 30 years old"
# We need to tell pipe what outputs we expect
# The README syntax `name, age = pipe(...)` might be conceptual.
# Let's assume we pass output names or a structure.
# Re-interpreting based on typical DSPy: we might define output fields.
# Since `simpledspy` aims for simplicity, perhaps it infers from variable names?
# Let's try the README syntax directly, assuming `pipe` handles it.
try:
    # This syntax requires pipe to understand the assignment target.
    # Let's define expected output names explicitly if the direct assignment isn't standard.
    # Since the README shows it, let's try it first.
    # If it fails, we might need to adjust based on the library's actual implementation.
    extracted_name, extracted_age = pipe(
        input_string,
        output_names=["name", "age"], # Assuming an explicit way is needed
        description="Extract name and age from the text"
    )
    print(f"Input: '{input_string}'")
    print(f"Output (Name): '{extracted_name}'")
    print(f"Output (Age): {extracted_age}") # Age might be string or int depending on LM
except Exception as e:
    print(f"Could not run multiple output example as shown in README: {e}")
    print("The library might require a different syntax for multiple outputs.")
    # Placeholder if direct assignment fails
    extracted_info = pipe(
        input_string,
        description="Extract name and age from the text"
        # output_format={"name": "string", "age": "integer"} # Hypothetical format hint
    )
    print(f"Input: '{input_string}'")
    print(f"Raw Output: {extracted_info}")

print("-" * 25)


print("--- Custom Description (Multiple Inputs) ---")
# Example from README: Combining names
first_name = "Jane"
last_name = "Smith"
# The pipe function needs to accept multiple inputs.
# The README shows `pipe(\"John\", \"Doe\", description=...)`
# Let's assume positional arguments are treated as inputs.
combined_name = pipe(
    first_name,
    last_name,
    description="Combine first and last names into a full name"
)
print(f"Inputs: '{first_name}', '{last_name}'")
print(f"Output (Full Name): '{combined_name}'")
print("-" * 25)

print("Demo finished.")
