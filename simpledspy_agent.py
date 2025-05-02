# simpledspy_agent_demo.py
import simpledspy
import dspy

# Configure a simple LM (replace with your preferred model if needed)
# simpledspy might handle this automatically if env vars are set,
# but being explicit can be clearer.
try:
    # lm = dspy.OpenAI(model="gpt-3.5-turbo")
    # Using a model from user preferences
    #lm = dspy.LM('openrouter/google/gemini-2.0-flash-001')
    #dspy.settings.configure(lm=lm)
    print("DSPy LM configured.")
except Exception as e:
    print(f"Could not configure LM: {e}")
    print("Ensure your API keys (e.g., OPENAI_API_KEY or OPENROUTER_API_KEY) are set.")
    # Exit if LM configuration fails, as the rest depends on it.
    exit(1)

# Agent Goal
goal = "Write a short blog post about the benefits of using DSPy."

print(f"--- Agent Goal ---")
print(goal)
print("-" * 25)

# 1. Planning Step: Break down the goal into steps
print("--- Step 1: Planning ---")
plan = simpledspy.pipe(
    goal, 
    description="Given a goal, break it down into a numbered list of actionable steps."
)
print("Generated Plan:")
print(plan)
print("-" * 25)

# 2. Action Step (Simulated): Elaborate on the first step
print("--- Step 2: Elaborating on First Action ---")
# Extract the first step (simple parsing, might need refinement)
try:
    first_step = plan.split('\n')[0] # Assumes numbered list format
    if first_step.strip() == "": # Handle potential empty lines
         first_step = plan.split('\n')[1]
    print(f"First step identified: {first_step}")
except IndexError:
    print("Could not parse the first step from the plan.")
    first_step = "Define the core message." # Fallback

elaboration = simpledspy.pipe(
    first_step, 
    description="Given a planning step, describe how you would execute it in detail."
)
print("\nElaboration on First Step:")
print(elaboration)
print("-" * 25)

print("Agent demo finished.")
