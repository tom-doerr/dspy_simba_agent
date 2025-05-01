import logging
import dspy

def configure_logging(level: int = logging.DEBUG) -> None:
    """Configures basic logging."""
    logging.basicConfig(level=level, format="%(asctime)s - %(levelname)s - %(message)s")

def configure_lm(lm_name: str) -> dspy.LM:
    """Configures the language model using dspy."""
    logging.info(f"Configuring LM: {lm_name}")
    lm = dspy.LM(lm_name)
    dspy.configure(lm=lm)
    return lm
