"""Ex vivo MesoChi workflows. Data are loaded only when a workflow is called."""


def prepare_exvivo_data(*args, **kwargs):
    """Load and prepare the six MATLAB inputs; see pipeline.prepare_exvivo_data."""
    from .pipeline import prepare_exvivo_data as prepare
    return prepare(*args, **kwargs)


def run_exvivo_reconstruction(*args, **kwargs):
    """Reconstruct with L2, TV, or both; see pipeline.run_exvivo_reconstruction."""
    from .pipeline import run_exvivo_reconstruction as run
    return run(*args, **kwargs)


def run_exvivo_lcurve(*args, **kwargs):
    """Evaluate two L2 L-curves; see lcurve.run_exvivo_lcurve."""
    from .lcurve import run_exvivo_lcurve as run
    return run(*args, **kwargs)
