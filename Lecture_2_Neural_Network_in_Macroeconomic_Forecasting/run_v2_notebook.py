"""Execute inflation_forecast_comparison_v2.ipynb end-to-end and save outputs in place."""
import time, sys
import nbformat
from nbclient import NotebookClient

NB_PATH = r"G:\My Drive\Colab Notebooks\ML_WAIFEM_2026\ML\inflation_forecast_comparison_v2.ipynb"

print(f"Executing {NB_PATH} ...")
t0 = time.time()

nb = nbformat.read(NB_PATH, as_version=4)
client = NotebookClient(
    nb,
    timeout=3600,                # per-cell timeout (1h cap for DNN cell)
    kernel_name="python3",
    allow_errors=False,
    record_timing=True,
)

try:
    client.execute(cwd=r"G:\My Drive\Colab Notebooks\ML_WAIFEM_2026\ML")
except Exception as e:
    print(f"!! Execution failed: {type(e).__name__}: {e}", flush=True)
    # Save partial outputs anyway so we can inspect
    nbformat.write(nb, NB_PATH)
    raise

nbformat.write(nb, NB_PATH)
dt = time.time() - t0
print(f"DONE in {dt:.1f}s ({dt/60:.1f} min). Notebook saved to {NB_PATH}")
