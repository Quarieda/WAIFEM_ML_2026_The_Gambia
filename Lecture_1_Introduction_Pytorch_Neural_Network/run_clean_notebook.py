"""Execute the cleaned notebook cell-by-cell and report failures.

Uses nbclient with allow_errors=True so we get the full list of failing cells
rather than bailing at the first one. After running, we print a summary of
which cells errored and their tracebacks (truncated).
"""
import json
import sys
import time
import nbformat
from nbclient import NotebookClient
from pathlib import Path

NB = Path("Zero_to_Hero_in_PyTorch_CLEAN.ipynb")

nb = nbformat.read(NB, as_version=4)

# Strip ipython magics that don't work outside a frontend, except %matplotlib inline
# which the kernel handles fine.
for cell in nb.cells:
    if cell.cell_type == "code":
        # Comment out plt.ion()/plt.ioff() to avoid interactive backend issues
        pass

t0 = time.time()
client = NotebookClient(
    nb,
    timeout=900,
    kernel_name="py-clean",
    allow_errors=True,
    raise_on_iopub_timeout=False,
)
client.execute()
elapsed = time.time() - t0
print(f"\n=== Execution finished in {elapsed:.1f}s ===")

errors = []
for i, cell in enumerate(nb.cells):
    if cell.cell_type != "code":
        continue
    for out in cell.get("outputs", []):
        if out.get("output_type") == "error":
            tb = "\n".join(out.get("traceback", []))
            errors.append((i, out.get("ename"), out.get("evalue"), tb))
            break

if not errors:
    print("ALL CELLS PASSED")
else:
    print(f"\n{len(errors)} cell(s) failed:")
    for i, ename, evalue, tb in errors:
        print(f"\n--- Cell {i} | {ename}: {evalue} ---")
        # strip ANSI
        import re
        tb = re.sub(r"\x1b\[[0-9;]*m", "", tb)
        print(tb[-2500:])

# Also persist the executed notebook (with outputs) for inspection.
nbformat.write(nb, NB.with_name("Zero_to_Hero_in_PyTorch_EXECUTED.ipynb"))
print(f"\nSaved executed notebook to {NB.with_name('Zero_to_Hero_in_PyTorch_EXECUTED.ipynb')}")
