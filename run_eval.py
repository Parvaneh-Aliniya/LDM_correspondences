"""
run_eval.py
===========

Thin wrapper around the LDM_correspondences eval/eval.py that:

  1. Imports embed_ldm_integration FIRST (so the dataset shim is registered
     before the model code's argparse runs)
  2. Forwards all command-line args to eval/eval.py's __main__

Place this at the ROOT of the LDM_correspondences repo, next to eval/ and
embed_ldm/, then invoke it INSTEAD of `python -m eval.eval`:

    python run_eval.py --benchmark custom --datapath /path ... [other args]

The shim activates when EMBED_ENABLE=1 is in the environment AND
--benchmark custom is passed. (We can't use --benchmark embed because the
upstream argparse hardcodes the choices to spair/pfwillow/cubs/custom.)
"""

# Step 1 — install the EMBED dataset hook before argparse fires
import embed_ldm_integration  # noqa: F401  (side-effect import)

# Step 2 — run eval/eval.py as if it had been invoked normally
import runpy

if __name__ == '__main__':
    # runpy executes the script with __name__ = '__main__', so its argparse
    # block runs against sys.argv (which we leave intact)
    runpy.run_path('eval/eval.py', run_name='__main__')
