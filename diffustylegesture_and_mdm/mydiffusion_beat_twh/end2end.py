import os
import runpy


if __name__ == "__main__":
    this_dir = os.path.dirname(os.path.abspath(__file__))
    canonical_entry = os.path.abspath(os.path.join(this_dir, "..", "end2end.py"))
    runpy.run_path(canonical_entry, run_name="__main__")
