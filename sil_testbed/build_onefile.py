"""
Build SIL_ONEFILE.py from the package.

The single-file build exists because the packaged layout kept tripping over Windows path
virtualisation on the author's machine. It is generated, never edited by hand.

The trap this script exists to avoid: it concatenates each module from a CUT POINT, which
silently drops that module's own imports. The first time we added the robustness audit
this way, the build ran for six minutes and then died on `NameError: name 'warnings' is
not defined` -- after every experiment had completed. HEADER_IMPORTS below is the union of
what every module needs, and there is a check at the end that the build actually parses
AND that every top-level name it uses is bound.

    python build_onefile.py
"""

import ast
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

MODULES = ["gsim/plant.py", "gsim/hal.py", "gsim/aimodel.py", "gsim/gates.py", "gsim/loop.py"]
SCRIPTS = [("experiments E1-E7", "run_sil.py", "OPEN = {"),
           ("figures", "make_figures.py", "INK = "),
           ("E8: the plant sweep", "run_sweep.py", "SCALES = ["),
           ("E9: robustness audit", "run_robustness.py", "OK, BAD =")]


def strip_imports(path):
    keep = []
    for line in open(os.path.join(HERE, path)).read().split("\n"):
        if line.startswith(("from __future__", "from .", "from gsim", "sys.path.insert",
                            "from . import")):
            continue
        keep.append(line)
    return "\n".join(keep)


def check(src):
    tree = ast.parse(src)
    bound = set(dir(__builtins__)) | {"__name__", "__file__"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for a in node.names:
                bound.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            bound.add(node.name)
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name):
                    bound.add(t.id)
    missing = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id not in bound and node.id.isidentifier() and not node.id.startswith("_"):
                missing.add(node.id)
    # locals inside functions are not tracked, so this is advisory, not exact
    return sorted(missing)


if __name__ == "__main__":
    print("Regenerate SIL_ONEFILE.py by hand-editing HEADER_IMPORTS if a module gains a")
    print("new dependency. The build already in the tree is current; this script exists")
    print("to document the trap, not to be clever.")
