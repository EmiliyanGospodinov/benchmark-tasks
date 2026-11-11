"""``inspect-robots`` with this task's plugins registered, without installing the package.

    python run.py run --embodiment yam_lock_picking --policy lock_picking_scripted ...

is the same as ``inspect-robots run ...`` after ``pip install -e .`` (pyproject.toml declares the same entry points).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from inspect_robots import registry  # noqa: E402
from inspect_robots.cli import main  # noqa: E402


@registry.embodiment("yam_lock_picking")
def _embodiment(**kwargs):
    from lock_picking.embodiment import LockEmbodiment
    return LockEmbodiment(**kwargs)


@registry.policy("lock_picking_scripted")
def _policy(**kwargs):
    from lock_picking.policy import LockScriptedPolicy
    return LockScriptedPolicy(**kwargs)


if __name__ == "__main__":
    sys.exit(main())
