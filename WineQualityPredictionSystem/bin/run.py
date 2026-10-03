#!/usr/bin/env python3
"""
run.py
------
Entry point for the Wine Quality Prediction System.

This is the file you Run in VSCode (Run > Run Without Debugging, or the
Play button, or `python bin/run.py` in the terminal). It does nothing but
wire up the Python path to find the `src` package and start the menu -
all real logic lives inside src/.
"""

import os
import sys

# Add the project root (the folder that CONTAINS bin/ and src/) to
# sys.path, so `from src import menu` works no matter which directory
# you launch this script from.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.menu import WineQualityApp  # noqa: E402  (import after sys.path fix)


def main() -> None:
    app = WineQualityApp()
    try:
        app.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
