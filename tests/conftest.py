import sys
from pathlib import Path

# Only needed when running tests without `pip install -e .`
_src = Path(__file__).parent.parent / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
