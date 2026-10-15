"""Phase 3 (new code): run frozen v1.1 with more input views, via scripts/run_extras.py with only the view list changed.

    python phase3/run_views.py <split> <n_views> --ids ids.json
Writes results/<split>/v11_views<n>/.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / 'scripts'))
import run_extras  # noqa: E402
from bladedelta.estimator import Config  # noqa: E402

VIEWS = {4: ('v001', 'v003', 'v004', 'v005'), 7: tuple(f'v00{i}' for i in range(7))}
split, n = sys.argv[1], int(sys.argv[2])
run_extras.Config = lambda: Config(views=VIEWS[n])
sys.argv = ['run_extras.py', split, 'v11', '--tag', f'views{n}'] + sys.argv[3:]
run_extras.main()
