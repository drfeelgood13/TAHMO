import sys
from pathlib import Path

# The tests import the scripts both as `downloaders.downloader_planet` (repository root on the path) and as
# `downloader_google_maps` (downloaders/ on the path); make both work from any working directory.
DOWNLOADERS = Path(__file__).resolve().parents[1]
for path in (DOWNLOADERS.parent, DOWNLOADERS):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
