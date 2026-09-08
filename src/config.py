"""Central configuration.

Every tunable lives here so no magic numbers end up scattered through the
notebooks. If a number matters to a result, it belongs in this file.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Resolved relative to this file, so imports work no matter where you launch
# Jupyter or Python from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FIGURES = PROJECT_ROOT / "reports" / "figures"

# --- What we are forecasting -----------------------------------------------
TICKER = "NFLX"

# Peers, used for the correlation heatmap and the risk-versus-return scatter.
# DIS and WBD are streaming competitors; AMZN is a partial competitor with a
# very different business mix; SPY is the market benchmark.
PEERS = ["DIS", "WBD", "AMZN", "SPY"]

# ~10 years. Deliberately spans the 2022 subscriber-loss crash and the
# streaming-wars regime shift, both of which we call out rather than hide.
START = "2015-01-01"
END = None  # None means "up to today"

# --- Modeling --------------------------------------------------------------
WINDOW = 60          # days of history fed to the LSTM, matching the reference notebook
HORIZON = 1          # predict one day ahead

TRAIN_FRAC = 0.70    # chronological splits — never shuffled
VAL_FRAC = 0.15
TEST_FRAC = 0.15

RANDOM_SEED = 7


def ensure_dirs() -> None:
    """Create the data and figure directories if they do not exist."""
    for d in (DATA_RAW, DATA_PROCESSED, FIGURES):
        d.mkdir(parents=True, exist_ok=True)
