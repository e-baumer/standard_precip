"""Regenerate the numerical-equivalence golden file (tests/regression_golden.csv).

This script pins the full calculated-index series for every supported
(dist_type, fit_type) combination on data/monthly_data.csv. It was originally
run against the vendored l-moments code immediately before the migration to the
external lmoments3 package, proving numerical equivalence across that swap.

Regenerated once since: observations outside the support of the fitted
distribution now map to large finite index values (CDF clipped at 1e-16)
instead of NaN. That change affected only cells that were previously NaN;
every previously finite value is unchanged.

Rerun only when a numerical behavior change is intended:

    uv run python tests/gen_golden.py
"""

from pathlib import Path

import pandas as pd

from standard_precip.spi import SPI

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
GOLDEN_CSV = Path(__file__).resolve().parent / "regression_golden.csv"

COMBOS = [
    ("gam", "lmom", {}),
    ("exp", "lmom", {}),
    ("gev", "lmom", {}),
    ("gpa", "lmom", {}),
    ("gum", "lmom", {}),
    ("nor", "lmom", {}),
    ("pe3", "lmom", {}),
    ("wei", "lmom", {}),
    ("wak", "lmom", {}),
    ("gam", "mle", {"floc": 0}),
    ("exp", "mle", {}),
    ("gev", "mle", {}),
    ("gpa", "mle", {}),
    ("gum", "mle", {}),
    ("nor", "mle", {}),
    ("wei", "mle", {"floc": 0}),
    ("glo", "mle", {}),
    ("gno", "mle", {}),
    ("kap", "mle", {}),
]


def main():
    df_rainfall = pd.read_csv(DATA_DIR / "monthly_data.csv")
    frames = []
    for dist_type, fit_type, kwargs in COMBOS:
        result = SPI().calculate(
            df_rainfall,
            "date",
            "TotalPrecipitation",
            freq="M",
            fit_type=fit_type,
            dist_type=dist_type,
            **kwargs,
        )
        frames.append(
            pd.DataFrame(
                {
                    "dist_type": dist_type,
                    "fit_type": fit_type,
                    "date": result["date"].astype(str),
                    "index_value": result["TotalPrecipitation_calculated_index"],
                }
            )
        )
    golden = pd.concat(frames, ignore_index=True)
    golden.to_csv(GOLDEN_CSV, index=False, float_format="%.12g")
    print(f"Wrote {len(golden)} rows to {GOLDEN_CSV}")


if __name__ == "__main__":
    main()
