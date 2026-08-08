"""Backward-compatible entry point for RiskSight.

Running ``python main.py`` continues to work for source checkouts.
After installation, the recommended entry point is the ``risksight`` CLI.
"""

try:
    # Use the installed package when available.
    from risksight.cli import main
except ModuleNotFoundError:
    # Fall back to the local source tree when running from the repository.
    from src.risksight.cli import main


if __name__ == "__main__":
    raise SystemExit(main())