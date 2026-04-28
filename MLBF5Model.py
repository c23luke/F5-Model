"""Compatibility entrypoint for Streamlit Cloud.

This file intentionally delegates to `f5_full_upgraded.py` so local and cloud
deployments use one canonical app implementation.
"""

from f5_full_upgraded import main


if __name__ == "__main__":
    main()
