#!/usr/bin/env python3
"""Alias for compare_prediction_head_ablation.py (internal ablation 07)."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).with_name("compare_prediction_head_ablation.py")), run_name="__main__")
