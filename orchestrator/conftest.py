"""Ensures `orchestrator/` is on sys.path regardless of pytest's invocation cwd."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
