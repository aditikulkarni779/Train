"""Entry point: python main.py --config config/pipeline_config.yaml"""
from __future__ import annotations

import argparse
import sys

from core.config_loader import load_config
from core.logging_config import configure_logging
from pipeline_coordinator import PipelineCoordinator


def main() -> int:
    parser = argparse.ArgumentParser(description="Railway AI Inspection — Phase-1 orchestrator")
    parser.add_argument("--config", default="config/pipeline_config.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_logging(args.log_level)
    config = load_config(args.config)

    coordinator = PipelineCoordinator(config)
    coordinator.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
