"""
SFT training entrypoint for C2CProjectorV3.

This reuses the existing SFT trainer and swaps only the Rosetta wrapper and
projector factory so the rest of the training pipeline stays identical.
"""

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from script.train import SFT_train as _sft

from rosetta.model.projector_v3 import C2CProjectorV3, create_projector, save_projector
from rosetta.model.wrapper_v3 import RosettaModel


_sft.RosettaModel = RosettaModel
_sft.create_projector = create_projector
_sft.save_projector = save_projector
_sft.AllInOneProjector = C2CProjectorV3


if __name__ == "__main__":
    _sft.main()
