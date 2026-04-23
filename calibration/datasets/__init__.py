"""calibration.datasets — dataset adapter registry for TurboQuant
calibration and `vllm bench serve` prompt generation.

Adding a new dataset: write a `*.py` with a class that implements the
DatasetAdapter protocol, import it below, and add one entry to
ADAPTERS.
"""

from .base import Conversation, DatasetAdapter, Message
from .bfcl import BFCLAdapter
from .builder import BuildReport, build
from .glaive import GlaiveAdapter
from .qwen_agent import QwenAgentAdapter
from .toolace import ToolACEAdapter
from .xlam import XLAMAdapter

ADAPTERS: dict[str, type] = {
    GlaiveAdapter.name: GlaiveAdapter,
    XLAMAdapter.name: XLAMAdapter,
    ToolACEAdapter.name: ToolACEAdapter,
    BFCLAdapter.name: BFCLAdapter,
    QwenAgentAdapter.name: QwenAgentAdapter,
}

__all__ = [
    "ADAPTERS",
    "BFCLAdapter",
    "BuildReport",
    "Conversation",
    "DatasetAdapter",
    "GlaiveAdapter",
    "Message",
    "QwenAgentAdapter",
    "ToolACEAdapter",
    "XLAMAdapter",
    "build",
]
