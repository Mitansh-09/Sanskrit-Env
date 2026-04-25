"""
Shared reward constants — used across all graders.

Centralizes reward values to ensure consistency across tasks
and make tuning straightforward.
"""

from .glossary_grader import GlossaryGrader
from .sandhi_grader import SandhiGrader
from .coherence_grader import CoherenceGrader
from .samasa_grader import SamasaGrader
from .restoration_grader import RestorationGrader
from .consistency_grader import ConsistencyGrader

# ── Canonical reward constants ────────────────────────────────────────────────

REWARD_CORRECT = 1.00
REWARD_PARTIAL = 0.40
REWARD_WRONG = 0.00
REWARD_ADJACENT_SANDHI = 0.25

# ── Restoration-specific ─────────────────────────────────────────────────────

REWARD_TOOL_RELEVANT = 0.08
REWARD_TOOL_SUPPORTING = 0.04
REWARD_WORKFLOW_PAIR = 0.05
PENALTY_REDUNDANT_TOOL = -0.05
PENALTY_IRRELEVANT_TOOL = -0.05
PENALTY_BUDGET_WASTE_RATE = 0.10

# ── Shaping bounds ───────────────────────────────────────────────────────────

REWARD_FLOOR_SHAPED = 0.50
REWARD_CEILING_SHAPED = 0.95
REWARD_PARTIAL_FLOOR_SHAPED = 0.40
REWARD_WRONG_SHAPED = 0.00  # TRUE ZERO — do not change

__all__ = [
    "GlossaryGrader",
    "SandhiGrader",
    "CoherenceGrader",
    "SamasaGrader",
    "RestorationGrader",
    "ConsistencyGrader",
    "REWARD_CORRECT",
    "REWARD_PARTIAL",
    "REWARD_WRONG",
    "REWARD_ADJACENT_SANDHI",
    "REWARD_TOOL_RELEVANT",
    "REWARD_TOOL_SUPPORTING",
    "REWARD_WORKFLOW_PAIR",
    "PENALTY_REDUNDANT_TOOL",
    "PENALTY_IRRELEVANT_TOOL",
    "PENALTY_BUDGET_WASTE_RATE",
    "REWARD_FLOOR_SHAPED",
    "REWARD_CEILING_SHAPED",
    "REWARD_PARTIAL_FLOOR_SHAPED",
    "REWARD_WRONG_SHAPED",
]
