"""
SanskritEnv Pydantic models — Action, Observation, State.

These types define the complete API contract for SanskritEnv.
All fields are documented. All types are JSON-serializable.
"""

from typing import List, Optional, Dict, Any, Union
from openenv.core.env_server import Action, Observation, State


class ManuscriptAction(Action):
    """
    The action an agent takes each step.

    For Tasks 1-4: set action_type="option_select" and selected_option.
    For Task 5 tool calls: set action_type="tool_call", tool_name, tool_input.
    For Task 5 commit: set action_type="commit" and final_answer.
    """
    action_type: str = "option_select"
    """'option_select' (tasks 1-4) | 'tool_call' | 'commit' (task 5)."""

    selected_option: str = ""
    """The agent's chosen interpretation. Must match one candidate_options entry exactly."""

    tool_name: Optional[str] = None
    """Tool to call (task 5 tool_call only)."""

    tool_input: Optional[str] = None
    """Input to the tool (task 5 tool_call only)."""

    final_answer: Optional[str] = None
    """Final interpretation (task 5 commit only)."""

    confidence: float = 0.5
    """Agent's self-reported confidence, 0.0-1.0. Used for logging only, not graded."""

    reasoning: str = ""
    """Agent's explanation of its choice. Logged for analysis, not graded."""


class ManuscriptObservation(Observation):
    """
    What the agent observes at each step.

    Inherits: done: bool, reward: Optional[float] from Observation base.

    The agent must read source_text_iast, decision_prompt, and
    candidate_options, then return a ManuscriptAction with
    selected_option set to exactly one of the candidate_options strings.
    """
    # Episode metadata
    task_id: str
    """Which task: 'glossary_anchoring' | 'sandhi_resolution' | 'referential_coherence'"""

    episode_id: str
    """Unique identifier for this episode."""

    # Source text
    source_text_iast: str
    """The Sanskrit passage in IAST (International Alphabet of Sanskrit Transliteration)."""

    source_text_devanagari: str
    """The Sanskrit passage in Devanagari script."""

    english_context: str
    """Brief English description of the text's source and domain."""

    domain: str
    """Domain of the passage: 'ayurveda' | 'astronomy' | 'philosophy' | 'narrative'"""

    # For Task 1 (Glossary Anchoring)
    target_term_iast: Optional[str] = None
    """The specific term the agent must interpret (Task 1 only)."""

    active_glossary: Optional[Dict[str, str]] = None
    """Domain glossary entries for reference (Task 1 only)."""

    # For Task 2 (Sandhi Resolution)
    compound_iast: Optional[str] = None
    """The compound word to split (Task 2 only)."""

    # For Task 3 (Referential Coherence)
    verses_so_far: Optional[List[Dict[str, Any]]] = None
    """All verses seen so far in this episode (Task 3 only). List of dicts with keys: verse_num, iast, english."""

    current_verse_num: Optional[int] = None
    """Current verse number being processed (Task 3 only)."""

    # Decision interface
    decision_prompt: str
    """The specific question the agent must answer this step."""

    candidate_options: List[str]
    """
    Exactly 4 options. The agent must select one verbatim.
    Selecting a string not in this list returns reward=0 and done=True.
    """

    # Feedback
    step_reward: float = 0.0
    """Reward emitted by the immediately preceding environment step. 0.0 before any action is taken."""

    cumulative_score: float = 0.0
    """Current episode score on the environment's emitted reward scale."""

    feedback_message: str = ""
    """Human-readable explanation of the previous step's reward."""

    # Consistency tracker (Task 3)
    consistency_history: Optional[List[Dict[str, str]]] = None
    """Prior checkpoint Q&A for this episode. Agent should maintain consistency."""

    # Task 5 (Manuscript Restoration)
    tool_call_history: Optional[List[Dict[str, Any]]] = None
    """History of tool calls and their outputs (task 5 only)."""

    steps_remaining: Optional[int] = None
    """Remaining tool budget (task 5 only)."""

    available_tools: Optional[List[str]] = None
    """List of available tool names (task 5 only)."""

    last_tool_output: Optional[Dict[str, Any]] = None
    """Output from the most recent tool call (task 5 only)."""

    ocr_noise_level: Optional[float] = None
    """OCR noise level applied to the passage (task 5 only)."""

    difficulty: Optional[str] = None
    """Current difficulty level (task 5 only)."""


class ManuscriptState(State):
    """
    Episode-level state. Persists across all steps of an episode.

    Inherits: episode_id: Optional[str], step_count: int from State base.
    """
    task_id: str = ""
    """Active task identifier."""

    passage_id: str = ""
    """Which passage/episode is loaded."""

    total_decisions: int = 0
    """Total number of graded decisions in this episode."""

    correct_decisions: int = 0
    """Decisions scored as fully correct so far."""

    partial_decisions: int = 0
    """Decisions scored as partially correct so far."""

    decision_history: List[Dict[str, Any]] = []
    """Full trace: [{step, prompt, selected, correct, reward, timestamp}]"""

    consistency_map: Dict[str, str] = {}
    """For Task 3: maps referent labels to resolved antecedents across this episode."""

    is_complete: bool = False
    """True when all decisions in the episode have been made."""

    # Task 5 adaptive difficulty
    current_difficulty: str = "beginner"
    """Current difficulty level for manuscript_restoration."""

    agent_recent_scores: List[float] = []
    """Rolling window of last 10 episode scores for difficulty curriculum."""

    difficulty_escalations: int = 0
    """Number of times difficulty has been escalated."""

    evidence_use_history: List[int] = []
    """Number of distinct tools used per episode."""
