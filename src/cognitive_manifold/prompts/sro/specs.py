"""Frozen task semantics for the original SRO reconstruction audit.

Recorded trial fields remain authoritative. These short instructions state only mechanics that
were available to participants and deliberately omit derived correctness fields and hidden rules.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    family: str
    instructions: str
    primary_stages: tuple[str, ...]
    response_universe: tuple[str, ...]


SPECS: dict[str, TaskSpec] = {
    "adaptive_n_back": TaskSpec(
        "memory_context",
        "Letters appear sequentially. Decide whether the current letter matches the letter at "
        "the current n-back load, ignoring case.",
        ("adaptive", "control"),
        ("key:37", "key:40"),
    ),
    "dot_pattern_expectancy": TaskSpec(
        "memory_context",
        "A cue pattern is followed by a probe pattern. Classify the cue-probe combination using "
        "one of the two response keys.",
        ("test",),
        ("key:37", "key:40"),
    ),
    "attention_network_task": TaskSpec(
        "attention_interference",
        "Report whether the middle arrow points left or right. Cues and flankers may vary.",
        ("test",),
        ("key:37", "key:39"),
    ),
    "simon": TaskSpec(
        "attention_interference",
        "Classify the stimulus color with the assigned left or right response; its screen side is "
        "irrelevant.",
        ("test",),
        ("key:37", "key:39"),
    ),
    "stroop": TaskSpec(
        "attention_interference",
        "Report the ink color of the displayed color word and ignore the word identity.",
        ("test",),
        ("key:66", "key:71", "key:82"),
    ),
    "go_nogo": TaskSpec(
        "inhibitory_control",
        "Press for a go stimulus and withhold the response for a no-go stimulus.",
        ("test",),
        ("key:32", "withhold"),
    ),
    "stop_signal": TaskSpec(
        "inhibitory_control",
        "Classify each shape with the assigned key, but withhold the response if a stop signal "
        "appears after the stated delay.",
        ("test",),
        ("key:77", "key:90", "withhold"),
    ),
    "stim_selective_stop_signal": TaskSpec(
        "inhibitory_control",
        "Classify each shape with the assigned key. A stop signal requires withholding only under "
        "the stimulus-selective stopping rule.",
        ("test",),
        ("key:77", "key:90", "withhold"),
    ),
    "probabilistic_selection": TaskSpec(
        "learning_decision",
        "Choose the left or right symbol. During training, feedback reveals whether the choice was "
        "rewarded; use experience to guide later choices.",
        ("training", "test"),
        ("key:37", "key:39"),
    ),
    "two_stage_decision": TaskSpec(
        "learning_decision",
        "Make a left or right choice at each of two stages. The first choice changes the likely "
        "second-stage state, and the second choice can produce reward.",
        ("test",),
        ("key:37", "key:39"),
    ),
    "information_sampling_task": TaskSpec(
        "learning_decision",
        "Open boxes to reveal colors, then decide which color is the majority. Opened boxes and "
        "the current points rule remain visible within a round.",
        ("Fixed_Win", "Decreasing_Win"),
        tuple([f"box:{index}" for index in range(25)] + ["choose:green", "choose:red"]),
    ),
    "discount_titrate": TaskSpec(
        "valuation_planning",
        "Choose between a smaller-sooner amount and a larger-later amount as if the choices were "
        "real.",
        ("test",),
        ("key:80", "key:81"),
    ),
    "columbia_card_task_hot": TaskSpec(
        "valuation_planning",
        "In each round, either turn another card for the stated gain while risking the stated "
        "loss, "
        "or collect the current round points.",
        ("test",),
        ("turn", "collect"),
    ),
    "tower_of_london": TaskSpec(
        "valuation_planning",
        "Move colored balls between three pegs to match the target arrangement. Select pegs in "
        "sequence while obeying peg capacity and top-ball constraints.",
        ("test",),
        ("peg:1", "peg:2", "peg:3"),
    ),
    "shift_task": TaskSpec(
        "flexibility",
        "Choose one of three objects. Feedback indicates whether it matches the currently rewarded "
        "feature; the hidden rule can change.",
        ("test",),
        ("key:37", "key:40", "key:39"),
    ),
    "threebytwo": TaskSpec(
        "flexibility",
        "Use the current cue to classify the displayed number by parity, magnitude, or color with "
        "the assigned response key.",
        ("test",),
        ("key:77", "key:90"),
    ),
}
