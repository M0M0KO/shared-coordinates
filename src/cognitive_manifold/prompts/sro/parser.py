"""Lossless, deterministic reconstruction of recorded SRO choice events."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import pandas as pd

from cognitive_manifold.data.splits import canonical_participant_id, participant_pseudonym
from cognitive_manifold.prompts.sro.specs import SPECS


@dataclass
class Event:
    source_row: int
    event_index: int
    stage: str
    block: str
    condition: str
    stimulus: str
    raw_response: str
    raw_candidates: tuple[str, ...]
    feedback: str
    reaction_time: float | None
    primary: bool
    valid: bool


def _missing(value: Any) -> bool:
    try:
        return value is None or bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _clean(value: Any) -> Any:
    if _missing(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    return value


def _compact(payload: dict[str, Any]) -> str:
    return json.dumps(
        {key: _clean(value) for key, value in payload.items() if not _missing(value)},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _key(value: Any) -> str:
    if _missing(value):
        return "missing"
    number = int(float(value))
    return "withhold" if number == -1 else f"key:{number}"


def _rt(row: pd.Series, column: str = "rt") -> float | None:
    value = row.get(column)
    return None if _missing(value) else float(value)


def _stage(row: pd.Series) -> str:
    value = row.get("exp_stage")
    return "" if _missing(value) else str(value)


def _block(row: pd.Series) -> str:
    for name in ("block_num", "which_round", "round_type", "stage"):
        value = row.get(name)
        if not _missing(value):
            return str(_clean(value))
    return ""


def _image_names(value: Any) -> list[str]:
    if _missing(value):
        return []
    return re.findall(r"(?:images/)?([A-Za-z0-9_-]+\.(?:png|jpg|jpeg|svg))", str(value))


def _label_map(participant: object, task: str, universe: Iterable[str]) -> dict[str, str]:
    values = list(dict.fromkeys(universe))
    canonical = canonical_participant_id(participant)
    values.sort(
        key=lambda value: hashlib.sha256(
            f"cognitive-manifold|original|response-map|{canonical}|{task}|{value}".encode()
        ).digest()
    )
    labels = [chr(ord("A") + index) for index in range(26)] + ["AA"]
    if len(values) > len(labels):
        raise ValueError(f"Response universe for {task} exceeds the frozen label set")
    return {raw: labels[index] for index, raw in enumerate(values)}


def _event(
    row: pd.Series,
    source_row: int,
    event_index: int,
    stimulus: dict[str, Any],
    response: str,
    candidates: Iterable[str],
    feedback: dict[str, Any] | None = None,
    *,
    rt_column: str = "rt",
    valid: bool = True,
) -> Event:
    task = str(row["_task"])
    stage = _stage(row)
    return Event(
        source_row=source_row,
        event_index=event_index,
        stage=stage,
        block=_block(row),
        condition=str(_clean(row.get("condition")) or ""),
        stimulus=_compact(stimulus),
        raw_response=response,
        raw_candidates=tuple(candidates),
        feedback=_compact(feedback or {}),
        reaction_time=_rt(row, rt_column),
        primary=stage in SPECS[task].primary_stages,
        valid=valid and response in candidates,
    )


def _standard_event(task: str, row: pd.Series, source_row: int) -> list[Event]:
    key = _key(row.get("key_press"))
    specs: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
        "adaptive_n_back": (
            {"letter": row.get("stim"), "load": row.get("load"), "stage": _stage(row)},
            {"correct": row.get("correct")},
        ),
        "attention_network_task": (
            {
                "cue": row.get("cue"),
                "location": row.get("flanker_location"),
                "middle_direction": row.get("flanker_middle_direction"),
                "flanker_type": row.get("flanker_type"),
            },
            {"correct": row.get("correct")},
        ),
        "dot_pattern_expectancy": (
            {
                "cue_probe_class": row.get("condition"),
                "probe_image": (_image_names(row.get("stimulus")) or [None])[0],
            },
            {"correct": row.get("correct")},
        ),
        "go_nogo": (
            {"stimulus_kind": row.get("condition")},
            {"correct": row.get("correct")},
        ),
        "probabilistic_selection": (
            {"symbols_left_to_right": _image_names(row.get("stimulus")), "phase": _stage(row)},
            {"feedback": row.get("feedback") if _stage(row) == "training" else None},
        ),
        "shift_task": (
            {"objects_left_to_right": row.get("stims")},
            {"feedback": row.get("feedback"), "total_points": row.get("total_points")},
        ),
        "simon": (
            {"color": row.get("stim_color"), "side": row.get("stim_side")},
            {"correct": row.get("correct")},
        ),
        "stroop": (
            {"display_html": row.get("stimulus")},
            {"correct": row.get("correct")},
        ),
        "threebytwo": (
            {
                "cue": row.get("cue"),
                "task": row.get("task"),
                "color": row.get("stim_color"),
                "number": row.get("stim_number"),
            },
            {"correct": row.get("correct")},
        ),
    }
    stimulus, feedback = specs[task]
    universe = SPECS[task].response_universe
    ordinary_timeout = task not in {"go_nogo"}
    return [
        _event(
            row,
            source_row,
            0,
            stimulus,
            key,
            universe,
            feedback,
            valid=not (ordinary_timeout and key == "withhold"),
        )
    ]


def _discount(row: pd.Series, source_row: int) -> list[Event]:
    if _missing(row.get("choice")):
        return []
    return [
        _event(
            row,
            source_row,
            0,
            {
                "left_or_right_options": row.get("stimulus"),
                "smaller_amount": row.get("smaller_amount"),
                "sooner_delay": row.get("sooner_delay"),
                "larger_amount": row.get("larger_amount"),
                "later_delay": row.get("later_delay"),
            },
            _key(row.get("key_press")),
            SPECS["discount_titrate"].response_universe,
        )
    ]


def _stop(task: str, row: pd.Series, source_row: int) -> list[Event]:
    return [
        _event(
            row,
            source_row,
            0,
            {
                "shape_image": (_image_names(row.get("stimulus")) or [None])[0],
                "stop_trial_type": row.get("SS_trial_type"),
                "stop_delay_ms": row.get("SS_delay"),
                "stop_signal": row.get("SS_stimulus"),
                "selection_condition": row.get("condition") if task.startswith("stim_") else None,
            },
            _key(row.get("key_press")),
            SPECS[task].response_universe,
            {"correct": row.get("correct"), "stopped": row.get("stopped")},
        )
    ]


def _cct(row: pd.Series, source_row: int, state: dict[str, Any]) -> list[Event]:
    click = str(_clean(row.get("mouse_click")) or "")
    if not click:
        return []
    current_round = _clean(row.get("which_round"))
    if "round" not in state or state.get("round") != current_round:
        state.clear()
        state.update({"round": current_round, "points": 0, "turned": 0})
    action = "collect" if click == "collectButton" else "turn"
    before = {"points": state["points"], "cards_turned": state["turned"]}
    event = _event(
        row,
        source_row,
        0,
        {
            "round": current_round,
            "gain_per_safe_card": row.get("gain_amount"),
            "loss_if_loss_card": row.get("loss_amount"),
            "loss_cards": row.get("num_loss_cards"),
            "total_cards": row.get("total_cards"),
            **before,
        },
        action,
        SPECS["columbia_card_task_hot"].response_universe,
        {
            "clicked_loss_card": row.get("clicked_on_loss_card"),
            "round_points_after": row.get("round_points"),
        },
    )
    if action == "turn":
        state["turned"] += 1
    if not _missing(row.get("round_points")):
        state["points"] = _clean(row.get("round_points"))
    return [event]


def _information_sampling(row: pd.Series, source_row: int, state: dict[str, Any]) -> list[Event]:
    click = row.get("mouse_click")
    if _missing(click):
        return []
    click_id = int(float(click))
    color = None if _missing(row.get("color_clicked")) else str(row.get("color_clicked")).lower()
    if _clean(row.get("which_click_in_round")) == 1 or state.get("stage") != _stage(row):
        state.clear()
        state.update({"stage": _stage(row), "opened": {}})
    opened_before = dict(state["opened"])
    response = (
        f"choose:{color}" if color in {"green", "red"} and click_id >= 25 else f"box:{click_id}"
    )
    candidates = [f"box:{index}" for index in range(25) if str(index) not in state["opened"]]
    candidates += ["choose:green", "choose:red"]
    event = _event(
        row,
        source_row,
        0,
        {
            "rule": _stage(row),
            "opened_boxes": opened_before,
            "available_box_ids": [
                item.split(":", 1)[1] for item in candidates if item.startswith("box:")
            ],
        },
        response,
        candidates,
        {
            "revealed_color": color if response.startswith("box:") else None,
            "reward": row.get("reward"),
        },
    )
    if response.startswith("box:") and color:
        state["opened"][str(click_id)] = color
    return [event]


def _tower(row: pd.Series, source_row: int, state: dict[str, Any]) -> list[Event]:
    click = str(_clean(row.get("mouse_click")) or "")
    match = re.fullmatch(r"tol_peg_([123])", click)
    if not match:
        return []
    problem = str(_clean(row.get("problem_id")) or "")
    if state.get("problem") != problem:
        state.clear()
        state.update(
            {"problem": problem, "position": row.get("current_position"), "selected": None}
        )
    event = _event(
        row,
        source_row,
        0,
        {
            "problem": problem,
            "current_position": state.get("position"),
            "target": row.get("target"),
            "minimum_moves": row.get("min_moves"),
            "previous_selected_peg": state.get("selected"),
        },
        f"peg:{match.group(1)}",
        SPECS["tower_of_london"].response_universe,
        {"position_after_click": row.get("current_position"), "solved": row.get("correct")},
    )
    state["position"] = row.get("current_position")
    state["selected"] = match.group(1) if state.get("selected") is None else None
    return [event]


def _two_stage(row: pd.Series, source_row: int) -> list[Event]:
    events: list[Event] = []
    first = _key(row.get("key_press_first"))
    second = _key(row.get("key_press_second"))
    universe = SPECS["two_stage_decision"].response_universe
    if first != "missing":
        events.append(
            _event(
                row,
                source_row,
                0,
                {"decision_stage": 1, "stimuli_left_to_right": row.get("stim_order_first")},
                first,
                universe,
                {
                    "transition": row.get("stage_transition"),
                    "second_stage": row.get("stage_second"),
                },
                rt_column="rt_first",
                valid=first != "withhold",
            )
        )
    if second != "missing":
        events.append(
            _event(
                row,
                source_row,
                1,
                {
                    "decision_stage": 2,
                    "second_stage": row.get("stage_second"),
                    "stimuli_left_to_right": row.get("stim_order_second"),
                },
                second,
                universe,
                {"reward": row.get("feedback")},
                rt_column="rt_second",
                valid=second != "withhold",
            )
        )
    return events


def _row_events(task: str, row: pd.Series, source_row: int, state: dict[str, Any]) -> list[Event]:
    if task in {
        "adaptive_n_back",
        "attention_network_task",
        "dot_pattern_expectancy",
        "go_nogo",
        "probabilistic_selection",
        "shift_task",
        "simon",
        "stroop",
        "threebytwo",
    }:
        return _standard_event(task, row, source_row)
    if task == "discount_titrate":
        return _discount(row, source_row)
    if task in {"stop_signal", "stim_selective_stop_signal"}:
        return _stop(task, row, source_row)
    if task == "columbia_card_task_hot":
        return _cct(row, source_row, state)
    if task == "information_sampling_task":
        return _information_sampling(row, source_row, state)
    if task == "tower_of_london":
        return _tower(row, source_row, state)
    if task == "two_stage_decision":
        return _two_stage(row, source_row)
    raise KeyError(task)


def _sort_session(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.copy()
    ordered["_source_row"] = range(len(ordered))
    sort_columns = [name for name in ("time_elapsed", "trial_id", "trial_num") if name in ordered]
    sort_columns.append("_source_row")
    return ordered.sort_values(sort_columns, kind="stable")


def build_session(
    task: str, participant: object, frame: pd.DataFrame, timepoint: str
) -> list[dict[str, Any]]:
    if task not in SPECS:
        raise KeyError(f"Unknown task: {task}")
    mapping = _label_map(participant, task, SPECS[task].response_universe)
    history: list[str] = []
    output: list[dict[str, Any]] = []
    state: dict[str, Any] = {}
    ordered = _sort_session(frame)
    for source_row, (_, original) in enumerate(ordered.iterrows()):
        row = original.copy()
        row["_task"] = task
        for event in _row_events(task, row, source_row, state):
            dynamic_mapping = mapping.copy()
            missing_candidates = [
                item for item in event.raw_candidates if item not in dynamic_mapping
            ]
            if missing_candidates:
                dynamic_mapping.update(
                    _label_map(participant, task, (*mapping, *missing_candidates))
                )
            option_text = ",".join(f"{dynamic_mapping[raw]}={raw}" for raw in event.raw_candidates)
            current = f"CURRENT {event.stimulus} OPTIONS[{option_text}]"
            prefix = build_prefix(SPECS[task].instructions, history, current)
            response = dynamic_mapping.get(event.raw_response, "")
            candidates = [dynamic_mapping[item] for item in event.raw_candidates]
            valid = bool(event.valid and response and response in candidates)
            output.append(
                {
                    "dataset": "SRO",
                    "participant": participant_pseudonym(participant),
                    "timepoint": timepoint,
                    "task": task,
                    "task_family": SPECS[task].family,
                    "trial": len(output),
                    "source_row": event.source_row,
                    "event_index": event.event_index,
                    "block": event.block,
                    "condition": event.condition,
                    "prompt_prefix": prefix,
                    "response": response,
                    "candidate_responses": candidates,
                    "reaction_time": event.reaction_time,
                    "validity": "primary"
                    if valid and event.primary
                    else "history_only"
                    if valid
                    else "invalid",
                    "raw_stage": event.stage,
                }
            )
            feedback = f" FEEDBACK {event.feedback}" if event.feedback != "{}" else ""
            historical_response = response if valid else "NO_VALID_RESPONSE"
            history.append(f"EVENT {event.stimulus} <<{historical_response}>>{feedback}")
    return output


def build_prefix(instructions: str, history: Iterable[str], current_stimulus: str) -> str:
    pieces = [f"INSTRUCTIONS {instructions}", *history, current_stimulus, "<<"]
    return "\n".join(pieces)


def candidate_responses(record: dict[str, Any]) -> tuple[str, ...]:
    return tuple(record["candidate_responses"])


def observed_response(record: dict[str, Any]) -> str:
    return str(record["response"])
