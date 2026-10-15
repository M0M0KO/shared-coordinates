#!/usr/bin/env python3
"""Prepare the frozen Psych-101 donor-response control and launch matched QLoRA training."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import yaml

from cognitive_manifold.data.marked_sessions import (
    marked_responses,
    parse_marked_choices,
    replace_marked_responses,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/raw/Psych-101/prompts_training.jsonl"
OUTPUT = ROOT / "data/derived/original/psych101_shuffled/prompts_training.jsonl"
MANIFEST = ROOT / "manifests/private/original/psych101_shuffled.json"
TRAINING_MANIFEST = ROOT / "manifests/private/original/psych101_control_training.json"
ENVIRONMENT_MANIFEST = ROOT / "manifests/private/original/o8_training_environment.json"
CONFIG = ROOT / "configs/original/main.yaml"
BASE_MODEL = ROOT / "models/Llama-3.1-8B"
TRAINING_ROOT = ROOT / "models/original_control_training"
FINAL_MODELS = {
    "intact": ROOT / "models/Psych101-Reproduced-8B",
    "shuffled": ROOT / "models/Psych101-Response-Shuffled-8B",
}
SEED = 100


@dataclass(frozen=True)
class SessionIndex:
    line: int
    experiment: str
    participant: str
    responses: tuple[str, ...]
    condition_keys: tuple[str, ...]
    choice_response_sets: tuple[tuple[str, ...], ...]
    open_responses: tuple[bool, ...]
    available_response_sets: tuple[tuple[str, ...], ...]
    open_response_hints: tuple[bool, ...]


@dataclass(frozen=True)
class ChoiceDonor:
    line: int
    participant: str
    trial: int
    condition_key: str
    response_set: tuple[str, ...]
    open_response: bool
    response: str


def _condition_keys(text: str) -> tuple[str, ...]:
    choices = parse_marked_choices(text)
    keys: list[str] = []
    previous = 0
    for choice in choices:
        marker = text.find("<<", previous)
        if marker < 0:
            raise RuntimeError("Marked-response parser and condition parser disagree")
        local = re.sub(r"\s+", " ", text[previous:marker]).strip()
        keys.append(hashlib.sha256(local.encode()).hexdigest()[:16])
        previous = choice.suffix_start + 2
    return tuple(keys)


def _response_signature(value: str) -> str:
    stripped = value.strip()
    if re.fullmatch(r"[-+]?\d+(?:\.\d+)?", stripped):
        return "numeric"
    if len(stripped) == 1:
        return "single_character"
    return "text"


def _available_response_sets(text: str, responses: tuple[str, ...]) -> tuple[tuple[str, ...], ...]:
    """Extract explicit single-key options from each local trial prompt when present."""
    choices = parse_marked_choices(text)
    spaces: list[tuple[str, ...]] = []
    previous = 0
    for choice, response in zip(choices, responses, strict=True):
        marker = text.find("<<", previous)
        if marker < 0:
            raise RuntimeError("Marked-response parser and option parser disagree")
        local_line = text[previous:marker].rsplit("\n", 1)[-1]
        available: tuple[str, ...] = ()
        if _response_signature(response) == "single_character":
            for fragment in reversed(re.split(r"[.!?]\s*", local_line)):
                fragment = re.sub(r"\b(?:Q|A)\s*:", " ", fragment)
                tokens = tuple(
                    sorted(
                        set(
                            re.findall(
                                r"(?<![A-Za-z0-9])[A-Za-z](?![A-Za-z0-9])",
                                fragment,
                            )
                        )
                    )
                )
                if response in tokens and 2 <= len(tokens) <= 8:
                    available = tokens
                    break
        spaces.append(available)
        previous = choice.suffix_start + 2
    return tuple(spaces)


def _open_response_hints(text: str, responses: tuple[str, ...]) -> tuple[bool, ...]:
    """Identify prompt language that explicitly requests free text entry."""
    choices = parse_marked_choices(text)
    hints: list[bool] = []
    previous = 0
    for choice, response in zip(choices, responses, strict=True):
        marker = text.find("<<", previous)
        if marker < 0:
            raise RuntimeError("Marked-response parser and open-response parser disagree")
        local_line = text[previous:marker].rsplit("\n", 1)[-1]
        hints.append(
            _response_signature(response) == "text"
            and bool(
                re.search(
                    r"\b(?:type(?:\s+out)?|write|enter|recall|name|describe|explain|provide)\b",
                    local_line,
                    flags=re.IGNORECASE,
                )
            )
        )
        previous = choice.suffix_start + 2
    return tuple(hints)


def _choice_spaces(
    responses: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], ...], tuple[bool, ...]]:
    spaces: list[tuple[str, ...]] = [()] * len(responses)
    open_flags: list[bool] = [False] * len(responses)
    signatures = [_response_signature(value) for value in responses]
    start = 0
    while start < len(responses):
        stop = start + 1
        while stop < len(responses) and signatures[stop] == signatures[start]:
            stop += 1
        values = tuple(sorted(set(responses[start:stop])))
        open_response = len(values) > 12 and len(values) / (stop - start) > 0.15
        response_set = () if open_response else values
        for index in range(start, stop):
            spaces[index] = response_set
            open_flags[index] = open_response
        start = stop
    return tuple(spaces), tuple(open_flags)


def _effective_choice_space(
    session: SessionIndex,
    trial: int,
    condition_values: dict[str, list[str]],
    condition_spaces: dict[str, tuple[tuple[str, ...], bool]],
    trial_values: dict[int, list[str]],
    trial_spaces: dict[int, tuple[tuple[str, ...], bool]],
) -> tuple[tuple[str, ...], bool]:
    if session.available_response_sets[trial]:
        return session.available_response_sets[trial], False
    if session.open_response_hints[trial]:
        return (), True
    condition = session.condition_keys[trial]
    condition_set, condition_open = condition_spaces[condition]
    if len(condition_values[condition]) > 1:
        return condition_set, condition_open
    # Dynamically named stimuli can make otherwise identical numeric rating
    # prompts hash differently for every participant.  Within an experiment,
    # the same trial position still has the same numeric response grammar.
    signature = _response_signature(session.responses[trial])
    if len(trial_values[trial]) > 1:
        trial_set, trial_open = trial_spaces[trial]
        if signature == "numeric" or (signature == "text" and trial_open):
            return trial_set, trial_open
    return session.choice_response_sets[trial], session.open_responses[trial]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _masked_payload(payload: dict[str, object]) -> bytes:
    masked = dict(payload)
    text = str(masked["text"])
    masked["text"] = replace_marked_responses(text, ["__RESPONSE__"] * len(marked_responses(text)))
    return (json.dumps(masked, ensure_ascii=False, sort_keys=True) + "\n").encode()


def index_source() -> list[SessionIndex]:
    records: list[SessionIndex] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            payload = json.loads(line)
            responses = marked_responses(payload["text"])
            if not responses:
                raise RuntimeError(f"No marked responses on source line {line_number}")
            response_sets, open_responses = _choice_spaces(responses)
            records.append(
                SessionIndex(
                    line=line_number,
                    experiment=str(payload["experiment"]),
                    participant=str(payload["participant"]),
                    responses=responses,
                    condition_keys=_condition_keys(payload["text"]),
                    choice_response_sets=response_sets,
                    open_responses=open_responses,
                    available_response_sets=_available_response_sets(payload["text"], responses),
                    open_response_hints=_open_response_hints(payload["text"], responses),
                )
            )
    return records


def _select_candidate(
    candidates: list[ChoiceDonor],
    target: SessionIndex,
    target_response_set: tuple[str, ...],
    target_open_response: bool,
    trial: int,
    level: str,
) -> tuple[ChoiceDonor, str] | None:
    if not candidates:
        return None
    digest = hashlib.sha256(f"{SEED}|{target.line}|{trial}|{level}".encode()).digest()
    start = int.from_bytes(digest[:8], "big") % len(candidates)
    for offset in range(len(candidates)):
        candidate = candidates[(start + offset) % len(candidates)]
        if candidate.participant == target.participant:
            continue
        if _response_signature(candidate.response) != _response_signature(target.responses[trial]):
            continue
        if candidate.open_response != target_open_response:
            continue
        if target_open_response:
            return candidate, candidate.response
        if len(candidate.response_set) != len(target_response_set):
            continue
        donor_index = candidate.response_set.index(candidate.response)
        return candidate, target_response_set[donor_index]
    return None


def donor_replacements(
    records: list[SessionIndex],
) -> tuple[dict[int, tuple[str, ...]], Counter[str]]:
    by_experiment: dict[str, list[SessionIndex]] = defaultdict(list)
    for record in records:
        by_experiment[record.experiment].append(record)
    replacements: dict[int, tuple[str, ...]] = {}
    levels: Counter[str] = Counter()
    same_participant = 0
    invalid_response = 0
    for experiment, sessions in by_experiment.items():
        # Infer the response space from all participants who saw the identical
        # local prompt context.  A single participant's observed answers are
        # not an allowed-label vocabulary: for example, an eight-question
        # probability block can contain only a few of the valid 0--100 values.
        condition_values: dict[str, list[str]] = defaultdict(list)
        trial_values: dict[int, list[str]] = defaultdict(list)
        for session in sessions:
            for trial, (condition, response) in enumerate(
                zip(session.condition_keys, session.responses, strict=True)
            ):
                condition_values[condition].append(response)
                trial_values[trial].append(response)
        condition_spaces: dict[str, tuple[tuple[str, ...], bool]] = {}
        for condition, values in condition_values.items():
            unique = tuple(sorted(set(values)))
            open_response = len(unique) > 12 and len(unique) / len(values) > 0.15
            condition_spaces[condition] = (() if open_response else unique, open_response)
        trial_spaces: dict[int, tuple[tuple[str, ...], bool]] = {}
        for trial, values in trial_values.items():
            unique = tuple(sorted(set(values)))
            open_response = len(unique) > 12 and len(unique) / len(values) > 0.15
            trial_spaces[trial] = (() if open_response else unique, open_response)

        fixed_sets = sorted(
            {
                response_set
                for session in sessions
                for trial in range(len(session.responses))
                for response_set, open_response in (
                    _effective_choice_space(
                        session,
                        trial,
                        condition_values,
                        condition_spaces,
                        trial_values,
                        trial_spaces,
                    ),
                )
                if not open_response and response_set
            },
            key=lambda value: (-len(value), value),
        )
        expanded_fixed_sets = {
            value: next(candidate for candidate in fixed_sets if set(value).issubset(candidate))
            for value in fixed_sets
        }
        indices: dict[str, dict[object, list[ChoiceDonor]]] = {
            name: defaultdict(list)
            for name in (
                "trial_condition_set",
                "trial_condition",
                "trial_set",
                "trial_cardinality",
                "trial",
                "condition_set",
                "condition_cardinality",
                "condition",
                "response_set",
                "response_cardinality",
                "trial_mode",
                "condition_mode",
                "mode",
            )
        }
        all_choices: list[ChoiceDonor] = []
        for session in sessions:
            if not (
                len(session.responses)
                == len(session.condition_keys)
                == len(session.choice_response_sets)
                == len(session.open_responses)
                == len(session.available_response_sets)
                == len(session.open_response_hints)
            ):
                raise RuntimeError(f"Condition-key count mismatch on line {session.line}")
            for trial, (response, condition) in enumerate(
                zip(session.responses, session.condition_keys, strict=True)
            ):
                response_set, open_response = _effective_choice_space(
                    session,
                    trial,
                    condition_values,
                    condition_spaces,
                    trial_values,
                    trial_spaces,
                )
                response_set = (
                    ()
                    if open_response
                    else response_set
                    if session.available_response_sets[trial]
                    else expanded_fixed_sets.get(response_set, response_set)
                )
                choice = ChoiceDonor(
                    line=session.line,
                    participant=session.participant,
                    trial=trial,
                    condition_key=condition,
                    response_set=response_set,
                    open_response=open_response,
                    response=response,
                )
                mode = (
                    f"open:{_response_signature(response)}"
                    if open_response
                    else f"fixed:{_response_signature(response)}"
                )
                indices["trial_condition_set"][(trial, condition, response_set)].append(choice)
                indices["trial_condition"][(trial, condition)].append(choice)
                indices["trial_set"][(trial, response_set)].append(choice)
                indices["trial_cardinality"][(trial, len(response_set))].append(choice)
                indices["trial"][trial].append(choice)
                indices["condition_set"][(condition, response_set)].append(choice)
                indices["condition_cardinality"][(condition, len(response_set))].append(choice)
                indices["condition"][condition].append(choice)
                indices["response_set"][response_set].append(choice)
                indices["response_cardinality"][len(response_set)].append(choice)
                indices["trial_mode"][(trial, mode)].append(choice)
                indices["condition_mode"][(condition, mode)].append(choice)
                indices["mode"][mode].append(choice)
                all_choices.append(choice)
        for target in sessions:
            target_replacements: list[str] = []
            for trial, condition in enumerate(target.condition_keys):
                target_response_set, target_open_response = _effective_choice_space(
                    target,
                    trial,
                    condition_values,
                    condition_spaces,
                    trial_values,
                    trial_spaces,
                )
                target_response_set = (
                    ()
                    if target_open_response
                    else target_response_set
                    if target.available_response_sets[trial]
                    else expanded_fixed_sets.get(target_response_set, target_response_set)
                )
                mode = (
                    f"open:{_response_signature(target.responses[trial])}"
                    if target_open_response
                    else f"fixed:{_response_signature(target.responses[trial])}"
                )
                search = (
                    (
                        "experiment_trial_condition_response_set",
                        indices["trial_condition_set"][(trial, condition, target_response_set)],
                    ),
                    (
                        "experiment_trial_condition",
                        indices["trial_condition"][(trial, condition)],
                    ),
                    (
                        "experiment_trial_response_set",
                        indices["trial_set"][(trial, target_response_set)],
                    ),
                    (
                        "experiment_trial_response_cardinality",
                        indices["trial_cardinality"][(trial, len(target_response_set))],
                    ),
                    ("experiment_trial_response_mode", indices["trial_mode"][(trial, mode)]),
                    ("experiment_trial_fallback", indices["trial"][trial]),
                    (
                        "experiment_condition_response_set",
                        indices["condition_set"][(condition, target_response_set)],
                    ),
                    (
                        "experiment_condition_response_cardinality",
                        indices["condition_cardinality"][(condition, len(target_response_set))],
                    ),
                    (
                        "experiment_condition_response_mode",
                        indices["condition_mode"][(condition, mode)],
                    ),
                    ("experiment_condition", indices["condition"][condition]),
                    (
                        "experiment_response_set",
                        indices["response_set"][target_response_set],
                    ),
                    (
                        "experiment_response_cardinality",
                        indices["response_cardinality"][len(target_response_set)],
                    ),
                    ("experiment_response_mode", indices["mode"][mode]),
                    ("experiment_fallback", all_choices),
                )
                selected: ChoiceDonor | None = None
                mapped_response = ""
                selected_level = ""
                for level, candidates in search:
                    match = _select_candidate(
                        candidates,
                        target,
                        target_response_set,
                        target_open_response,
                        trial,
                        level,
                    )
                    if match is not None:
                        selected, mapped_response = match
                        selected_level = level
                        break
                if selected is None:
                    raise RuntimeError(
                        "No valid different-participant donor response for "
                        f"line {target.line}, experiment {experiment}, trial {trial}"
                    )
                same_participant += int(selected.participant == target.participant)
                invalid_response += int(
                    not target_open_response and mapped_response not in target_response_set
                )
                levels["donor_label_remapped"] += int(mapped_response != selected.response)
                levels["open_response_direct_donor"] += int(target_open_response)
                target_replacements.append(mapped_response)
                levels[selected_level] += 1
            replacements[target.line] = tuple(target_replacements)
    if same_participant:
        raise RuntimeError(f"Found {same_participant} same-participant donor choices")
    if invalid_response:
        raise RuntimeError(f"Found {invalid_response} donor responses outside target response sets")
    return replacements, levels


def prepare(overwrite: bool) -> None:
    if OUTPUT.exists() and MANIFEST.exists() and not overwrite:
        print(MANIFEST.read_text())
        return
    records = index_source()
    replacements, levels = donor_replacements(records)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    temporary = OUTPUT.with_suffix(".jsonl.tmp")
    source_structure = hashlib.sha256()
    output_structure = hashlib.sha256()
    with SOURCE.open(encoding="utf-8") as source, temporary.open("w", encoding="utf-8") as out:
        for line_number, line in enumerate(source):
            payload = json.loads(line)
            source_structure.update(_masked_payload(payload))
            shuffled_payload = dict(payload)
            shuffled_payload["text"] = replace_marked_responses(
                payload["text"], list(replacements[line_number])
            )
            output_structure.update(_masked_payload(shuffled_payload))
            out.write(
                json.dumps(shuffled_payload, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    if source_structure.digest() != output_structure.digest():
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Shuffling changed prompt content outside marked responses")
    temporary.replace(OUTPUT)
    manifest = {
        "schema_version": 1,
        "profile": "original",
        "seed": SEED,
        "sessions": len(records),
        "choices": int(sum(len(record.responses) for record in records)),
        "donor_match_levels": dict(sorted(levels.items())),
        "same_participant_donor_choices": 0,
        "donor_responses_outside_target_set": 0,
        "non_response_content_unchanged": True,
        "masked_payload_sha256": source_structure.hexdigest(),
        "source_sha256": _sha256(SOURCE),
        "output_sha256": _sha256(OUTPUT),
    }
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, sort_keys=True))


def _last_checkpoint(output: Path) -> str | None:
    checkpoints = sorted(
        (
            path
            for path in output.glob("checkpoint-*")
            if path.is_dir() and path.name.split("-")[-1].isdigit()
        ),
        key=lambda path: int(path.name.split("-")[-1]),
    )
    return str(checkpoints[-1]) if checkpoints else None


def preflight() -> None:
    import torch
    from transformers import AutoTokenizer
    from unsloth import UnslothTrainingArguments, is_bfloat16_supported

    from cognitive_manifold.models.completion_collator import MarkedResponseCollator

    config = yaml.safe_load(CONFIG.read_text())["models"]["independent_finetune"]
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, local_files_only=True)
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    left = tokenizer(" <<").input_ids[1:]
    right = tokenizer(">>").input_ids[1:]
    collator = MarkedResponseCollator(
        response_template=left,
        instruction_template=right,
        tokenizer=tokenizer,
    )
    sample = tokenizer("Task one. You press <<A>>. Task two. You press <<B>>.")
    batch = collator([sample])
    kept = [
        int(token)
        for token, label in zip(batch["input_ids"][0], batch["labels"][0], strict=True)
        if int(label) != -100
    ]
    if tokenizer.decode(kept) != "AB":
        raise RuntimeError("Completion collator did not isolate the two marked responses")
    arguments = UnslothTrainingArguments(
        per_device_train_batch_size=int(config["batch_size_per_device"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        warmup_steps=int(config["warmup_steps"]),
        num_train_epochs=int(config["epochs"]),
        learning_rate=float(config["learning_rate"]),
        embedding_learning_rate=float(config["learning_rate"])
        * float(config["embedding_learning_rate_ratio"]),
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_strategy="steps",
        logging_steps=1,
        eval_strategy="no",
        save_strategy="steps",
        save_steps=int(config["save_steps"]),
        save_total_limit=3,
        optim=str(config["optimizer"]),
        weight_decay=float(config["weight_decay"]),
        lr_scheduler_type=str(config["scheduler"]),
        seed=SEED,
        output_dir=str(TRAINING_ROOT / "preflight"),
        report_to="none",
    )
    payload = {
        "schema_version": 1,
        "profile": "original",
        "status": "passed",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "packages": {
            name: importlib.metadata.version(name)
            for name in ("unsloth", "torch", "transformers", "trl", "bitsandbytes")
        },
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "bf16": bool(arguments.bf16),
        "collator_kept_text": "AB",
        "response_template_ids": left,
        "instruction_template_ids": right,
        "eval_strategy": str(arguments.eval_strategy),
        "source_exists": SOURCE.is_file(),
        "shuffled_source_exists": OUTPUT.is_file(),
    }
    if not payload["source_exists"] or not payload["shuffled_source_exists"]:
        raise RuntimeError("O8 intact or shuffled training source is missing")
    ENVIRONMENT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    ENVIRONMENT_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, sort_keys=True))


def train(kind: str) -> None:
    if not ENVIRONMENT_MANIFEST.exists():
        raise RuntimeError("Run the O8 training preflight before training")
    if json.loads(ENVIRONMENT_MANIFEST.read_text()).get("status") != "passed":
        raise RuntimeError("The O8 training preflight did not pass")
    try:
        from datasets import load_dataset
        from transformers import set_seed
        from unsloth import (
            FastLanguageModel,
            UnslothTrainer,
            UnslothTrainingArguments,
            is_bfloat16_supported,
        )

        from cognitive_manifold.models.completion_collator import MarkedResponseCollator
    except ImportError as error:
        raise RuntimeError(
            "O8 training dependencies are missing; install the frozen Unsloth/TRL stack first"
        ) from error
    source = SOURCE if kind == "intact" else OUTPUT
    if kind == "shuffled" and not source.exists():
        raise RuntimeError("Run the response-shuffled prepare action before training")
    config = yaml.safe_load(CONFIG.read_text())["models"]["independent_finetune"]
    if int(config["seed"]) != SEED:
        raise RuntimeError("Training seed differs from the frozen publisher-reproduction seed")
    set_seed(SEED)
    dataset = load_dataset("json", data_files={"train": str(source)})["train"].shuffle(seed=SEED)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(BASE_MODEL),
        max_seq_length=int(config["max_sequence_length"]),
        dtype=None,
        load_in_4bit=bool(config["load_in_4bit"]),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=int(config["lora_rank"]),
        target_modules=list(config["target_modules"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=bool(config["use_rslora"]),
        loftq_config=None,
    )
    tokenizer.pad_token_id = 0
    tokenizer.padding_side = "right"
    left = tokenizer(" <<").input_ids[1:]
    right = tokenizer(">>").input_ids[1:]
    collator = MarkedResponseCollator(
        response_template=left,
        instruction_template=right,
        tokenizer=tokenizer,
    )
    output = TRAINING_ROOT / kind
    output.mkdir(parents=True, exist_ok=True)
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=int(config["max_sequence_length"]),
        dataset_num_proc=8,
        data_collator=collator,
        args=UnslothTrainingArguments(
            per_device_train_batch_size=int(config["batch_size_per_device"]),
            gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
            warmup_steps=int(config["warmup_steps"]),
            num_train_epochs=int(config["epochs"]),
            learning_rate=float(config["learning_rate"]),
            embedding_learning_rate=float(config["learning_rate"])
            * float(config["embedding_learning_rate_ratio"]),
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_strategy="steps",
            logging_steps=1,
            eval_strategy="no",
            save_strategy="steps",
            save_steps=int(config["save_steps"]),
            save_total_limit=3,
            optim=str(config["optimizer"]),
            weight_decay=float(config["weight_decay"]),
            lr_scheduler_type=str(config["scheduler"]),
            seed=SEED,
            output_dir=str(output),
            report_to="none",
        ),
    )
    checkpoint = _last_checkpoint(output)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model()
    final = FINAL_MODELS[kind]
    if final.exists():
        raise RuntimeError(
            f"Final model path already exists: {final}; preserve it and use a new versioned path"
        )
    temporary = final.with_name(final.name + ".partial")
    if temporary.exists():
        raise RuntimeError(f"Incomplete merge directory already exists: {temporary}")
    model.save_pretrained_merged(
        str(temporary),
        tokenizer,
        save_method="merged_16bit",
    )
    temporary.replace(final)
    payload = json.loads(TRAINING_MANIFEST.read_text()) if TRAINING_MANIFEST.exists() else {}
    payload[kind] = {
        "profile": "original",
        "source_sha256": _sha256(source),
        "source_sessions": len(dataset),
        "publisher_training_revision": "f8038fb57e052614a1d0aae3e58bf5cca93da883",
        "resume_from_checkpoint": checkpoint,
        "global_step": int(result.global_step),
        "training_loss": float(result.training_loss),
        "final_model": str(final),
        "files": {
            str(path.relative_to(final)): _sha256(path)
            for path in sorted(final.rglob("*"))
            if path.is_file()
        },
    }
    TRAINING_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    TRAINING_MANIFEST.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    if not any(final.glob("*.safetensors")):
        raise RuntimeError("Merged control model has no safetensors weights")
    print(json.dumps(payload[kind], sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "preflight", "train"))
    parser.add_argument("--kind", choices=("intact", "shuffled"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.action == "prepare":
        prepare(args.overwrite)
    elif args.action == "preflight":
        if args.kind is not None or args.overwrite:
            raise RuntimeError("The preflight action does not accept --kind or --overwrite")
        preflight()
    else:
        if args.kind is None:
            raise RuntimeError("The train action requires --kind intact or --kind shuffled")
        if args.overwrite:
            raise RuntimeError("Training outputs are immutable; --overwrite is not permitted")
        train(args.kind)


if __name__ == "__main__":
    main()
