"""Reward function for verl.

Wire in with:
    custom_reward_function.path=<path>/adaptsoft/reward.py
    custom_reward_function.name=compute_score
"""
from __future__ import annotations

import json
import re
from typing import Iterable, Optional

FORMAT_REWARD_WEIGHT = 0.2
ACCURACY_REWARD_WEIGHT = 1.0

_FORMAT_PATTERN = re.compile(r"^\s*.*?</think>\s*[A-D]\s*$", re.IGNORECASE | re.DOTALL)


def _normalize_answer(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).rstrip(".").lower()


def _infer_option_letter(text: str, options: Iterable[str]) -> Optional[str]:
    matches = re.findall(r"\b([A-D])\b", text or "", flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    normalized = (text or "").strip().lower()
    for i, option in enumerate(list(options or [])):
        if normalized == str(option).strip().lower():
            return chr(ord("A") + i)
    return None


def _normalize_prediction(raw_text: str, options: Iterable[str]) -> str:
    """Answer letter given after the last </think>. A generation without </think> has none."""
    if not re.search(r"</think>", raw_text or "", flags=re.IGNORECASE):
        return ""
    post = re.split(r"</think>", raw_text or "", flags=re.IGNORECASE)[-1]
    letter = _infer_option_letter(post, options)
    return letter.lower() if letter is not None else _normalize_answer(post)


def _normalize_target(answer: str, options: Iterable[str]) -> str:
    if answer is None:
        return ""
    letter = _infer_option_letter(answer, options)
    return letter.lower() if letter is not None else _normalize_answer(answer)


def format_score(completion: str) -> float:
    """FORMAT_REWARD_WEIGHT if </think> occurs exactly once and the tail is a single A-D letter."""
    content = (completion or "").strip()
    if not re.search(r"</think>", content, re.IGNORECASE):
        return 0.0
    if _FORMAT_PATTERN.fullmatch(content) is None:
        return 0.0
    if len(re.findall(r"</think>", content, flags=re.IGNORECASE)) != 1:
        return 0.0
    tail = re.split(r"</think>", content, flags=re.IGNORECASE, maxsplit=1)[-1].strip()
    if re.fullmatch(r"[A-D]", tail, flags=re.IGNORECASE) is None:
        return 0.0
    return float(FORMAT_REWARD_WEIGHT)


def accuracy_score(completion: str, answer: str, options: Iterable[str]) -> float:
    """ACCURACY_REWARD_WEIGHT if the predicted option matches the gold option.

    A letter different from the gold letter but mapping to the same option text is credited,
    which covers questions whose option list repeats a choice.
    """
    options = list(options or [])
    pred = _normalize_prediction(completion or "", options)
    gold = _normalize_target(answer, options)
    if not (pred and gold):
        return 0.0
    if pred == gold:
        return float(ACCURACY_REWARD_WEIGHT)

    def _opt_text(letter):
        if letter and len(letter) == 1 and "a" <= letter <= "d":
            i = ord(letter) - ord("a")
            if 0 <= i < len(options):
                return _normalize_answer(str(options[i]))
        return None

    pt, gt = _opt_text(pred), _opt_text(gold)
    if pt is not None and pt == gt:
        return float(ACCURACY_REWARD_WEIGHT)
    return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """verl calls this per sample. Options are stored as a JSON string in extra_info."""
    ei = extra_info or {}
    opts = ei.get("options", [])
    if isinstance(opts, str):
        try:
            opts = json.loads(opts)
        except Exception:
            opts = []
    fmt = format_score(solution_str)
    acc = accuracy_score(solution_str, ground_truth, opts)
    return {"score": fmt + acc, "format": fmt, "accuracy": acc}
