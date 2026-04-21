import re
from typing import Optional


_ANSWER_PATTERNS = [
    r"#### (-?\d[\d,]*\.?\d*)",       # GSM8K gold format
    r"The answer is[:\s]+(-?\d[\d,]*\.?\d*)",
    r"Answer[:\s]+(-?\d[\d,]*\.?\d*)",
    r"=\s*(-?\d[\d,]*\.?\d*)\s*$",    # trailing equation result
]


def extract_answer(text: str) -> Optional[str]:
    """
    Extract numeric answer from model output or gold solution.
    Returns the raw string (with commas) or None.
    """
    for pattern in _ANSWER_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")
    # Fallback: last number in text
    numbers = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return None


def _normalize(s: str) -> str:
    s = s.strip().lstrip("0") or "0"
    try:
        return str(float(s))
    except ValueError:
        return s


def gsm8k_reward_fn(responses: list[str], gold_answers: list[str]) -> list[float]:
    """
    Binary reward: +1.0 if predicted answer matches gold, else 0.0.
    Also applies a small length penalty to discourage verbose padding.
    """
    rewards = []
    for response, gold in zip(responses, gold_answers):
        pred = extract_answer(response)
        gold_clean = extract_answer(gold) or gold.strip()

        if pred is not None and _normalize(pred) == _normalize(gold_clean):
            base = 1.0
        else:
            base = 0.0

        # Small length penalty: deduct up to 0.05 for outputs > 512 chars
        length_penalty = min(0.05, max(0.0, (len(response) - 512) / 10000))
        rewards.append(base - length_penalty)

    return rewards


def format_gsm8k_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "End your response with '#### <answer>' where <answer> is the numeric answer.\n\n"
        f"Problem: {question}\n\nSolution:"
    )
