"""
Утилиты для проверки корректности выражений Countdown.
Используется и при инференсе, и при генерации датасета.
"""

import re
from typing import Optional


def parse_equation(text: str) -> Optional[str]:
    """
    Вытаскивает уравнение из ответа модели.
    Модель может болтать лишнее — ищем выражение в конце или после маркеров.
    """
    # Ищем после явных маркеров типа "Answer:", "answer is", "equation:"
    patterns = [
        r"(?:answer|equation|result)\s*[:\-=]\s*([0-9\s\+\-\*\/\(\)]+)",
        r"(?:therefore|so|thus)[,\s]+([0-9\s\+\-\*\/\(\)]+)\s*=",
        r"([0-9][0-9\s\+\-\*\/\(\)]+)\s*=\s*\d+\s*$",  # "выражение = число" в конце
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            expr = match.group(1).strip()
            expr = re.sub(r"\s+", " ", expr)
            return expr

    # Fallback: берём последнюю строку с числами и операциями
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        # Убираем "= TARGET" в конце если есть
        line = re.sub(r"\s*=\s*\d+\s*$", "", line).strip()
        if re.match(r"^[0-9\s\+\-\*\/\(\)]+$", line):
            return line

    return None


def check_equation(equation: str, nums: list[int], target: int) -> dict:
    """
    Проверяет корректность уравнения по правилам Countdown:
    1. Результат вычисления равен target
    2. Используются только числа из nums (каждое ровно 1 раз)
    3. Только операции +, -, *, /
    """

    result = {
        "is_correct": False,
        "parse_failed": equation is None,
        "eval_result": None,
        "error": None,
    }

    if equation is None:
        return result

    if not re.match(r"^[0-9\s\+\-\*\/\(\)]+$", equation):
        result["error"] = "invalid_chars"
        return result

    used_nums = [int(n) for n in re.findall(r"\d+", equation)]
    if sorted(used_nums) != sorted(nums):
        result["error"] = "wrong_numbers"
        return result

    try:
        result["eval_result"] = eval(equation)
        if abs(result["eval_result"] - target) < 1e-9:
            result["is_correct"] = True
    except ZeroDivisionError:
        result["error"] = "division_by_zero"
    except (SyntaxError, NameError):
        result["error"] = "eval_failed"

    return result
