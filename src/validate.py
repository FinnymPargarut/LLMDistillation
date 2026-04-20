import ast
import operator
import re
from dataclasses import dataclass


VALID_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}

VALID_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

@dataclass
class ValidationResult:
    ok: bool
    reason: str = ""
    result: int | None = None


def _extract_nums_from_ast(expr: ast.Expression) -> list[int]:
    nums = []
    for node in ast.walk(expr):
        if isinstance(node, ast.Constant) and isinstance(node.value, (float, int)):
            if isinstance(node.value, float) and not node.value.is_integer():
                raise TypeError(f"Invalid constant type (float) {node.value}")
            nums.append(node.value)
    return nums


def _safe_eval(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if not isinstance(node.value, (int, float)):
            raise TypeError(f"Invalid constant type {node.value}")
        return node.value
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        left_num = _safe_eval(node.left)
        right_num = _safe_eval(node.right)
        if op_type not in VALID_BINARY_OPERATORS:
            raise TypeError(f"Unsupported operator {op_type}")
        if op_type == ast.Div and right_num == 0:
            raise ValueError("Division by zero")
        return VALID_BINARY_OPERATORS[op_type](left_num, right_num)
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in VALID_UNARY_OPERATORS:
            raise TypeError(f"Unsupported operator {op_type}")
        return VALID_UNARY_OPERATORS[op_type](_safe_eval(node.operand))
    raise TypeError(f"Unsupported ast node {type(node)}")


def validate_equation(equation: str, nums: list[int], target: int, tolerance: float = 1e-6) -> ValidationResult:
    """
    Countdown — математическая задача, в которой даны несколько чисел и целевое число.
    Нужно составить арифметическое выражение, чтобы получить целевое число.
    Правила:
    1. Разрешенные арифметические операции: +, -, *, /
    2. Каждое число из nums должно встречаться РОВНО один раз.
    3. Результат равен target

    Функция проверяет выражение equation на соответствие правилам.
    """
    if not isinstance(equation, str) or not equation.strip():
        return ValidationResult(ok=False, reason="Invalid equation")

    equation = equation.strip()
    try:
        parsed_equation = ast.parse(equation, mode="eval")
    except SyntaxError as e:
        return ValidationResult(ok=False, reason=f"SyntaxError={e}")
    # print(parsed_equation)

    try:
        equation_nums = _extract_nums_from_ast(parsed_equation)
        res = _safe_eval(parsed_equation)
    except (ValueError, TypeError) as e:
        return ValidationResult(ok=False, reason=f"While ast processing raised Error={e}")

    if sorted(equation_nums) != sorted(nums):
        return ValidationResult(ok=False, reason="Numbers in equation do not match nums list")

    if abs(res - target) > tolerance:
        return ValidationResult(ok=False, reason=f"Equation {equation} != {target}", result=res)

    return ValidationResult(ok=True, reason=f"All good", result=res)


def _strip_think_block(text: str) -> str:
    idx = text.rfind("</think>")
    if idx == -1:
        return text
    return text[idx + len("</think>"):]


def extract_equation_from_llm_response(text: str) -> str | None:
    """
    Вытаскивает уравнение из ответа модели.
    Стратегия:
    1. Сначала отсекаем всё внутри thinking-блока (до последнего </think>).
    2. Ищем <answer>...</answer>.
    3. Фоллбэки: "Answer: ...", последняя строка, весь текст.
    4. Возвращаем первого кандидата, который валидно парсится как ast-выражение.
    """
    if not text or not text.strip():
        return None
    text = text.strip()
    post_think = _strip_think_block(text).strip()

    search_space = post_think if post_think else text

    patterns = [
        r"<(?:answer|Answer)>\s*(.*?)\s*</(?:answer|Answer)>",
        r"(?:answer|Answer)\s*:\s*([^\n]+)",
    ]

    candidates = []
    for pattern in patterns:
        for m in re.finditer(pattern, search_space, flags=re.DOTALL):
            candidates.append(m[1])

    lines = [l.strip() for l in search_space.split("\n") if l.strip()]
    if lines:
        candidates.append(lines[-1])
    candidates.append(search_space)

    for cand in candidates:
        cand = cand.strip()
        if '=' in cand:
            cand = cand.split('=')[0].strip()
        if not re.search(r"\d", cand) or not re.search(r"[*/+\-]", cand):
            continue
        try:
            ast.parse(cand, mode="eval")
            return cand
        except SyntaxError:
            continue
    return None


if __name__ == '__main__':
    print(validate_equation( '90 - 80 + 75 - 24', [75, 80, 90, 24], 61))

    print(extract_equation_from_llm_response("I need to find a way to use 93 directly.\n\nThe correct combination is:\n93 - (78 - 46) = 93 - 32 = 61\n\nThis uses each number exactly once and the result is 61.\n</think>\n<answer>\n93 - (78 - 46) = 61\n</answer>"))
