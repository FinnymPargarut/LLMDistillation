import pandas as pd

from src.validate import validate_equation


FALLBACK_EQUATION = "0"


def _equation_or_fallback(eq: str | None) -> str:
    if eq is None:
        return FALLBACK_EQUATION
    eq = eq.strip()
    return eq if eq else FALLBACK_EQUATION


def build_submission_df(results: list[dict],
                        sample_submission_path: str) -> tuple[pd.DataFrame, dict]:
    """
    results: вывод generate_student_responses с 'id', 'equation', 'nums', 'target'
    sample_submission_path: путь к sample_submission.csv (определяет схему).

    Returns:
        (submission_df, stats) где stats содержит:
        - n_total, n_extracted, n_fallback, n_valid_by_local_validator,
          expected_accuracy (по нашему валидатору).
    """
    sample = pd.read_csv(sample_submission_path)

    required_cols = {"id", "equation"}
    if not required_cols.issubset(set(sample.columns)):
        raise ValueError(
            f"sample_submission должен содержать колонки {required_cols}, "
            f"а содержит {list(sample.columns)}"
        )

    if not all("id" in r for r in results):
        raise ValueError(
            "В results нет 'id'. Убедись что test_public.csv загружается с 'id'."
        )

    results_by_id = {r["id"]: r for r in results}

    submission = sample.copy()
    equations = []
    fallback_count = 0
    extracted_count = 0
    for sid in submission["id"]:
        if sid not in results_by_id:
            equations.append(FALLBACK_EQUATION)
            fallback_count += 1
            continue
        raw_eq = results_by_id[sid]["equation"]
        if raw_eq is None:
            fallback_count += 1
        else:
            extracted_count += 1
        equations.append(_equation_or_fallback(raw_eq))

    submission["equation"] = equations

    valid_by_local = 0
    for sid, eq in zip(submission["id"], equations):
        if sid not in results_by_id:
            continue
        r = results_by_id[sid]
        vr = validate_equation(eq, r["nums"], r["target"])
        if vr.ok:
            valid_by_local += 1

    n_total = len(submission)
    stats = {
        "n_total": n_total,
        "n_extracted": extracted_count,
        "n_fallback": fallback_count,
        "extracted_rate": extracted_count / n_total if n_total else 0.0,
        "n_valid_by_local_validator": valid_by_local,
        "expected_accuracy": valid_by_local / n_total if n_total else 0.0,
    }

    return submission, stats