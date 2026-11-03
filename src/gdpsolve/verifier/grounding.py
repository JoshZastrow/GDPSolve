"""check_field(): rule-based grounding for extracted and computed fields (docs/verifier.md §2)."""

from __future__ import annotations

import ast
import math
import operator
import re

from gdpsolve.schemas import Document, ExtractedField, FieldCheck

_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
_NUMBER = re.compile(r"\(?-?\$?\d[\d,]*(?:\.\d+)?%?\)?")


def evaluate(formula: str, names: dict[str, float]) -> float:
    """Arithmetic over span ids: numbers, + - * /, unary minus, sum(...). Nothing else."""

    def ev(node: ast.AST) -> float:
        match node:
            case ast.Expression(body=body):
                return ev(body)
            case ast.Constant(value=v) if isinstance(v, int | float) and not isinstance(v, bool):
                return float(v)
            case ast.Name(id=name) if name in names:
                return names[name]
            case ast.BinOp(left=left, op=op, right=right) if type(op) in _BINOPS:
                return _BINOPS[type(op)](ev(left), ev(right))
            case ast.UnaryOp(op=ast.USub(), operand=x):
                return -ev(x)
            case ast.Call(func=ast.Name(id="sum"), args=args, keywords=[]):
                return sum(ev(a) for a in args)
        raise ValueError(f"not allowed in a formula: {ast.dump(node)[:60]}")

    return ev(ast.parse(formula, mode="eval"))


def parse_number(text: str) -> float | None:
    """First number in text; "(1,204)" -> -1204, "$1,240" -> 1240, "12.5%" -> 12.5."""
    m = _NUMBER.search(text)
    if not m:
        return None
    s = m.group()
    negative = (s.startswith("(") and s.endswith(")")) or "-" in s
    digits = re.sub(r"[^\d.]", "", s)
    return -float(digits) if negative else float(digits)


def _norm(text: str) -> str:
    return " ".join(text.split()).casefold()


def check_field_pages(field: ExtractedField, pages: dict[int, str]) -> FieldCheck:
    """check_field against already-read page text (1-indexed page -> text)."""
    if field.kind == "summary":
        raise ValueError("summary fields are checked by support(), not check_field()")
    if not field.sources:
        return FieldCheck(field=field.name, ok=False, reason="no sources")
    for s in field.sources:
        if _norm(s.text) not in _norm(pages.get(s.page, "")):
            return FieldCheck(field=field.name, ok=False, reason=f"{s.id} not found on page {s.page}")

    if field.kind == "extracted":
        value = str(field.value)
        if any(_norm(value) in _norm(s.text) for s in field.sources):
            return FieldCheck(field=field.name, ok=True)
        expected = parse_number(value)
        if expected is not None and expected in [parse_number(s.text) for s in field.sources]:
            return FieldCheck(field=field.name, ok=True)
        return FieldCheck(field=field.name, ok=False, reason=f"value {value!r} not in cited spans")

    if not field.formula:
        return FieldCheck(field=field.name, ok=False, reason="computed field has no formula")
    names = {}
    for s in field.sources:
        n = parse_number(s.text)
        if n is None:
            return FieldCheck(field=field.name, ok=False, reason=f"{s.id} has no number")
        names[s.id] = n
    try:
        result = evaluate(field.formula, names)
    except (ValueError, ZeroDivisionError, SyntaxError) as e:
        return FieldCheck(field=field.name, ok=False, reason=str(e))
    expected = parse_number(str(field.value))
    if expected is None or not math.isclose(result, expected, rel_tol=1e-3, abs_tol=1e-6):
        return FieldCheck(field=field.name, ok=False, reason=f"formula gives {result:g}, field says {field.value}")
    return FieldCheck(field=field.name, ok=True)


def check_field(field: ExtractedField, document: Document) -> FieldCheck:
    """Re-read the cited pages from the document's text layer, then check the field."""
    import pypdfium2

    from gdpsolve.verifier import store

    pdf = pypdfium2.PdfDocument(store.get(document.pdf))
    wanted = {s.page for s in field.sources}
    pages = {p: pdf[p - 1].get_textpage().get_text_range() for p in wanted if 1 <= p <= len(pdf)}
    return check_field_pages(field, pages)
