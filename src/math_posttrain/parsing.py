"""Strict, bounded numeric parsing. Units, percentages and exponents are invalid."""
import re
from fractions import Fraction

LIMIT = 32768
NUMBER = r'[+-]?(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)'
MARKER = re.compile(r'Final Answer:|####|\\boxed')


def parse_number(text):
    text = text.strip()
    if len(text) > 256:
        return None
    text = re.sub(r'^([+-]?)[\$£€]\s*', r'\1', text)
    if not re.fullmatch(NUMBER + r'(?:\s*/\s*' + NUMBER + r')?', text, re.ASCII):
        return None
    try:
        parts = text.replace(',', '').split('/')
        value = Fraction(parts[0].strip())
        return value if len(parts) == 1 else value / Fraction(parts[1].strip())
    except (ValueError, ZeroDivisionError):
        return None


def extract_gold(text):
    if not isinstance(text, str) or len(text) > LIMIT or text.count('####') != 1:
        raise ValueError('gold must contain exactly one #### marker')
    value = parse_number(text.split('####')[1])
    if value is None:
        raise ValueError('invalid gold number')
    return value


def prediction_status(completion):
    if not isinstance(completion, str) or len(completion) > LIMIT:
        return None, 'invalid_input'
    markers = list(MARKER.finditer(completion))
    if len(markers) != 1:
        return None, 'ambiguous' if markers else 'missing_marker'
    marker = markers[0]
    suffix = completion[marker.end():].strip()
    if marker.group() == '\\boxed':
        match = re.fullmatch(r'\{([^{}]+)\}', suffix)
        suffix = match[1] if match else ''
    value = parse_number(suffix)
    return (value, None) if value is not None else (None, 'invalid_number_or_trailing_text')


def extract_prediction(completion):
    return prediction_status(completion)[0]


def is_correct(pred, gold):
    return pred is not None and gold is not None and pred == gold


def format_ok(completion):
    if not isinstance(completion, str) or len(completion) > LIMIT:
        return False
    lines = completion.rstrip().splitlines()
    return bool(lines and lines[-1].startswith('Final Answer:')
                and len(MARKER.findall(completion)) == 1
                and parse_number(lines[-1][len('Final Answer:'):]) is not None)
