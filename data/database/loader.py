import json
from pathlib import Path

GLOSSARY_PATH = Path(__file__).parent / "glossary.json"


def load_glossary() -> dict:
    with open(GLOSSARY_PATH, "r") as f:
        return json.load(f)


def build_glossary_prompt(glossary: dict) -> str:
    """Convert glossary dict into a system prompt injection string."""
    lines = ["## Business Glossary", "Use the following definitions when generating SQL or interpreting user questions:\n"]

    lines.append("### Metrics")
    for name, info in glossary["metrics"].items():
        lines.append(f"- **{name}**: {info['definition']}")
        lines.append(f"  SQL: `{info['sql_formula']}`")
        lines.append(f"  Note: {info['notes']}")

    lines.append("\n### Time Periods")
    for name, definition in glossary["time_periods"].items():
        lines.append(f"- **{name}**: {definition}")

    lines.append("\n### Campaign Types")
    for name, definition in glossary["campaign_types"].items():
        lines.append(f"- **{name}**: {definition}")

    lines.append("\n### User Segments")
    for name, definition in glossary["user_segments"].items():
        lines.append(f"- **{name}**: {definition}")

    return "\n".join(lines)


def get_glossary_prompt() -> str:
    return build_glossary_prompt(load_glossary())
