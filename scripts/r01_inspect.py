"""Read bounded text sections or selected JSON fields without dumping artifacts."""
from argparse import ArgumentParser
import json
from pathlib import Path


def select(data, field):
    keys = [key.replace("~1", "/").replace("~0", "~") for key in field[1:].split("/")] if field.startswith("/") else field.split(".")
    for key in keys:
        if key == "@keys":
            data = list(data.keys()) if isinstance(data, dict) else list(range(len(data))) if isinstance(data, list) else None
            if data is None:
                raise ValueError("@keys requires a JSON object or array")
        else:
            data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def inspect(path, *, fields=(), section=None, start=1, lines=40, chars=6000):
    if start < 1 or lines < 1 or chars < 1 or lines > 100 or chars > 12000:
        raise ValueError("Use 1–100 lines and 1–12000 characters; defaults 40/6000")
    path = Path(path)
    if path.suffix in (".json", ".jsonl"):
        if path.suffix == ".jsonl" or not fields:
            raise ValueError("For JSON use --field; inspect JSONL through its summary/report")
        data = json.loads(path.read_text(encoding="utf-8"))
        text = json.dumps({field: select(data, field) for field in fields},
                          ensure_ascii=False, indent=2)
    else:
        selected, active, level = [], section is None, None
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if section is not None and not active:
                    if line.strip() != section:
                        continue
                    active = True
                    level = len(line) - len(line.lstrip("#"))
                    if not level:
                        raise ValueError("--section must name a Markdown heading including #")
                elif section is not None and line.startswith("#"):
                    next_level = len(line) - len(line.lstrip("#"))
                    if next_level <= level:
                        break
                if active and (section is not None or number >= start):
                    selected.append(f"{number}: {line.rstrip()}")
                    if len(selected) >= lines or sum(len(item) + 1 for item in selected) >= chars:
                        break
        if section is not None and not active:
            raise ValueError(f"Section not found: {section}")
        text = "\n".join(selected)
    if len(text) > chars:
        text = text[:chars] + "\n[output capped; select fewer fields or a smaller section]"
    return text


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--field", action="append", default=[], help="Dotted field or /JSON/pointer; @keys lists object keys/array indices")
    parser.add_argument("--section", help="Exact Markdown heading, including #")
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--lines", type=int, default=40)
    parser.add_argument("--chars", type=int, default=6000)
    args = parser.parse_args()
    try:
        print(inspect(args.path, fields=args.field, section=args.section, start=args.start, lines=args.lines, chars=args.chars))
    except (KeyError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
