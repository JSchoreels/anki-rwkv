"""Rebase generated report links without changing the linked evidence."""

import os
from pathlib import Path
import re


def rebase_links(text, source_directory, target_directory):
    source_directory, target_directory = Path(source_directory), Path(target_directory)

    def replace(match):
        href = match.group(1)
        if href.startswith("#") or re.match(r"^[a-zA-Z][a-zA-Z0-9+.-]*:", href):
            return match.group(0)
        filename, separator, fragment = href.partition("#")
        destination = os.path.abspath(source_directory / filename)
        relative = Path(os.path.relpath(destination, target_directory)).as_posix()
        return "](" + relative + (separator + fragment if separator else "") + ")"

    return re.sub(r"\]\(([^)\n]+)\)", replace, text)
