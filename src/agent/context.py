"""The agent's reference documents: the policy wording, the claims manual and the rules in the context folder, and the tools to search them."""

import re
from pathlib import Path

from langchain_core.tools import tool

CONTEXT = Path(__file__).resolve().parents[2] / "context"
SEARCH_RESULTS = 5
STOPWORDS = {"the", "and", "for", "that", "with", "are", "was", "from", "this", "what", "when", "how", "does", "can", "any", "not"}


def load_context(folder: Path = CONTEXT) -> dict[str, str]:
    """file name -> text, for every document in the context folder."""
    return {path.name: path.read_text(encoding="utf-8") for path in sorted(folder.iterdir()) if path.suffix in {".md", ".yaml"}}


def _chunks(name: str, text: str) -> list[dict]:
    """Split a document into its paragraphs, tables and yaml entries, each tagged with the markdown heading it sits under."""
    heading, chunks = "", []
    for block in re.split(r"\n\s*\n", text):
        block = block.strip()
        if not block or block == "---":
            continue
        if block.startswith("#"):
            heading = block.splitlines()[0].lstrip("# ").strip()
        chunks.append({"file": name, "heading": heading, "text": block})
    return chunks


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9$]+", text.lower()) if len(w) > 2 and w not in STOPWORDS}


def make_context_tools(documents: dict[str, str]) -> list:
    """list_context_files, read_context_file and search_context over these documents."""
    chunks = [chunk for name, text in documents.items() for chunk in _chunks(name, text)]

    @tool
    def list_context_files() -> str:
        """List the reference documents: the policy wording, the claims handling manual, the claim rules and so on."""
        lines = []
        for name, text in documents.items():
            title = next((line.lstrip("# ") for line in text.splitlines() if line.startswith("# ")), name)
            lines.append(f"{name}: {title} ({len(text):,} characters)")
        return "\n".join(lines)

    @tool
    def read_context_file(name: str, section: str = "") -> str:
        """Read a reference document, or one section of it.

        Args:
            name: The file name, as list_context_files shows it, for example claims_manual.md
            section: Part of a section heading, for example "Interpretation notes". Leave empty to read the whole file
        """
        if name not in documents:
            return f"There is no file {name}. The files are: {', '.join(documents)}"
        if not section:
            return documents[name]
        parts = re.split(r"(?m)^(?=## )", documents[name])
        matches = [part for part in parts if part.startswith("## ") and section.lower() in part.splitlines()[0].lower()]
        return "\n".join(matches) if matches else f"No section of {name} has a heading containing {section!r}"

    @tool
    def search_context(query: str) -> str:
        """Search all the reference documents for passages about a topic, such as a benefit, an exclusion, a document requirement or a rule.

        Args:
            query: What to look for, in a few keywords, for example "unattended belongings vehicle" or "police report theft"
        """
        wanted = _words(query)
        scored = []
        for chunk in chunks:
            hits = len(wanted & _words(f"{chunk['heading']} {chunk['text']}"))
            if hits:
                scored.append((hits, chunk))
        scored.sort(key=lambda pair: -pair[0])
        if not scored:
            return "Nothing found. Try other keywords, or list_context_files and read a file."
        return "\n\n".join(f"[{c['file']} > {c['heading']}]\n{c['text']}" for _, c in scored[:SEARCH_RESULTS])

    return [list_context_files, read_context_file, search_context]
