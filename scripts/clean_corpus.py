from pathlib import Path
import re

ABBREVIATIONS = [
    "Mr.", "Mrs.", "Ms.", "Dr.", "Jr.", "Sr.", "Sra.", "Gen.", "Gov.", "Rep.", "Sen.",
    "St.", "Prof.", "Inc.", "Ltd.", "vs.", "e.g.", "i.e.", "NCRPO.", "PNP.", "ICC.", "DDS.",
    "EJKs.", "E-Gov.", "BJCul.", "Duterte.", "Marcos.", "Filipinolohiya.", "Filipino.", "Filipinas.",
]

SENTENCE_SPLIT_PATTERN = re.compile(
    r'(?<=[.!?])\s+(?=(?:["\'\u201c\u2018]*)(?:[A-ZÁÉÍÓÚÑ]))'
)


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def normalize_text(text: str) -> str:
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2013", "-").replace("\u2014", " - ")
    text = text.replace("\u00A0", " ")
    text = text.replace("\t", " ").replace("\r", "\n")
    text = re.sub(r"\bSNT\.\d+\.\d+\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[ ]+", " ", text)
    text = re.sub(r"\n+", "\n", text)
    text = text.strip()
    return text


def protect_abbreviations(text: str) -> str:
    for abbr in ABBREVIATIONS:
        protected = abbr.replace(".", "<DOT>")
        text = text.replace(abbr, protected)
    return text


def restore_abbreviations(text: str) -> str:
    return text.replace("<DOT>", ".")


def split_sentences(text: str) -> list[str]:
    text = protect_abbreviations(text)
    text = re.sub(r"\s+", " ", text)
    parts = SENTENCE_SPLIT_PATTERN.split(text)
    sentences = [restore_abbreviations(part).strip() for part in parts if part.strip()]
    return sentences


def save_sentences(sentences: list[str], output_path: Path) -> None:
    output_path.write_text("\n".join(sentences) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean text and split it into separate sentences.")
    parser.add_argument(
        "input",
        nargs="?",
        default="combined.txt",
        help="Input text file to clean (default: combined.txt)",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default="cleaned_sentences.txt",
        help="Output file for one sentence per line (default: cleaned_sentences.txt)",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    text = load_text(input_path)
    normalized = normalize_text(text)
    sentences = split_sentences(normalized)
    save_sentences(sentences, output_path)
    print(f"Saved {len(sentences)} sentences to {output_path}")


if __name__ == "__main__":
    import argparse

    main()
