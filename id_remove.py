import re

with open("combined.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Remove SNT IDs
text = re.sub(
    r"\bSNT\.\d+\.\d+\b",
    "",
    text,
    flags=re.IGNORECASE
)

# Clean spaces/newlines
text = re.sub(r"\s+", " ", text)

# Fix punctuation spacing
text = re.sub(r"\s+([.,!?])", r"\1", text)

# Split sentences
sentences = re.split(
    r'(?<=[.!?])\s+',
    text
)

# Remove empty lines
sentences = [s.strip() for s in sentences if s.strip()]

with open("sentences_output.txt", "w", encoding="utf-8") as file:
    for sentence in sentences:
        file.write(sentence + "\n")

print(f"Finished: {len(sentences)} sentences")