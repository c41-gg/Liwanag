import subprocess
import tempfile
import os


class FSPOSTTagger:
    """FSPOST POS tagger (primary tagger for Tagalog)."""
 
    def __init__(self, jar_path: str, model_path: str):
        self.jar_path = jar_path
        self.model_path = model_path
 
        if not os.path.exists(jar_path):
            raise FileNotFoundError(f"JAR file not found: {jar_path}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
 
    def tag(self, tokens: list[str]) -> list[tuple]:
        """
        Tag tokens with FSPOST (returns MGNN tags).
 
        Args:
            tokens: list of word tokens
 
        Returns:
            list of (word, mgnn_tag) tuples
        """
        if not tokens:
            return []
 
        sentence = ' '.join(tokens)
 
        try:
            with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.txt') as temp_file:
                temp_file.write(sentence)
                temp_file_path = temp_file.name
 
            command = [
                'java', '-mx1g',
                '-cp', self.jar_path,
                'edu.stanford.nlp.tagger.maxent.MaxentTagger',
                '-model', self.model_path,
                '-textFile', temp_file_path
            ]
 
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                encoding='utf-8'
            )
            output, error = process.communicate()
 
            os.unlink(temp_file_path)
 
            if process.returncode != 0:
                raise Exception(f"POS tagging failed: {error}")
 
            # Parse FSPOST output: word|MGNN_TAG format
            tagged_output = output.strip().split()
            tagged_tokens = []
 
            for tag in tagged_output:
                if '|' in tag:
                    word, mgnn_tag = tag.rsplit('|', 1)
                    tagged_tokens.append((word, mgnn_tag))
 
            return tagged_tokens
 
        except Exception as e:
            print(f"Error during FSPOST tagging: {e}")
            return []