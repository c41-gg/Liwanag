"""
scripts/tag_and_parse_corpus.py

Tags an ALT corpus with FSPOST and generates phrase structure trees
using Tagalog-specific constituency rules with MGNN tagset.

MGNN Tagset Reference:
  VBAF, VBCF, VBLF, VBBF = Verbs (Actor, Object, Locative, Benefactive focus)
  NNC, NNP = Nouns (Common, Proper)
  JJ, JJR, JJS = Adjectives
  RB = Adverbs
  DTC, DTP, DTPP = Determiners (Topic, Predicative, Plural topic)
  CCB, CCBG = Case markers (Non-topic, Genitive)
  IN, PRP = Prepositions, Pronouns
  MD = Modals
  PMP, PMS = Punctuation (Mark, Sentence-ending)

Input:  alt_corpus.txt (one raw sentence per line)
Output: parsed_trees.txt (one constituency tree per line in NLTK format)

Usage:
    python scripts/tag_and_parse_corpus.py --input data\processed\corpus_clean1.txt --output data/processed/parsed_trees2.txt --max 1000
"""


import argparse
import subprocess
import tempfile
import os
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from nltk import Tree
from dotenv import load_dotenv

load_dotenv()  # reads .env into os.environ

HF_TOKEN = os.environ["HF_TOKEN"] 
 
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
 
 
class XLMRFallbackTagger:
    """XLM-R fallback tagger for Foreign Words only."""
 
    # UD → MGNN mapping
    UD_TO_MGNN = {
        'VERB': 'VBAF',
        'NOUN': 'NNC',
        'PROPN': 'NNP',
        'ADJ': 'JJ',
        'ADV': 'RB',
        'DET': 'DTC',
        'ADP': 'CCB',
        'PRON': 'PRP',
        'AUX': 'VBAF',
        'PUNCT': 'PMP',
        'PART': 'RB',
        'NUM': 'NNC',
        'X': 'FW',
        'INTJ': 'RB',
    }
    
    def __init__(self, model_name: str = "xlm-roberta-large"):
        """Initialize XLM-R (lazy-loaded on first use)."""
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = None
        self.loaded = False
 
    def _lazy_load(self):
        """Load XLM-R model on first use only."""
        if self.loaded:
            return
 
        print(f"Loading XLM-R for FW fallback...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.model = self.model.to(self.device)
            self.loaded = True
            print(f"XLM-R loaded on {self.device}")
        except Exception as e:
            print(f"Warning: could not load XLM-R: {e}")
            self.loaded = False
 
    def tag_fw_token(self, word: str) -> str:
        """
        Use XLM-R to classify a single FW (Foreign Word) token.
 
        Args:
            word: the foreign word to classify
 
        Returns:
            MGNN tag (e.g., 'VBAF', 'NNC')
        """
        if not self.loaded:
            self._lazy_load()
 
        if not self.loaded:
            return 'FW'  # fallback if XLM-R failed to load
 
        try:
            inputs = self.tokenizer(word, return_tensors='pt', padding=True, truncation=True)
 
            with torch.no_grad():
                outputs = self.model(**{k: v.to(self.device) for k, v in inputs.items()})
                logits = outputs.logits
 
            prediction = torch.argmax(logits[0], dim=-1)[0]
            ud_tag = self.model.config.id2label[prediction.item()]
            mgnn_tag = self.UD_TO_MGNN.get(ud_tag, 'FW')
 
            return mgnn_tag
 
        except Exception as e:
            return 'FW'  # fallback on error
 
 
class TagalogConstituencyRuleGenerator:
    """Generates phrase structure trees using Tagalog-specific MGNN constituency rules."""
 
    # MGNN tag groupings for easier rule matching
    NOUNS = {'NNC', 'NNP', 'NNPPA', 'NNCA'}  # Common, Proper, Plural nouns
    PRONOUNS = {'PRP', 'PRS', 'PRSPP', 'PRQ', 'PRO', 'PRQP', 'PRL', 'PRC', 'PRF', 'PRI'}  # Pronouns
    DETERMINERS = {'DTC', 'DTCP','DTP', 'DTPP'}  # Topic, Predicative determiners
    CASE_MARKERS = {'CCT', 'CCR', 'CCB', 'CCA', 'CCP', 'CCU', 'LM'}  # Case markers, prepositions
    VERBS = {'VBW', 'VBS', 'VBH', 'VBN', 'VBTS', 'VBTR', 'VBTF', 'VBTP', 'VBAF', 'VBOF', 'VBOB', 'VBOL', 'VBOI', 'VBRF'}  # All verb focus types + modals
    ADJECTIVES = {'JJD', 'JJC', 'JJCC', 'JJCS', 'JJCN', 'JJN'}  # Adjectives
    ADVERBS = {'RBD', 'RBN', 'RBK', 'RBP', 'RBB', 'RBR', 'RBQ', 'RBT', 'RBF', 'RBW', 'RBM', 'RBL', 'RBI', 'RBJ', 'RBS'}  # Adverbs
    PUNCTUATION = {'PMP', 'PMS', 'PMC', 'PMQ', 'PMSC', 'PME'}  # Punctuation marks
    FOREIGN = {'FW'}  # Foreign words (to be refined by XLM-R)
 
    def generate_tree(self, tagged_tokens: list[tuple]) -> str:
        """
        Generate a constituency tree from POS-tagged tokens.
 
        Args:
            tagged_tokens: list of (word, mgnn_tag) tuples
 
        Returns:
            NLTK tree format string
        """
        if not tagged_tokens:
            return None
 
        # Remove punctuation
        filtered = [t for t in tagged_tokens if t[1] not in self.PUNCTUATION]
 
        if not filtered:
            return None
 
        # Generate tree structure
        tree = self._build_tree_from_mgnn(filtered)
 
        return str(tree) if tree else None
 
    def _build_tree_from_mgnn(self, tagged_tokens: list[tuple]) -> Tree:
        """
        Build phrase structure tree from MGNN-tagged tokens.
 
        Args:
            tagged_tokens: (word, mgnn_tag) tuples
 
        Returns:
            NLTK Tree object
        """
        if not tagged_tokens:
            return None
 
        children = [Tree(mgnn_tag, [word]) for word, mgnn_tag in tagged_tokens]
        tree_children = []
        i = 0
 
        while i < len(children):
            current_tag = children[i].label()
 
            # Rule: DET + NOUN(s) → NP
            if (i < len(children) - 1 and
                current_tag in self.DETERMINERS and
                children[i + 1].label() in self.NOUNS | self.FOREIGN):
 
                np_children = [children[i], children[i + 1]]
                i += 2
 
                while i < len(children) and children[i].label() in self.ADJECTIVES | self.NOUNS | self.CASE_MARKERS | self.FOREIGN:
                    np_children.append(children[i])
                    i += 1
 
                tree_children.append(Tree('NP', np_children))
 
            # Rule: CASE + NOUN(s) → NP
            elif (i < len(children) - 1 and
                  current_tag in self.CASE_MARKERS and
                  children[i + 1].label() in self.NOUNS | self.FOREIGN):
 
                np_children = [children[i], children[i + 1]]
                i += 2
 
                while i < len(children) and children[i].label() in self.ADJECTIVES | self.NOUNS | self.FOREIGN:
                    np_children.append(children[i])
                    i += 1
 
                tree_children.append(Tree('NP', np_children))
 
            # Rule: NOUN/FW → NP
            elif current_tag in self.NOUNS | self.FOREIGN:
                tree_children.append(Tree('NP', [children[i]]))
                i += 1
 
            # Rule: PRONOUN → NP
            elif current_tag in self.PRONOUNS:
                tree_children.append(Tree('NP', [children[i]]))
                i += 1
 
            # Rule: VERB(+ADV) → VP
            elif current_tag in self.VERBS | self.FOREIGN:
                vp_children = [children[i]]
                i += 1
 
                while i < len(children) and children[i].label() in self.ADVERBS | self.FOREIGN:
                    vp_children.append(children[i])
                    i += 1
 
                tree_children.append(Tree('VP', vp_children))
 
            # Rule: ADJ + NOUN → NP
            elif (i < len(children) - 1 and
                  current_tag in self.ADJECTIVES and
                  children[i + 1].label() in self.NOUNS | self.FOREIGN):
 
                np_children = [children[i], children[i + 1]]
                i += 2
                tree_children.append(Tree('NP', np_children))
 
            # Rule: ADV (standalone)
            elif current_tag in self.ADVERBS:
                tree_children.append(Tree('RB', [children[i]]))
                i += 1
 
            # Default
            else:
                tree_children.append(Tree('NP', [children[i]]))
                i += 1
 
        return Tree('S', tree_children)
 
 
def tokenize_sentence(sentence: str) -> list[str]:
    """Simple whitespace tokenization."""
    return sentence.strip().split()
 
 
def tag_and_parse_corpus(
    input_path: str,
    output_path: str,
    jar_path: str,
    model_path: str,
    max_sentences: int = None
):
    """
    Main function: tag and parse an ALT corpus using hybrid FSPOST + XLM-R.
 
    Args:
        input_path:     path to raw ALT corpus (one sentence per line)
        output_path:    path to save parsed trees
        jar_path:       path to stanford-postagger.jar
        model_path:     path to Tagalog model file
        max_sentences:  limit to first N sentences (None = all)
    """
    print(f"Initializing hybrid tagger (FSPOST + XLM-R fallback)...")
    fspost = FSPOSTTagger(jar_path, model_path)
    xlmr_fallback = XLMRFallbackTagger()
    rule_gen = TagalogConstituencyRuleGenerator()
 
    parsed = 0
    failed = 0
    fw_count = 0
 
    with open(input_path, encoding='utf-8', errors='replace') as f_in, \
         open(output_path, 'w', encoding='utf-8') as f_out:
 
        for i, line in enumerate(f_in):
            if max_sentences and i >= max_sentences:
                break
 
            sentence = line.strip()
            if not sentence:
                failed += 1
                continue
 
            try:
                # Tokenize
                tokens = tokenize_sentence(sentence)
 
                # POS tag with FSPOST
                tagged_tokens = fspost.tag(tokens)
 
                if not tagged_tokens:
                    failed += 1
                    continue
 
                # Refine FW tags with XLM-R
                refined_tokens = []
                for word, tag in tagged_tokens:
                    if tag == 'FW':
                        refined_tag = xlmr_fallback.tag_fw_token(word)
                        refined_tokens.append((word, refined_tag))
                        fw_count += 1
                    else:
                        refined_tokens.append((word, tag))
 
                # Generate tree
                tree_str = rule_gen.generate_tree(refined_tokens)
 
                if tree_str:
                    f_out.write(tree_str + '\n')
                    parsed += 1
                else:
                    failed += 1
 
                # Progress
                if (parsed + failed) % 500 == 0:
                    print(f"  {parsed:,} parsed, {failed:,} failed, {fw_count} FW refined...")
 
            except Exception as e:
                if parsed < 5:
                    print(f"Warning at line {i}: {e}")
                failed += 1
 
    print(f"\nDone.")
    print(f"  Parsed:        {parsed:,} sentences")
    print(f"  Failed:        {failed:,} sentences")
    print(f"  FW refined:    {fw_count} foreign words reclassified by XLM-R")
    print(f"  Output:        {output_path}")
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Hybrid POS tagger: FSPOST primary + XLM-R for Foreign Words"
    )
    parser.add_argument(
        '--input',
        default='alt_corpus.txt',
        help='Input corpus (one sentence per line)'
    )
    parser.add_argument(
        '--output',
        default='data/processed/parsed_trees.txt',
        help='Output trees (NLTK format, one per line)'
    )
    parser.add_argument(
        '--jar',
        default='Libraries/FSPOST/stanford-postagger.jar',
        help='Path to stanford-postagger.jar'
    )
    parser.add_argument(
        '--model',
        default='Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger',
        help='Path to Tagalog model file'
    )
    parser.add_argument(
        '--max',
        type=int,
        default=None,
        help='Limit to first N sentences (default: all)'
    )
 
    args = parser.parse_args()
 
    tag_and_parse_corpus(
        input_path=args.input,
        output_path=args.output,
        jar_path=args.jar,
        model_path=args.model,
        max_sentences=args.max
    )
 
