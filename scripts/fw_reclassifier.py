from dataclasses import dataclass
from typing import Optional
from nltk.tag import PerceptronTagger
import string

@dataclass
class Token:

    word: str

    stanford_tag: str

    penn_tag: Optional[str] = None

    @property
    def normalized(self):

        return self.word.strip(
            string.punctuation +
            "“”‘’«»"
        ).lower()

class FWReclassifier:

    def __init__(self):

        self.tagger = PerceptronTagger()

def read_conll(self, filename):

    sentences = []

    sentence = []

    with open(filename, encoding="utf8") as f:

        for line in f:

            line = line.strip()

            if not line:

                if sentence:
                    sentences.append(sentence)
                    sentence = []

                continue

            word, tag = line.split()

            sentence.append(
                Token(word, tag)
            )

    if sentence:
        sentences.append(sentence)

    return sentences

def tag_sentence(self, sentence):

    words = [

        token.word

        for token in sentence

    ]

    penn = self.tagger.tag(words)

    for token, (_, penn_tag) in zip(sentence, penn):

        if token.stanford_tag == "FW":

            token.penn_tag = penn_tag

def reclassify(self, sentences):

    for sentence in sentences:

        self.tag_sentence(sentence)

for sentence in sentences:

    for token in sentence:

        if token.stanford_tag == "FW":

            print(
                token.word,
                token.penn_tag
            )