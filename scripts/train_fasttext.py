"""
scripts/train_fasttext.py
Trains the Tagalog FastText morphological embedding model.

Usage:
    python scripts/train_fasttext.py --corpus corpus_sample.txt
"""

import argparse
import multiprocessing
from gensim.models import FastText # type: ignore

def train(corpus_path: str, output_path: str):
    print(f"Training FastText on {corpus_path}...")
    model = FastText(
        corpus_file=corpus_path,
        vector_size=300,
        min_n=2,
        max_n=6,
        epochs=10,
        workers=multiprocessing.cpu_count(),
        min_count=3,
        negative=10,
        sample=1e-4,
    )
    model.save(output_path)
    print(f"Saved to {output_path}")
    print(f"Vocabulary size: {len(model.wv):,}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--corpus', default='corpus_sample.txt')
    parser.add_argument('--output', default='data/models/tagalog_fasttext.model')
    args = parser.parse_args()
    train(args.corpus, args.output)