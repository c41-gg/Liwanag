"""
silver_parallel.py
==================
Parallelized version of silver.py using ThreadPoolExecutor.

Runs multiple Stanford tagger batches in parallel to maximize throughput.

Key design:
  - ThreadPoolExecutor: each thread runs one Java subprocess batch independently
  - Threads are I/O-bound (waiting for Java), not CPU-bound, so no GIL issues
  - Independent file I/O per thread (no contention)
  - Progress tracking across all threads
  - Graceful error handling with batch-level retries
  - Resume support: skip already-processed sentences and append

Hardware requirement:
  - Each Java subprocess uses ~2GB heap (set --java-heap)
  - With N parallel workers × 2GB per worker, ensure you have 2N GB + 2GB headroom
  - Example: 4 parallel workers = 8GB minimum, 12GB recommended

Speed estimate:
  - Single-threaded: 200,000 sentences at 50/min = 4,000 minutes (~67 hours)
  - 4 parallel workers: ~1,000 minutes (~17 hours)
  - 8 parallel workers: ~500 minutes (~8.5 hours)

Usage:
    python liwanag\pos_tagger\silver.py `
        --raw-corpus "data/processed/corpus_clean3.txt" `
        --stanford-jar "Libraries/FSPOST/stanford-postagger.jar" `
        --stanford-model "Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger" `
        --output "data/processed/silver_raw.conll" `
        --max-sentences 200000 `
        --batch-size 50 `
        --max-sentence-len 60 `
        --java-heap 2g `
        --num-workers 4

Resume interrupted run:
    python liwanag\pos_tagger\silver.py `
        --raw-corpus "data/processed/corpus_clean3.txt" `
        --stanford-jar "Libraries/FSPOST/stanford-postagger.jar" `
        --stanford-model "Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger" `
        --output "data/processed/silver_raw.conll" `
        --batch-size 50 `
        --max-sentence-len 60 `
        --java-heap 2g `
        --num-workers 4 `
        --resume
"""

import argparse
import logging
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# Lock for thread-safe stats updates
stats_lock = Lock()


# ---------------------------------------------------------------------------
# Full MGNN tagset
# ---------------------------------------------------------------------------

MGNN_TAGS = {
    "NNC", "NNP", "NNPA", "NNCA",
    "PRS", "PRP", "PRSP", "PRO", "PRQ", "PRQP", "PRL", "PRC", "PRF", "PRI",
    "DTC", "DTCP", "DTP", "DTPP",
    "CCT", "CCR", "CCB", "CCA", "CCP", "CCU",
    "LM",
    "VBW", "VBS", "VBH", "VBN",
    "VBTS", "VBTR", "VBTF", "VBTP",
    "VBAF", "VBOF", "VBOB", "VBOL", "VBOI", "VBRF",
    "JJD", "JJC", "JJCC", "JJCS", "JJCN", "JJN",
    "RBD", "RBN", "RBK", "RBP", "RBB", "RBR",
    "RBQ", "RBT", "RBF", "RBW", "RBM", "RBL", "RBI", "RBJ", "RBS",
    "CDB",
    "TS",
    "FW",
    "PMP", "PME", "PMQ", "PMC", "PMSC", "PMS",
}

_COARSE_FALLBACK = [
    ("NNP", "NNP"), ("NNC", "NNC"), ("NN", "NNC"),
    ("VBT", "VBTS"), ("VBA", "VBAF"), ("VBO", "VBOF"),
    ("VBW", "VBW"), ("VB", "VBW"),
    ("JJC", "JJC"), ("JJ", "JJD"),
    ("RBI", "RBI"), ("RB", "RBD"),
    ("DTC", "DTC"), ("DTP", "DTP"), ("DT", "DTC"),
    ("PRS", "PRS"), ("PRP", "PRP"), ("PR", "PRS"),
    ("CC", "CCT"), ("CD", "CDB"), ("PM", "PMS"),
]


def normalize_tag(raw_tag: str) -> str:
    """Normalize raw Stanford output tag to MGNN tagset."""
    tag = raw_tag.strip().upper().rstrip(".,;:!?\"'")
    if tag in MGNN_TAGS:
        return tag
    for prefix, mgnn_tag in _COARSE_FALLBACK:
        if tag.startswith(prefix):
            return mgnn_tag
    return "FW"


# ---------------------------------------------------------------------------
# Resume support
# ---------------------------------------------------------------------------

def _count_sentences_in_output(output_path: str) -> tuple[int, dict]:
    """
    Count sentences already in output CoNLL file and extract stats.
    
    Returns: (sentence_count, stats_dict)
    CoNLL format: blank lines mark sentence boundaries
    """
    if not Path(output_path).exists():
        return 0, {"total_sentences": 0, "total_tokens": 0, "total_fw": 0}
    
    sentences = 0
    tokens = 0
    tokens_fw = 0
    in_sentence = False
    
    with open(output_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            
            if line == "":
                if in_sentence:
                    sentences += 1
                    in_sentence = False
            else:
                parts = line.split("\t")
                if len(parts) == 2:
                    tokens += 1
                    if parts[1] == "FW":
                        tokens_fw += 1
                    in_sentence = True
    
    if in_sentence:  # Last sentence not followed by blank line
        sentences += 1
    
    return sentences, {"total_sentences": sentences, "total_tokens": tokens, "total_fw": tokens_fw}


# ---------------------------------------------------------------------------
# Worker task for a single batch
# ---------------------------------------------------------------------------

def process_batch(
    batch_id: int,
    sentences: list[str],
    stanford_jar: str,
    stanford_model: str,
    java_heap: str,
    output_dir: Path,
    stats_dict: dict,
) -> tuple[int, int, int, int]:
    """
    Process one batch of sentences using Stanford tagger.

    Returns: (batch_id, sentences_written, tokens_total, tokens_fw)
    """
    output_path = output_dir / f"batch_{batch_id}.conll"
    tmp_input = output_dir / f"_batch_{batch_id}_input.txt"

    try:
        tmp_input.write_text("\n".join(sentences), encoding="utf-8")

        cmd = [
            "java",
            f"-Xmx{java_heap}",
            "-cp", stanford_jar,
            "edu.stanford.nlp.tagger.maxent.MaxentTagger",
            "-model", stanford_model,
            "-textFile", str(tmp_input),
            "-outputFormat", "slashTags",
            "-tagSeparator", "_",
            "-sentenceDelimiter", "newline",
            "-tokenize", "false",
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=600,
        )

        if result.returncode != 0:
            log.warning(f"Batch {batch_id} Stanford error: {result.stderr[:100]}")
            return batch_id, 0, 0, 0

        sentences_written = 0
        tokens_total = 0
        tokens_fw = 0

        with open(output_path, "w", encoding="utf-8") as out_f:
            for line in result.stdout.split("\n"):
                line = line.strip()

                if not line:
                    out_f.write("\n")
                    continue

                sentence_had_tokens = False
                for token_str in line.split():
                    if "_" not in token_str:
                        continue

                    sep = token_str.rfind("_")
                    word = token_str[:sep]
                    raw_tag = token_str[sep + 1:]

                    if not word:
                        continue

                    tag = normalize_tag(raw_tag)
                    out_f.write(f"{word}\t{tag}\n")
                    tokens_total += 1
                    if tag == "FW":
                        tokens_fw += 1
                    sentence_had_tokens = True

                if sentence_had_tokens:
                    out_f.write("\n")
                    sentences_written += 1

        log.info(
            f"Batch {batch_id:3d}: {sentences_written:>5} sentences | "
            f"{tokens_total:>7} tokens | FW: {tokens_fw:>5} ({tokens_fw/max(tokens_total,1)*100:>5.1f}%)"
        )

        return batch_id, sentences_written, tokens_total, tokens_fw

    except subprocess.TimeoutExpired:
        log.warning(f"Batch {batch_id} timed out")
        return batch_id, 0, 0, 0
    except Exception as e:
        log.error(f"Batch {batch_id} error: {e}")
        return batch_id, 0, 0, 0
    finally:
        tmp_input.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main parallel runner
# ---------------------------------------------------------------------------

def run_stanford_tagger_parallel(
    raw_corpus: str,
    stanford_jar: str,
    stanford_model: str,
    output_conll: str,
    max_sentences: int = 200_000,
    batch_size: int = 50,
    max_sentence_len: int = 60,
    java_heap: str = "2g",
    num_workers: int = 4,
    resume: bool = False,
) -> None:
    """
    Run Stanford tagger in parallel using ThreadPoolExecutor.

    Args:
        raw_corpus: Path to raw text, one sentence per line
        stanford_jar: Path to stanford-postagger.jar
        stanford_model: Path to Filipino tagger model
        output_conll: Output CoNLL file
        max_sentences: Hard cap on sentences processed
        batch_size: Sentences per batch
        max_sentence_len: Filter sentences longer than this
        java_heap: Java heap per subprocess
        num_workers: Number of parallel Java processes
        resume: If True, skip already-processed sentences and append
    """
    _check_java()

    log.info(f"Reading raw corpus: {raw_corpus}")
    sentences, n_dropped = _read_sentences(raw_corpus, max_sentences, max_sentence_len)
    log.info(
        f"Loaded {len(sentences):,} sentences "
        f"(dropped {n_dropped:,} longer than {max_sentence_len} tokens)"
    )

    output_path = Path(output_conll)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    batch_dir = output_path.parent / ".batch_temp"
    batch_dir.mkdir(parents=True, exist_ok=True)

    # Handle resume: skip already-processed sentences
    start_idx = 0
    existing_stats = None
    file_mode = "w"
    
    if resume and output_path.exists():
        sentences_done, existing_stats = _count_sentences_in_output(str(output_path))
        start_idx = sentences_done
        file_mode = "a"
        log.info(f"Resuming: {sentences_done:,} sentences already processed")
        log.info(f"Remaining to process: {len(sentences) - start_idx:,} sentences")
        sentences = sentences[start_idx:]
    
    if not sentences:
        log.info("No new sentences to process. Output file is complete.")
        return

    # Split sentences into batches
    num_batches = (len(sentences) + batch_size - 1) // batch_size
    batches = [
        sentences[i * batch_size : (i + 1) * batch_size]
        for i in range(num_batches)
    ]

    log.info(
        f"Processing {len(batches)} batches × {batch_size} sentences "
        f"across {num_workers} parallel workers"
    )
    log.info(f"Estimated time: ~{len(batches) // num_workers} minutes")
    log.info("")

    # Initialize stats from existing output if resuming
    if resume and existing_stats:
        stats = existing_stats.copy()
        log.info(f"Continuing from: {stats['total_sentences']:,} sentences, {stats['total_tokens']:,} tokens")
    else:
        stats = {"total_sentences": 0, "total_tokens": 0, "total_fw": 0}
    
    completed = 0

    # Run batches in parallel
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(
                process_batch,
                batch_id=i,
                sentences=batch,
                stanford_jar=stanford_jar,
                stanford_model=stanford_model,
                java_heap=java_heap,
                output_dir=batch_dir,
                stats_dict=stats,
            ): i
            for i, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            batch_id, sent_written, tok_total, tok_fw = future.result()
            with stats_lock:
                stats["total_sentences"] += sent_written
                stats["total_tokens"] += tok_total
                stats["total_fw"] += tok_fw
                completed += 1

            fw_rate = (
                tok_fw / max(tok_total, 1) * 100 if tok_total > 0 else 0.0
            )
            log.debug(f"Batch {batch_id} complete. Progress: {completed}/{len(batches)}")

    log.info("")
    log.info("All batches complete. Merging output files...")

    # Merge all batch files in order
    with open(output_path, file_mode, encoding="utf-8") as out_f:
        for batch_id in range(len(batches)):
            batch_file = batch_dir / f"batch_{batch_id}.conll"
            if batch_file.exists():
                with open(batch_file, "r", encoding="utf-8") as f:
                    out_f.write(f.read())
                batch_file.unlink()

    batch_dir.rmdir()

    _print_summary(output_conll, stats, resumed=(resume and start_idx > 0))


def _read_sentences(
    path: str,
    max_sentences: int,
    max_sentence_len: int,
) -> tuple[list[str], int]:
    """Read and filter sentences."""
    sentences = []
    n_dropped = 0

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if len(line.split()) > max_sentence_len:
                n_dropped += 1
                continue
            sentences.append(line)
            if len(sentences) >= max_sentences:
                break

    return sentences, n_dropped


def _check_java() -> None:
    """Verify Java is available."""
    try:
        result = subprocess.run(
            ["java", "-version"],
            capture_output=True,
            text=True,
        )
        version_line = (result.stderr or result.stdout).split("\n")[0]
        log.info(f"Java found: {version_line.strip()}")
    except FileNotFoundError:
        log.error("Java not found on PATH. Install Java 8+ and add to PATH.")
        sys.exit(1)


def _print_summary(output_conll: str, stats: dict, resumed: bool = False) -> None:
    """Print final summary."""
    fw_rate = (
        stats["total_fw"] / stats["total_tokens"] * 100
        if stats["total_tokens"] > 0
        else 0.0
    )
    mode_str = "RESUMED AND COMPLETED" if resumed else "COMPLETE"
    log.info("=" * 50)
    log.info(f"PARALLEL SILVER ANNOTATION {mode_str}")
    log.info("=" * 50)
    log.info(f"  Output file      : {output_conll}")
    log.info(f"  Sentences written: {stats['total_sentences']:,}")
    log.info(f"  Total tokens     : {stats['total_tokens']:,}")
    log.info(f"  FW tokens        : {stats['total_fw']:,}  ({fw_rate:.1f}%)")
    log.info("=" * 50)
    log.info("Next step:")
    log.info("  python fw_reclassifier.py \\")
    log.info(f"      --input  {output_conll} \\")
    log.info("      --output data/processed/silver_corrected.conll \\")
    log.info("      --report data/processed/reclassification_report.txt \\")
    log.info("      --gold-output data/processed/gold_annotation_todo.conll")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel MGNN silver-label generation via Stanford FSPOST tagger"
    )
    parser.add_argument(
        "--raw-corpus", required=True,
        help="Raw Tagalog text file — one sentence per line"
    )
    parser.add_argument(
        "--stanford-jar", required=True,
        help="Path to stanford-postagger.jar"
    )
    parser.add_argument(
        "--stanford-model", required=True,
        help="Path to Filipino tagger model"
    )
    parser.add_argument(
        "--output", required=True,
        help="Output CoNLL file path"
    )
    parser.add_argument(
        "--max-sentences", type=int, default=200_000,
        help="Maximum sentences to process (default: 200,000)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50,
        help="Sentences per batch (default: 50)"
    )
    parser.add_argument(
        "--max-sentence-len", type=int, default=60,
        help="Filter sentences longer than this (default: 60)"
    )
    parser.add_argument(
        "--java-heap", default="2g",
        help="Java heap per subprocess (default: 2g)"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4,
        help="Number of parallel Java processes (default: 4)"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume interrupted run: skip processed sentences, append results"
    )
    args = parser.parse_args()

    # Warn if num_workers seems too high for available RAM
    try:
        import psutil
        available_gb = psutil.virtual_memory().available / (1024 ** 3)
        needed_gb = args.num_workers * 2 + 2
        if needed_gb > available_gb:
            log.warning(
                f"⚠️  You have ~{available_gb:.1f}GB available RAM, "
                f"but {args.num_workers} workers × 2GB + 2GB overhead = {needed_gb:.1f}GB needed."
            )
            log.warning(f"   Consider using --num-workers {max(1, int(available_gb / 2) - 1)}")
    except ImportError:
        pass

    run_stanford_tagger_parallel(
        raw_corpus=args.raw_corpus,
        stanford_jar=args.stanford_jar,
        stanford_model=args.stanford_model,
        output_conll=args.output,
        max_sentences=args.max_sentences,
        batch_size=args.batch_size,
        max_sentence_len=args.max_sentence_len,
        java_heap=args.java_heap,
        num_workers=args.num_workers,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()