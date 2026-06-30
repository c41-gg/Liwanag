import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "silver_module",
    ROOT / "liwanag" / "pos_tagger" / "silver.py",
)
assert SPEC is not None and SPEC.loader is not None
silver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(silver)


def test_build_batches_splits_sentences():
    sentences = [f"s{i}" for i in range(7)]

    batches = silver._build_batches(sentences, batch_size=3)

    assert batches == [
        ["s0", "s1", "s2"],
        ["s3", "s4", "s5"],
        ["s6"],
    ]


def test_count_completed_sentences_from_conll(tmp_path):
    output_path = tmp_path / "resume.conll"
    output_path.write_text(
        "foo\tNN\n\n"
        "bar\tVB\n\n"
        "baz\tNN\n",
        encoding="utf-8",
    )

    assert silver._count_completed_sentences(output_path) == 2


def test_parallel_settings_shrink_batches_for_many_workers():
    effective_batch_size, worker_count = silver._resolve_parallel_settings(
        batch_size=500,
        num_workers=4,
        total_sentences=1000,
    )

    assert worker_count == 4
    assert effective_batch_size < 500
