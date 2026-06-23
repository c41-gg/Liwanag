from gensim.models import FastText # type: ignore
import multiprocessing

model = FastText(
    corpus_file='corpus_sample.txt',
    vector_size=300,
    min_n=2,
    max_n=6,
    epochs=10,        # back to 10 since corpus is smaller now
    workers=multiprocessing.cpu_count(),
    min_count=3,       # slightly lower than before since corpus is smaller
    negative=10,        # default is 5 — increase to push unrelated words apart
    ns_exponent=0.75,   # keep default — controls negative sample distribution
    sample=1e-4,
)

model.save('data/modelstagalog_fasttext.model')
print("Training complete.")