import random

random.seed(42)  # reproducible sample every time you run it

with open('corpus_clean.txt', encoding='utf-8') as f_in, \
     open('corpus_sample.txt', 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        if random.random() < 0.05:   # ~5% of 22M = ~1.1M sentences
            f_out.write(line)

print("Done.")

# Verify count
with open('corpus_sample.txt') as f:
    count = sum(1 for _ in f)
print(f"Sampled: {count:,} sentences")