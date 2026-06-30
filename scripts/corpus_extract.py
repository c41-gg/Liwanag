import random

random.seed(43)  # reproducible sample every time you run it

with open('data\\raw\\CC-100 Tagalog\\corpus_tl.txt', encoding='utf-8') as f_in, \
     open('data\\raw\\CC-100 Tagalog\\corpus_sample2.txt', 'w', encoding='utf-8') as f_out:
    
    for line in f_in:
        if random.random() < 0.009:  
            f_out.write(line)

print("Done.")

# Verify count
with open('data\\raw\\CC-100 Tagalog\\corpus_sample2.txt') as f:
    count = sum(1 for _ in f)
print(f"Sampled: {count:,} sentences")