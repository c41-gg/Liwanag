import lzma, shutil

with lzma.open('data/raw/CC-100 Tagalog/tl.txt.xz') as f_in, \
     open('tl.txt', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)