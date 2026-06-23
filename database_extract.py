import lzma, shutil

with lzma.open('tl.txt.xz') as f_in, \
     open('tl.txt', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)