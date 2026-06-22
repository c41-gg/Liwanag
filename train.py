import fasttext

model = fasttext.train_unsupervised(
    'tagalog_clean.txt',
    model='skipgram',     # better for morphology than cbow
    dim=300,              # embedding size
    minn=2,               # min char n-gram length
    maxn=6,               # max char n-gram length — covers Tagalog affixes
    epoch=10,
    lr=0.05,
    thread=4
)

model.save_model('tagalog_fasttext.bin')