import subprocess
import tempfile
import os

jar = 'Libraries/FSPOST/stanford-postagger.jar'
model = 'Libraries/FSPOST/filipino-left5words-owlqn2-distsim-pref6-inf2.tagger'
sentence = 'Kumain ang bata ng tinapay tinoast ito ng matagal para ipeanut butter.'

with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt') as f:
    f.write(sentence)
    temp = f.name

result = subprocess.run(
    ['java', '-mx1g', '-cp', jar, 
     'edu.stanford.nlp.tagger.maxent.MaxentTagger', 
     '-model', model, 
     '-textFile', temp],
    capture_output=True,
    text=True
)

print("Output:")
print(result.stdout)
print("\nParsed tags:")
for tag in result.stdout.strip().split():
    print(tag)

os.unlink(temp)