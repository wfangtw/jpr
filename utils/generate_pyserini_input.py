import sys
import json
# from nltk.tokenize import word_tokenize

# max_length = 128
with open(sys.argv[1]) as in_f, open(sys.argv[2], 'w') as out_f:
    data = [json.loads(line) for line in in_f]
    for datum in data:
        contents = datum['text']
        # contents = '\n\n'.join(contents.split('\n\n')[::-1])
        # contents = ' '.join(word_tokenize(contents)[:max_length])
        new_datum = {
            'id': datum['_id'],
            'contents': contents,
            'title': datum['title'],
        }
        print(json.dumps(new_datum), file=out_f)


