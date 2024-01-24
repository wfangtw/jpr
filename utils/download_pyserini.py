from pyserini.search.lucene import LuceneSearcher

datasets = ['msmarco', 'trec-covid', 'nfcorpus', 'nq', 'hotpotqa', #'bioasq',
            'fiqa', 'arguana', 'webis-touche2020', #'signal1m', 'trec-news', 'robust04'
            'cqadupstack-android', 
            'cqadupstack-english', 
            'cqadupstack-gaming', 
            'cqadupstack-gis', 
            'cqadupstack-mathematica', 
            'cqadupstack-physics', 
            'cqadupstack-programmers', 
            'cqadupstack-stats', 
            'cqadupstack-tex', 
            'cqadupstack-unix', 
            'cqadupstack-webmasters', 
            'cqadupstack-wordpress', 
            'quora', 'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact',]
for dataset in datasets:
    if "msmarco" not in dataset:
        index_name = 'beir-v1.0.0-{}-multifield'.format(dataset)
    elif dataset == 'msmarco':
        index_name = 'msmarco-v1-passage-slim'
    LuceneSearcher.from_prebuilt_index(index_name)
