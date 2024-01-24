from argparse import ArgumentParser
import json
from typing import Type, List, Dict, Union, Tuple

from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.lexical import BM25Search as BM25
from beir.retrieval.search.dense import DenseRetrievalExactSearch as DRES
from pyserini.search.lucene import LuceneSearcher
from nltk.tokenize import word_tokenize

import tqdm
import pathlib, os, random
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = ArgumentParser()
parser.add_argument('--model', required=True, type=str, choices=['dpr', 'bm25'])
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--split', required=True, type=str)
parser.add_argument('--data_dir', type=str)
parser.add_argument('--save_dir', type=str)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--overwrite', action='store_true')
parser.add_argument('--num_workers', type=int, default=10)
parser.add_argument('--index_name', type=str, default=None)
args = parser.parse_args()


class BM25_Pyserini(BM25):
    def __init__(self, index_name: str, batch_size: int = 128, threads: int = 1, prebuilt: bool = True):
        self.index_name = index_name
        self.results = {}
        self.batch_size = batch_size
        self.threads = threads
        if prebuilt:
            full_index_name = index_name if 'msmarco' in index_name else 'beir-v1.0.0-{}-multifield'.format(index_name)
            self.searcher = LuceneSearcher.from_prebuilt_index(full_index_name)
        else:
            self.searcher = LuceneSearcher(index_name)
        self.searcher.set_bm25(0.9, 0.4)

    def search(self, corpus: Dict[str, Dict[str, str]], queries: Dict[str, str], top_k: List[int], *args, **kwargs) -> Dict[str, Dict[str, float]]:
        query_ids = list(queries.keys())
        queries = [queries[qid] for qid in query_ids]
        queries = [' '.join(word_tokenize('\n\n'.join(query.split('\n\n')[::-1]))[:128]) for query in queries]
        
        for start_idx in tqdm.trange(0, len(queries), self.batch_size, desc='que'):
            query_ids_batch = query_ids[start_idx:start_idx+self.batch_size]
            queries_batch = queries[start_idx:start_idx+self.batch_size]
            results = self.searcher.batch_search(
                queries_batch, query_ids_batch, top_k + 1, 
                threads=self.threads, fields={'contents': 1.0, 'title': 1.0})
            for query_id in query_ids_batch:
                hits = results[query_id]
                scores = {}
                for hit in hits:
                    corpus_id = hit.docid
                    score = hit.score
                    if corpus_id != query_id:
                        scores[corpus_id] = score
                self.results[query_id] = scores
            
        return self.results


dataset_path = args.dataset.replace('-', '/') if 'cqadupstack' in args.dataset else args.dataset
data_path = os.path.join(args.data_dir, dataset_path)

save_dir = os.path.join(args.save_dir, args.model)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
save_path = os.path.join(save_dir, 'beir-{}-{}.json'.format(args.dataset, args.split))
if os.path.exists(save_path) and not args.overwrite:
    exit()

logging.info("Dataset: {}-{}".format(args.dataset, args.split))
corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)

if args.model == 'dpr':
    model = DRES(
        models.DPR((
                "dpr-question_encoder-multiset-base",
                "dpr-ctx_encoder-multiset-base"
        )), 
        batch_size=args.batch_size
    )
else:
    prebuilt = True
    if args.dataset == 'msmarco':
        index_name = 'msmarco-v1-passage-slim'
    elif args.dataset == 'msmarco-v2':
        index_name = 'msmarco-v2-passage-slim'
    elif args.index_name is not None:
        index_name = args.index_name
        prebuilt = False
    else:
        index_name = args.dataset
    model = BM25_Pyserini(index_name=index_name, batch_size=args.batch_size, threads=args.num_workers, prebuilt=prebuilt)

retriever = EvaluateRetrieval(model, score_function="dot", k_values=[1, 5, 10, 1000])

#### Retrieve dense results (format of results is identical to qrels)
results = retriever.retrieve(corpus, queries)

#### Evaluate your retrieval using NDCG@k, MAP@K ...
logging.info("Retriever evaluation for k in: {}".format(retriever.k_values))
ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)

results_json = []
for qid, qres in results.items():
    sample = {}
    query = queries[qid]
    sample['qid'] = qid
    sample['question'] = query
    sample['answers'] = ['dummy']
    labels = qrels[qid]
    posids = set(docid for docid, label in labels.items() if label > 0)
    ctxs = []
    for k, (docid, score) in enumerate(qres.items()):
        if k >= 1001:
            break
        ctxs.append({
            'score': str(score),
            'has_answer': docid in posids,
            'id': docid,
            'rank': k + 1,
        })
    sample['ctxs'] = ctxs
    results_json.append(sample)

with open(save_path, 'w') as f:
    json.dump(results_json, f, indent=4)

