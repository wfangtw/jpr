from argparse import ArgumentParser
import json
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval

import pathlib, os, random
import logging

#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

parser = ArgumentParser()
parser.add_argument('--dataset', required=True, type=str)
parser.add_argument('--beir_data_dir', required=True, type=str)
parser.add_argument('--split', required=True, type=str)
parser.add_argument('--orig_result_path', required=True, type=str)
parser.add_argument('--rerank_result_path', required=True, type=str)
parser.add_argument('--k_values', default=[10, 50], nargs='+', type=int)
args = parser.parse_args()

data_path = os.path.join(args.beir_data_dir, args.dataset)
corpus, queries, qrels = GenericDataLoader(data_path).load(split=args.split)
retriever = EvaluateRetrieval(None, score_function="dot", k_values=args.k_values)

def load_results(path):
    results = {}
    with open(path) as f:
        result_json = json.load(f)
        for sample in result_json:
            qid = sample['qid']
            qres = {}
            for doc_res in sample['ctxs']:
                score = float(doc_res['score'])
                doc_id = doc_res['id']
                qres[doc_id] = score

            results[qid] = qres

    return results

orig_results = load_results(args.orig_result_path)
rerank_results = load_results(args.rerank_result_path)

logging.info("Original ranking:")
retriever.evaluate(qrels, orig_results, retriever.k_values)
logging.info("Reranking:")
retriever.evaluate(qrels, rerank_results, retriever.k_values)
