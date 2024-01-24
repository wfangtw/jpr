import json
import csv
import torch
from torch.utils.data import Dataset, DataLoader
from beir.datasets.data_loader import GenericDataLoader


def print_rank_0(message):
    """If distributed is initialized print only on rank 0."""
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


class JointDataset(Dataset):
    def __init__(self, data_path, json_data_path, split='test', top_k=None):
        self.corpus, self.queries, self.qrels = GenericDataLoader(data_path).load(split=split)
        self.data = self.load_dataset(json_data_path)
        self.samples = self._preprocess_samples(top_k)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bm25_ctxs = []
        for ctx in sample['bm25_contexts']:
            doc_id = ctx['doc_id']
            doc = self.corpus[doc_id]
            bm25_ctxs.append({
                'doc_id': doc_id,
                'doc_title': doc['title'],
                'doc_text': doc['text'],
                'label': int(ctx['label']),
            })
        gold_ctxs = []
        for ctx in sample['gold_contexts']:
            doc_id = ctx['doc_id']
            doc = self.corpus[doc_id]
            gold_ctxs.append({
                'doc_id': doc_id,
                'doc_title': doc['title'],
                'doc_text': doc['text'],
                'label': int(ctx['label']),
            })

        return {
            'q_id': sample['q_id'],
            'question': sample['question'],
            'bm25_contexts': bm25_ctxs,
            'gold_contexts': gold_ctxs,
        }

    @staticmethod
    def load_dataset(filepath):
        print_rank_0('Loading dataset from {}...'.format(filepath))
        with open(filepath) as fp:
            data = json.load(fp)
        return data

    def _preprocess_samples(self, top_k=None):
        samples = []
        for i, row in enumerate(self.data, 1):
            bm25_contexts = row['ctxs'] if top_k is None else row['ctxs'][:top_k]
            bm25_contexts = [{'doc_id': str(ctx['id']), 'label': ctx['has_answer']} for ctx in bm25_contexts]

            q_id = row['qid']
            gold_contexts = [{'doc_id': doc_id, 'label': label} for doc_id, label in self.qrels[q_id].items()]
            # gold_contexts = [{'doc_id': doc_id, 'label': label} for doc_id, label in self.qrels[q_id].items() if int(label) == 1]
            # if len(bm25_contexts) + len(gold_contexts) < top_k:
            if top_k is not None and len(bm25_contexts) < top_k:
                continue

            samples.append({
                'q_id': q_id,
                'question': row['question'],
                'bm25_contexts': bm25_contexts,
                'gold_contexts': gold_contexts,
            })
        return samples


class JointOpenQADataset(JointDataset):
    def __init__(self, bm25_json_data_path, dpr_json_data_path=None, psgs_path=None, corpus=None, top_k=None, for_eval=False, max_samples=None, index_range=None):
        self.for_eval = for_eval
        self.data = self.load_dataset(bm25_json_data_path)
        if max_samples is not None:
            self.data = self.data[:max_samples]
        if index_range is not None:
            self.data = self.data[index_range[0]:index_range[1]]
        self.dpr_data = self.load_dataset(dpr_json_data_path) if dpr_json_data_path is not None else None
        if psgs_path is not None:
            self.corpus = self.load_corpus(psgs_path)
        elif corpus is not None:
            self.corpus = corpus
        else:
            raise NotImplementedError
        self.samples = self._preprocess_samples(top_k)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        bm25_ctxs = []
        for ctx in sample['bm25_contexts']:
            doc_id = ctx['doc_id']
            doc = self.corpus[doc_id]
            bm25_ctxs.append({
                'doc_id': doc_id,
                'doc_title': doc['title'],
                'doc_text': doc['text'],
                'label': int(ctx['label']),
            })
        gold_ctxs = []
        for ctx in sample['gold_contexts']:
            doc_id = ctx['doc_id']
            doc = self.corpus[doc_id]
            gold_ctxs.append({
                'doc_id': doc_id,
                'doc_title': doc['title'],
                'doc_text': doc['text'],
                'label': int(ctx['label']),
            })

        return {
            'question': sample['question'],
            'bm25_contexts': bm25_ctxs,
            'gold_contexts': gold_ctxs,
        }

    @staticmethod
    def load_corpus(psgs_path):
        print_rank_0(' > Processing {} ...'.format(psgs_path))
        total = 0
        id2text = {}

        is_csv = psgs_path.endswith('.tsv')

        with open(psgs_path, encoding='utf8') as f:
            if is_csv:
                reader = csv.reader(f, delimiter='\t')
                next(reader, None)  # skip the headers
            else:
                reader = f

            for total, row in enumerate(reader, 1):
                if is_csv:
                    # file format: doc_id, doc_text, title
                    doc_id = row[0]
                    text = row[1]
                    title = row[2]
                else:
                    # jsonlines format
                    row_data = json.loads(row)
                    doc_id = row_data['_id']
                    text = row_data['text']
                    title = row_data['title']

                assert doc_id not in id2text
                id2text[doc_id] = {
                    'title': title,
                    'text': text
                }
                if total % 100000 == 0:
                    print_rank_0('  > processed {} rows so far ...'.format(total))

        print_rank_0(' > processed {} rows in total'.format(total))

        return id2text

    def _preprocess_samples(self, top_k=None):
        if not self.for_eval:
            q2gold = {}
            for i, row in enumerate(self.dpr_data, 1):
                question = row['question']
                gold_contexts = row['positive_ctxs']
                q2gold[question.lower()] = (gold_contexts, question)

        samples = []
        for i, row in enumerate(self.data, 1):
            question = row['question']
            if not self.for_eval and question.lower() not in q2gold:
                continue

            bm25_contexts = row['ctxs'] if top_k is None else row['ctxs'][:top_k]
            bm25_contexts = [{'doc_id': str(ctx['id']), 'label': ctx['has_answer']} for ctx in bm25_contexts]
            if not self.for_eval and len(bm25_contexts) < top_k:
                continue

            if not self.for_eval:
                gold_contexts, question = q2gold[question.lower()]
                if len(gold_contexts) == 0:
                    continue
                id_key = 'passage_id' if 'passage_id' in gold_contexts[0] else 'psg_id'
                gold_contexts = [{'doc_id': str(ctx[id_key]), 'label': 1} for ctx in gold_contexts]
            else:
                gold_contexts = bm25_contexts

            samples.append({
                'question': question,
                'bm25_contexts': bm25_contexts,
                'gold_contexts': gold_contexts,
            })
        return samples


class BEIRDataset(Dataset):
    def __init__(self, data_path, split, gen_data_path=None):
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=split)

        self.corpus = corpus
        self.queries = queries
        self.qrels = qrels

        sample_pairs = []
        for qid, docids in qrels.items():
            for docid in docids.keys():
                sample_pairs.append((qid, docid))

        if gen_data_path is not None:
            gen_data = {}
            with open(gen_data_path, 'r') as f:
                gen_data_json = json.load(f)

            for d in gen_data_json:
                for docid, sents in d.items():
                    top_sent = sents[0]['text']
                    gen_data[docid] = top_sent

        samples = []
        for qid, docid in sample_pairs:
            query = queries[qid]
            doc = corpus[docid]
            gen_query = gen_data[docid] if gen_data_path is not None else None
            sample = {
                'docid': docid,
                'query': query,
                'gen_query': gen_query,
                'title': doc['title'],
                'text': doc['text'],
            }
            samples.append(sample)

        self.samples = samples

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


        
