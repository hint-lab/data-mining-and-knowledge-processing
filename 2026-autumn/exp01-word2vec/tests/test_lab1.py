import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

LAB = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(LAB))
from lab_common import Review, read_reviews, tokenize, write_reviews
from prepare_data import local_sample, download_sample, cached_batch


class DataTests(unittest.TestCase):
    def test_headerless_first_row_and_multiline_are_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'train.csv'
            with p.open('w', newline='') as f:
                csv.writer(f).writerows([[1, 'bad, item', 'not good\nat all'], [2, '', 'great phone']])
            rows = list(read_reviews(p))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0].label, 1)
            self.assertIn('not', rows[0].tokens())
            write_reviews(p, rows)
            self.assertEqual(list(read_reviews(p)), rows)

    def test_tokenizer_keeps_negation_and_apostrophes(self):
        self.assertEqual(tokenize("<p>NOT good &amp; I don’t recommend it.</p>"),
                         ['not', 'good', 'i', "don't", 'recommend', 'it'])

    def test_bad_label_or_blank_text_fails(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)/'train.csv'
            for text in ['0,title,hello\n', '1,,\n', '1,missing_column\n']:
                p.write_text(text)
                with self.assertRaises(ValueError): list(read_reviews(p))

    def test_reservoir_dedup_exclusion_and_seed(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'train.csv'
            rows=[Review(1, '', f'review number {i} is not good') for i in range(30)]
            write_reviews(p, rows+[rows[0]])
            a, info=local_sample(p, 10, 7, {rows[0].key()})
            b, _=local_sample(p, 10, 7, {rows[0].key()})
            self.assertEqual(a, b)
            self.assertEqual(info['eligible_rows'], 29)
            self.assertNotIn(rows[0], a)
            self.assertEqual(len({r.key() for r in a}), 10)
            with self.assertRaises(ValueError): local_sample(p, 40, 7, set())

    def test_download_label_mapping_and_truncation(self):
        batch={'num_rows_total':1000, 'rows':[
            {'row':{'label':0, 'title':'bad', 'content':'not good'}, 'truncated_cells':[]},
            {'row':{'label':1, 'title':'great', 'content':'very good'}, 'truncated_cells':[]}]}
        with patch('prepare_data.get_batch', return_value=batch):
            rows, info=download_sample(2, 42, set())
            self.assertEqual([r.label for r in rows], [1,2])
            self.assertEqual(info['split'], 'train')
        batch['rows'][0]['truncated_cells']=['content']
        with patch('prepare_data.get_batch', return_value=batch):
            with self.assertRaises(ValueError): download_sample(2, 42, set())

    def test_download_cache_reuses_successful_batch(self):
        with tempfile.TemporaryDirectory() as d:
            batch={'num_rows_total':1000,'rows':[]}
            with patch('prepare_data.get_batch',return_value=batch) as fetch:
                self.assertEqual(cached_batch(100,100,d),batch)
                self.assertEqual(cached_batch(100,100,d),batch)
                fetch.assert_called_once_with(100,100)


class PipelineTests(unittest.TestCase):
    def test_handoff_reproducibility_oov_and_no_overwrite(self):
        import numpy as np
        from gensim.models import KeyedVectors
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for name in ['first', 'repeat']:
                r=subprocess.run([sys.executable,str(LAB/'run_lab.py'),
                                  '--train-csv',str(LAB/'examples/demo_train.csv'),
                                  '--output-dir',str(root/name),'--epochs','2','--vector-size','16'],
                                 capture_output=True,text=True,timeout=90)
                self.assertEqual(r.returncode,0,r.stdout+r.stderr)
            first=root/'first'
            manifest=json.loads((first/'run_manifest.json').read_text())
            self.assertEqual(manifest['status'],'complete')
            self.assertEqual(manifest['rows'],24)
            with (first/'comparison.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),2)
            for window in [2,5]:
                a=first/f'window_{window}'
                matrix=np.load(a/'embedding.npy',allow_pickle=False)
                vocab=json.loads((a/'vocab.json').read_text())
                kv=KeyedVectors.load(str(a/'vectors.kv'))
                self.assertEqual(matrix.shape,(len(vocab['id_to_token']),16))
                np.testing.assert_array_equal(matrix[0],np.zeros(16))
                for word in kv.index_to_key:
                    np.testing.assert_array_equal(matrix[vocab['token_to_id'][word]],kv[word])
                np.testing.assert_array_equal(matrix,np.load(root/'repeat'/f'window_{window}'/'embedding.npy'))
                self.assertGreater((a/'pca.png').stat().st_size,1000)
            q=subprocess.run([sys.executable,str(LAB/'query_model.py'),'--vectors',str(first/'window_5/vectors.kv'),
                              'not_in_the_vocabulary'],capture_output=True,text=True)
            self.assertEqual(q.returncode,0,q.stderr)
            self.assertIn('OOV',q.stdout)
            marker=(first/'run_manifest.json').read_bytes()
            r=subprocess.run([sys.executable,str(LAB/'run_lab.py'),'--train-csv',str(LAB/'examples/demo_train.csv'),
                              '--output-dir',str(first)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertEqual(marker,(first/'run_manifest.json').read_bytes())

    def test_train_dev_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            dev=Path(d)/'dev.csv'
            write_reviews(dev,[next(read_reviews(LAB/'examples/demo_train.csv'))])
            r=subprocess.run([sys.executable,str(LAB/'run_lab.py'),'--train-csv',str(LAB/'examples/demo_train.csv'),
                              '--output-dir',str(Path(d)/'out'),'--exclude-csv',str(dev)],capture_output=True,text=True)
            self.assertNotEqual(r.returncode,0)
            self.assertIn('also occur',r.stderr)
            self.assertFalse((Path(d)/'out').exists())


if __name__ == '__main__': unittest.main()
