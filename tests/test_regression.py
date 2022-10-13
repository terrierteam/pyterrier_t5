import unittest
import pyterrier as pt
import pyterrier_t5

# A simple regression test
class T5RegressionTests(unittest.TestCase):

    def test_monot5_vaswani(self):
        if not pt.started():
            pt.init()
        bm25 = pt.BatchRetrieve(pt.get_dataset('vaswani').get_index(), wmodel='BM25')
        monoT5 = pyterrier_t5.MonoT5ReRanker()
        pipeline = bm25 % 20 >> pt.text.get_text(pt.get_dataset('irds:vaswani'), 'text') >> monoT5
        result = pipeline.search('fluid dynamics')
        self.assertEqual(result.iloc[0]['docno'], '11216')
        self.assertAlmostEqual(result.iloc[0]['score'], -2.186261, places=4)
        self.assertEqual(result.iloc[0]['rank'], 0)
        self.assertEqual(result.iloc[1]['docno'], '5299')
        self.assertAlmostEqual(result.iloc[1]['score'], -8.078399, places=4)
        self.assertEqual(result.iloc[1]['rank'], 1)
        self.assertEqual(result.iloc[-1]['docno'], '3442')
        self.assertAlmostEqual(result.iloc[-1]['score'], -12.725513, places=4)
        self.assertEqual(result.iloc[-1]['rank'], 19)

    def test_duot5_vaswani(self):
        if not pt.started():
            pt.init()
        bm25 = pt.BatchRetrieve(pt.get_dataset('vaswani').get_index(), wmodel='BM25')
        duoT5 = pyterrier_t5.DuoT5ReRanker()
        pipeline = bm25 % 10 >> pt.text.get_text(pt.get_dataset('irds:vaswani'), 'text') >> duoT5
        result = pipeline.search('fluid dynamics')
        self.assertEqual(result.iloc[0]['docno'], '11216')
        self.assertAlmostEqual(result.iloc[0]['score'], 93.090627, places=4)
        self.assertEqual(result.iloc[0]['rank'], 0)
        self.assertEqual(result.iloc[1]['docno'], '4767')
        self.assertAlmostEqual(result.iloc[1]['score'], 22.323915, places=4)
        self.assertEqual(result.iloc[1]['rank'], 1)
        self.assertEqual(result.iloc[-1]['docno'], '10073')
        self.assertAlmostEqual(result.iloc[-1]['score'], -22.371883, places=4)
        self.assertEqual(result.iloc[-1]['rank'], 9)


if __name__ == '__main__':
    unittest.main()
