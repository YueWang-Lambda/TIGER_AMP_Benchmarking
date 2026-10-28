"""Regression and fault-injection checks for the public evaluation contract."""
import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'code'))
import evaluate as ev


class EndpointTests(unittest.TestCase):
    def setUp(self):
        self.pairs = pd.DataFrame(dict(family=['F']*3,anchor_sequence=['A','A','B'],query_sequence=['B','C','C'],y_true=[1.,2.,1.]))
        self.pred = self.pairs.assign(y_pred=[1.1,1.9,.8])

    def test_reordering_is_aligned_by_identity(self):
        np.testing.assert_array_equal(ev.aligned_prediction(self.pred.iloc[::-1],self.pairs),[1.1,1.9,.8])

    def test_missing_duplicate_nan_and_wrong_label_rejected(self):
        invalid=[self.pred.iloc[:2],pd.concat([self.pred,self.pred.iloc[:1]]),self.pred.assign(y_pred=[1,np.nan,2]),self.pred.assign(y_true=self.pred.y_true*np.log10(2))]
        for frame in invalid:
            with self.subTest(frame=frame.to_dict()),self.assertRaises(ValueError):
                ev.aligned_prediction(frame,self.pairs)

    def test_pcc_does_not_validate_scale(self):
        y=np.array([-2.,-1.,1.,3.])
        correct=ev.family_metrics(y,y);wrong=ev.family_metrics(y,2*y+1)
        self.assertAlmostEqual(correct['PCC'],wrong['PCC'])
        self.assertEqual(correct['RSE'],0)
        self.assertGreater(wrong['RSE'],0)

    def test_constant_prediction_remains_undefined(self):
        result=ev.family_metrics(np.array([1.,2.,4.]),np.zeros(3))
        self.assertTrue(np.isnan(result['PCC']) and np.isnan(result['KCC']))
        self.assertTrue(np.isfinite(result['RSE']))
        rows=[dict(method='test',condition='with_negatives',seed=s,**{m:(np.nan if s==122 and m=='PCC' else .5) for m in ev.METRICS}) for s in sorted(ev.SEEDS)]
        pcc=ev.summarize(pd.DataFrame(rows)).set_index('metric').loc['PCC']
        self.assertTrue(np.isnan(pcc['mean']))
        self.assertEqual(pcc.n_defined,9)

    def test_undefined_holm_slot_is_retained(self):
        np.testing.assert_allclose(ev.holm([.01,.03,np.nan]),[.03,.06,np.nan],equal_nan=True)

    def test_pareto_directions_and_ties(self):
        d=pd.DataFrame(dict(method=['a','b','c','d'],time_s_with_struct=[1,2,3,2],PCC=[.2,.4,.3,.4],RSE=[2,1,1.5,1]))
        self.assertEqual(set(ev.pareto(d,'PCC').method),{'a','b','d'})
        self.assertEqual(set(ev.pareto(d,'RSE').method),{'a','b','d'})

    def test_wrong_units_zero_and_reversed_values_fail_reference(self):
        ref=pd.DataFrame(dict(method=['m']*3,seed=[120,121,122],PCC=[.1,.2,.4]))
        for values in [[0,0,0],[.4,.2,.1],[.1*np.log10(2),.2*np.log10(2),.4*np.log10(2)]]:
            with self.subTest(values=values),self.assertRaises(ValueError):
                ev.compare(ref.assign(PCC=values),ref,['method','seed'],['PCC'])


if __name__=='__main__':
    unittest.main()
