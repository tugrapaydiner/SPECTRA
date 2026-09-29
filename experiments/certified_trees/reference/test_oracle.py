"""Explicitly synthetic correctness tests, never empirical accuracy evidence."""
from __future__ import annotations
import copy
import itertools
import json
import math
import random
import unittest
from fractions import Fraction as F
import certificate_oracle as co


def source(leaves, *, d=1, classes=1, bias=None, scale=1.0, thresholds=None):
    trees=[]
    for k, values in enumerate(leaves):
        if thresholds is None:
            splits=[{'split_type':'FloatFeature','float_feature_index':k % d,'border':0.5}]
        else:
            splits=[{'split_type':'FloatFeature','float_feature_index':f,'border':b}
                    for f,b in thresholds[k]]
        trees.append({'splits':splits,'leaf_values':values})
    return co.canonical({'features_info':{'float_features':[
        {'feature_index':f,'flat_feature_index':f,'borders':[]} for f in range(d)]},
        'scale_and_bias':[scale,bias if bias is not None else [0.0]*classes],
        'oblivious_trees':trees})


class OracleContracts(unittest.TestCase):
    def test_rounding(self):
        for numerator in range(-100,101):
            for denominator in range(1,20):
                value=F(numerator,denominator)
                self.assertEqual(co.nearest_even(value),round(value))
                self.assertLessEqual(co.floor_fraction(value),value)
                self.assertGreaterEqual(co.ceil_fraction(value),value)

    def test_power_of_two_selection(self):
        for n in range(1,100):
            value=F(n,37); exponent=co.ceil_log2(value)
            self.assertGreaterEqual(co.power2(exponent),value)
            self.assertLess(co.power2(exponent-1),value)

    def test_wrong_8bit_answer_is_refused(self):
        raw=source([[0.125,100.0]])
        compact=co.compile_source(raw,1,bits=8,pairwise=True)
        oracle=co.CertifiedOracle(raw,compact)
        result=oracle.predict([0],checkpoint=0)
        self.assertEqual(result['approximate_class_index'],0)
        self.assertEqual(co.source_scores(oracle.source,[0]),[0.0,0.125])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertIsNone(result['class_index'])
        result16=co.CertifiedOracle(raw,co.compile_source(raw,1,bits=16,pairwise=True)).predict([0])
        self.assertEqual(result16['status'],'CERTIFIED')
        self.assertEqual(result16['class_index'],1)

    def test_exhaustive_domain_against_separate_source_sum(self):
        rng=random.Random(731)
        for classes in (1,2,3,5):
            for bits in (8,16):
                for repetition in range(5):
                    # Different sizes and close competing leaves, with all domain
                    # rows checked, not selected from classifier correctness.
                    leaves=[[rng.uniform(-4,4) for _ in range(4*classes)] for t in range(7)]
                    split=[[(0,0.5),(1,1.5)] for _ in leaves]
                    bias=[rng.uniform(-.2,.2) for _ in range(classes)]
                    raw=source(leaves,d=2,classes=classes,bias=bias,scale=1.25,thresholds=split)
                    compact=co.compile_source(raw,3,bits=bits,pairwise=True)
                    oracle=co.CertifiedOracle(raw,compact)
                    for row in itertools.product(range(4),repeat=2):
                        scores=co.source_scores(oracle.source,row)
                        exact=co.source_scores(oracle.source,row,exact=True)
                        winner=max(range(len(scores)),key=lambda c:scores[c])
                        for checkpoint in (0,1,3,16):
                            result=oracle.predict(row,checkpoint=checkpoint)
                            if result['status']=='CERTIFIED':
                                self.assertEqual(result['class_index'],winner)
                        # Verify underlying interval algebra independently of settle.
                        totals=list(compact['bias'])
                        for tree in compact['trees']:
                            index=sum((1<<b) for b,(f,t) in enumerate(zip(tree['features'],tree['thresholds'])) if row[f]>t)
                            totals=[a+b for a,b in zip(totals,tree['leaves'][index])]
                        step=co.power2(compact['quantization']['step_exponent']); fine=1<<20
                        # Common source offset is exact source class0/scale,
                        # since class0's quantized contrast and bias are zero.
                        common=exact[0]/oracle.source.scale
                        for c,score in enumerate(exact):
                            normalized=(score/oracle.source.scale-common)/step
                            lower=F(totals[c]*fine+compact['error_low'][c],fine)
                            upper=F(totals[c]*fine+compact['error_high'][c],fine)
                            self.assertLessEqual(lower,normalized)
                            self.assertGreaterEqual(upper,normalized)

    def test_pairwise_never_weakens_completed_box_certificate(self):
        raw=source([[1.1,-2.2,3.3,-.4,2.3,1.2],[1.4,.2,-.8,-.2,.9,1.2]],classes=3)
        oracle=co.CertifiedOracle(raw,co.compile_source(raw,1,bits=8,pairwise=True))
        for row in ([0],[1]):
            box=oracle.predict(row,checkpoint=0,use_pairwise=False)
            pair=oracle.predict(row,checkpoint=0,use_pairwise=True)
            if box['status']=='CERTIFIED':
                self.assertEqual(pair['class_index'],box['class_index'])

    def test_early_stop(self):
        leaves=[[.01,-.01] for _ in range(40)]
        raw=source(leaves,bias=[10.0])
        oracle=co.CertifiedOracle(raw,co.compile_source(raw,1,bits=16))
        result=oracle.predict([0],checkpoint=8)
        self.assertEqual(result['class_index'],1)
        self.assertLess(result['trees_evaluated'],result['total_trees'])
        self.assertIsNone(result['approximate_class_index'])

    def test_duplicate_feature_predicates_and_constant_splits(self):
        raw=source([[float(i-3) for i in range(16)]],
                   thresholds=[[(0,-2.0),(0,1.0),(0,1.0),(0,20.0)]])
        oracle=co.CertifiedOracle(raw,co.compile_source(raw,3,bits=16))
        for row in ([0],[1],[2],[3]):
            result=oracle.predict(row,checkpoint=0)
            if result['status']=='CERTIFIED':
                scores=co.source_scores(oracle.source,row)
                self.assertEqual(result['class_index'],max(range(2),key=lambda c:scores[c]))

    def test_compact_mutations_rejected(self):
        raw=source([[.125,100.]])
        base=co.compile_source(raw,1,bits=16,pairwise=True)
        for damage in ('hash','leaf','lower','upper','pair','domain','bits','contract'):
            obj=copy.deepcopy(base)
            if damage=='hash':obj['source_sha256']='0'*64
            elif damage=='leaf':obj['trees'][0]['leaves'][0][1]+=1
            elif damage=='lower':obj['error_low'][1]+=1
            elif damage=='upper':obj['error_high'][1]-=1
            elif damage=='pair':obj['pair_upper'][0][1]-=1
            elif damage=='domain':obj['domain']['maximum']=0
            elif damage=='bits':obj['quantization']['bound_fraction_bits']=1
            else:obj['contract']['rounding']='anything'
            with self.subTest(damage=damage),self.assertRaises(ValueError):
                co.verify_compiled(raw,obj)

    def test_bad_input_and_json(self):
        raw=source([[.125,100.]])
        oracle=co.CertifiedOracle(raw,co.compile_source(raw,1,bits=16))
        for query in ([True],[-1],[2],[.5],[],[0,1],'0'):
            with self.subTest(query=query),self.assertRaises(ValueError):oracle.predict(query)
        for raw_bad in (b'{"x":1,"x":2}',b'[NaN]',b'[1e999]',b'\xff'):
            with self.assertRaises(ValueError):co.loads(raw_bad)
        with self.assertRaises(ValueError):oracle.predict([0],checkpoint=True)

    def test_unsupported_source_rejected(self):
        raw=source([[.125,100.]])
        obj=json.loads(raw)
        for damage in ('scale','nan','split','leafcount','categorical','threshold'):
            val=copy.deepcopy(obj)
            if damage=='scale':val['scale_and_bias'][0]=-1.
            elif damage=='nan':val['oblivious_trees'][0]['leaf_values'][0]='NaN'
            elif damage=='split':val['oblivious_trees'][0]['splits'][0]['split_type']='OnlineCtr'
            elif damage=='leafcount':val['oblivious_trees'][0]['leaf_values'].pop()
            elif damage=='categorical':val['features_info']['categorical_features']=[{}]
            else:val['oblivious_trees'][0]['splits'][0]['border']=.1
            with self.subTest(damage=damage),self.assertRaises(ValueError):
                co.compile_source(co.canonical(val),1)

    def test_roundoff_encloses_different_source_summation_orders(self):
        values=[2.0**40,1.0,-2.0**40,.1]
        # Every tree has a constant pair of leaves; the true arithmetic still
        # includes all four contributions and a nontrivial final scale/bias.
        raw=source([[v,-v,v,-v] for v in values],classes=2,bias=[.2,-.3],scale=1.25)
        src=co.parse_source(raw,1)
        rounding=co._roundoff_bounds(src)
        for row in ([0],[1]):
            real=co.source_scores(src,row,exact=True)
            selected=[t.leaves[co.leaf_index(t.features,t.thresholds,row)] for t in src.trees]
            for order in itertools.permutations(range(len(selected))):
                for c in range(src.classes):
                    total=0.0
                    for t in order:total+=float(selected[t][c])
                    computed=float(src.scale)*total+float(src.bias[c])
                    self.assertLessEqual(abs(F.from_float(computed)-real[c]),rounding[c]*src.scale)

    def test_verified_object_is_copied(self):
        raw=source([[.125,100.]])
        compact=co.compile_source(raw,1,bits=16)
        oracle=co.CertifiedOracle(raw,compact)
        before=oracle.predict([0]);compact['trees'][0]['leaves'][0][1]=0
        self.assertEqual(before,oracle.predict([0]))

    def test_exact_zero_tie_is_not_falsely_certified(self):
        raw=source([[0.0,0.0]])
        oracle=co.CertifiedOracle(raw,co.compile_source(raw,1))
        result=oracle.predict([0])
        self.assertEqual(result['status'],'UNRESOLVED')
        self.assertIsNone(result['class_index'])


if __name__=='__main__':
    unittest.main(verbosity=2)
