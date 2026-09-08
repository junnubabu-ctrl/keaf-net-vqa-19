import tempfile,unittest
from pathlib import Path
from evitrust_vqa.research import *

def p(i,text):return Passage(i,text,'https://example.org/'+i,'family','snapshot-1','CC0')
class ResearchContracts(unittest.TestCase):
 def test_retrieval_depends_on_visual_context(self):
  r=BM25([p('a','giraffe eats leaves'),p('b','penguin eats fish')])
  self.assertEqual(r.retrieve('what does it eat','penguin')[0][1].evidence_id,'b')
  self.assertEqual(r.retrieve('what does it eat','giraffe')[0][1].evidence_id,'a')
 def test_snapshot_content_changes_identity(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'config.json';f.write_text('one')
   a=snapshot_fingerprint(d)['sha256'];f.write_text('two')
   self.assertNotEqual(a,snapshot_fingerprint(d)['sha256'])
 def test_cache_configuration_changes_identity(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'im';f.write_bytes(b'p');q=Query('q','why',str(f));e=[p('a','fact')]
   args=[q,e,'model','prompt',{'temperature':0},17,{'pixels':256}]
   a=prediction_key(*args)
   for index,value in [(2,'other'),(3,'other'),(4,{'temperature':1}),(5,42),(6,{'pixels':512})]:
    changed=args.copy();changed[index]=value
    self.assertNotEqual(a,prediction_key(*changed))
 def test_empty_retrieval(self):self.assertEqual(BM25([]).retrieve('question',''),[])
 def test_duplicates_do_not_increase_support(self):
  out,aliases=deduplicate([p('a','Paris is French.'),p('b','Paris is French!')])
  self.assertEqual(len(out),1);self.assertEqual(aliases['a'],['a','b'])
 def test_exact_cache_identity(self):
  with tempfile.TemporaryDirectory() as d:
   f=Path(d)/'image';f.write_bytes(b'pixels1');q=Query('q','where',str(f))
   def key(e):return prediction_key(q,e,'rev','prompt',{'do_sample':False},1,{})
   a=key([p('a','Paris')]);self.assertNotEqual(a,key([p('a','London')]))
   f.write_bytes(b'pixels2');self.assertNotEqual(a,key([p('a','Paris')]))
 def test_ties_not_split(self):
  c=risk_curve([.8,.8],[0,1]);self.assertEqual(len(c),2)
  self.assertIsNone(c[0]['risk']);self.assertEqual(c[-1]['risk'],.5)
 def test_gate_aware_fit(self):
  self.assertEqual(fit_release_threshold([.9,.8],[1,0],[False,True],0),.8)
  self.assertIsNone(fit_release_threshold([.8,.8],[0,1],[True,True],0))
 def test_hard_grounding(self):
  self.assertIsNone(release('Paris',1,0,True,.5)['answer'])
 def test_partition_leakage(self):
  with self.assertRaises(ValueError):verify_disjoint({'fit':['im1'],'test':['im1']})
 def test_invalid_scores(self):
  with self.assertRaises(ValueError):risk_curve([float('nan')],[0])
 def test_no_calibration_means_no_release(self):
  self.assertIsNone(release('Paris',1,1,True,None)['answer'])
 def test_unresolved_conflict_withholds_both(self):
  class N:
   def score(self,a,b):return {'contradiction':.9,'entailment':.05,'neutral':.05}
  out,trace=select_consistent([p('a','capital is Paris'),p('b','capital is London')],N())
  self.assertEqual(out,[]);self.assertEqual(len(trace['conflicts']),1)
if __name__=='__main__':unittest.main()
