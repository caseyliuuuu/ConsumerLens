import io
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.preprocessing import preprocess
from src.topic_modeling import discover_topics
from src.sentiment import analyze_sentiment
from src.trend_analysis import analyze_trends
from src.impact_scoring import score_topics
from src.insight_agent import build_structured_evidence, generate_insights

class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.data,_=preprocess(pd.read_csv(Path(__file__).resolve().parents[1]/'data/sample_reviews.csv'))
        cls.data,cls.topics,_=discover_topics(cls.data,'offline');cls.data,_=analyze_sentiment(cls.data,False)
        cls.trends,cls.meta=analyze_trends(cls.data);cls.scores=score_topics(cls.data,cls.trends)
    def test_quality_aliases(self):
        d,q=preprocess(pd.DataFrame({'comment':[' Good! ','good!',None,'123','No'],'stars':[5,5,3,2,9]}))
        self.assertEqual(len(d),2);self.assertEqual(q['duplicates_removed'],1);self.assertEqual(q['invalid_ratings'],1)
    def test_missing_and_small(self):
        for texts in [['ok'],['a','the'],['great','bad','fine']]:
            d,_=preprocess(pd.DataFrame({'text':texts}));d,t,_=discover_topics(d,'offline');d,_=analyze_sentiment(d,False)
            tr,meta=analyze_trends(d);s=score_topics(d,tr)
            self.assertTrue(d.sentiment_label.eq('Unknown').all());self.assertFalse(meta['available']);self.assertTrue(s.impact_score.between(0,10).all())
    def test_model_failure_paths(self):
        with patch('src.topic_modeling.embedding_model',side_effect=RuntimeError('offline')):
            d,t,m=discover_topics(self.data.head(20))
            self.assertIn('TF-IDF',m['method']);self.assertTrue(m['warnings'])
        with patch('src.sentiment.sentiment_model',side_effect=RuntimeError('offline')):
            d,m=analyze_sentiment(self.data.head(20))
            self.assertTrue(d.sentiment_method.eq('Rating proxy').all())
    def test_embedding_kmeans_fallback(self):
        import numpy as np
        with patch('src.topic_modeling.embedding_model') as model:
            model.return_value.encode.return_value=np.random.default_rng(42).normal(size=(20,6))
            with patch.dict(sys.modules,{'bertopic':None}):
                d,t,m=discover_topics(self.data.head(20))
                self.assertEqual(m['method'],'MiniLM + KMeans');self.assertEqual(len(t),5)
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):preprocess(pd.DataFrame({'wrong':['hi']}))
        d,_=preprocess(pd.DataFrame({'text':[None,' ']}))
        with self.assertRaises(ValueError):discover_topics(d,'offline')
    def test_quantitative_invariants(self):
        self.assertAlmostEqual(self.scores.share_pct.sum(),100)
        self.assertAlmostEqual(self.trends.change_pp.sum(),0)
        self.assertTrue(self.scores.impact_score.between(0,10).all())
        self.assertAlmostEqual(self.scores[['positive_pct','neutral_pct','negative_pct','unknown_pct']].sum(axis=1).min(),100)
    def test_known_trend(self):
        d=pd.DataFrame({'topic_id':[0,0,0,1],'date':pd.to_datetime(['2026-01-01','2026-01-01','2026-02-01','2026-02-01'],utc=True)})
        t,_=analyze_trends(d);self.assertEqual(t.set_index('topic_id').loc[1,'change_pp'],50)
    def test_grounding_and_api_fallback(self):
        brief=generate_insights(self.data,self.scores,self.topics)
        self.assertEqual(brief['method'],'Deterministic evidence-grounded brief')
        for c in brief['insights']:self.assertTrue(set(c['review_ids'])<=set(self.data.review_id))
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test'}):
            with patch('urllib.request.urlopen',side_effect=TimeoutError):
                result=generate_insights(self.data,self.scores,self.topics,True)
                self.assertEqual(result['method'],'Deterministic evidence-grounded brief')
                self.assertIn('grounding validation',result['warning'])
    def test_llm_receives_structured_evidence_and_valid_citations(self):
        evidence=build_structured_evidence(self.data,self.scores,self.topics)
        self.assertEqual(set(evidence),{'analysis_scope','topics'})
        allowed_topic_keys={'topic_id','topic_name','impact_rank','review_count','share_of_voice_pct','sentiment','trend','purchase_relevance','business_impact','evidence_reviews'}
        self.assertTrue(all(set(topic)==allowed_topic_keys for topic in evidence['topics']))
        selected=evidence['topics'][0]
        review_id=selected['evidence_reviews'][0]['review_id']
        recommendations=[]
        for category in ['Product','Pricing','Marketing']:
            recommendations.append({'category':category,'topic_id':selected['topic_id'],
                'finding':'The supplied reviews describe friction that merits validation.',
                'business_implication':'The pattern may affect the experience represented in this dataset and warrants review.',
                'recommended_action':'Inspect the cited experience and test a focused response with affected reviewers.',
                'evidence_review_ids':[review_id]})
        parsed={'executive_synthesis':'The supplied evidence suggests a focused issue that merits coordinated validation across teams.',
                'recommendations':recommendations}
        response={'choices':[{'message':{'content':json.dumps(parsed)}}]}
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test'}):
            with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
                result=generate_insights(self.data,self.scores,self.topics,True)
        self.assertEqual(result['method'],'LLM-assisted evidence-grounded brief')
        self.assertEqual({item['category'] for item in result['insights']},{'Product','Pricing','Marketing'})
        self.assertTrue(all(item['review_ids']==[review_id] for item in result['insights']))
    def test_llm_rejects_unknown_citations_and_numeric_claims(self):
        evidence=build_structured_evidence(self.data,self.scores,self.topics)
        selected=evidence['topics'][0]
        invalid={'executive_synthesis':'This will increase revenue by 20 percent across the market.',
                 'recommendations':[{'category':category,'topic_id':selected['topic_id'],
                    'finding':'A supplied review supports investigation of this topic.',
                    'business_implication':'The evidence is limited and should be validated.',
                    'recommended_action':'Review the evidence before making a change.',
                    'evidence_review_ids':['R999999']} for category in ['Product','Pricing','Marketing']]}
        response={'choices':[{'message':{'content':json.dumps(invalid)}}]}
        with patch.dict(os.environ,{'OPENAI_API_KEY':'test'}):
            with patch('urllib.request.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
                result=generate_insights(self.data,self.scores,self.topics,True)
        self.assertEqual(result['method'],'Deterministic evidence-grounded brief')
        self.assertIn('grounding validation',result['warning'])

if __name__=='__main__':unittest.main()
