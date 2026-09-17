"""Recruiter flow and uploaded-file presentation regressions (no model downloads)."""
import io
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import pandas as pd
from streamlit.testing.v1 import AppTest
from streamlit.runtime.uploaded_file_manager import UploadedFile, UploadedFileRec
from streamlit.proto.Common_pb2 import FileURLs

ROOT = Path(__file__).resolve().parents[1]

class DashboardTests(unittest.TestCase):
    def test_demo_explorer_and_insights_export(self):
        app = AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
        self.assertFalse(app.exception)
        app.button(key='demo_landing').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.session_state.result[0]),397)
        self.assertEqual(app.session_state.result[3]['method'],'MiniLM + BERTopic')
        self.assertEqual([t.label for t in app.tabs],['Overview','Topic Explorer','Recommendations','Methodology'])
        select=next(s for s in app.selectbox if s.label=='Select a topic')
        select.select(select.options[-1]).run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any('Business Impact Score' in m.label for m in app.metric))
        next(b for b in app.button if b.label=='Generate recommendation brief').click().run(timeout=30)
        self.assertFalse(app.exception)
        self.assertTrue(any(m.label=='Investigation rank' for m in app.metric))
        self.assertEqual(app.session_state.insights['method'],'Deterministic evidence-grounded brief')
        self.assertEqual(len(app.get('download_button')),4)
        from src.exporting import insight_export
        output=insight_export(app.session_state.insights,app.session_state.result[0],'Synthetic demo',True)
        exported=pd.read_csv(io.BytesIO(output))
        self.assertTrue(exported.synthetic_data.all())
        self.assertTrue(exported.review_ids.notna().all())
        self.assertTrue({'Product','Pricing','Marketing'}<=set(exported.category))
        self.assertTrue(exported.supporting_reviews.str.contains('R000').all())
    def test_optional_fields_and_invalid_uploads(self):
        for filename,payload,expected in [('minimal.csv',b'text\nGood battery\nPoor screen\n',2),('empty.csv',b'text\n',0),('wrong.csv',b'wrong\nx\n',0)]:
            with self.subTest(filename=filename):
                upload=UploadedFile(UploadedFileRec('test',filename,'text/csv',payload),FileURLs())
                with patch('streamlit.file_uploader',return_value=upload):
                    app=AppTest.from_file(str(ROOT/'app.py')).run(timeout=30)
                    next(s for s in app.selectbox if s.label=='Topic discovery').select('Offline · TF-IDF + KMeans')
                    next(c for c in app.checkbox if c.label=='Use text sentiment model').uncheck()
                    next(b for b in app.button if b.label=='Analyze uploaded CSV').click().run(timeout=30)
                    self.assertFalse(app.exception)
                    if expected:
                        self.assertEqual(len(app.session_state.result[0]),expected)
                        self.assertFalse(app.session_state.result[6]['available'])
                        self.assertFalse(app.session_state.is_demo)
                    else:self.assertTrue(app.error)

if __name__=='__main__':unittest.main()
