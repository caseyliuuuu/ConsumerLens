"""ConsumerLens: an evidence-first consumer intelligence workspace."""
from pathlib import Path
import hashlib
import html
import json
import textwrap
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv
from src.pipeline import analyze
from src.impact_scoring import WEIGHTS
from src.insight_agent import generate_insights
from src.exporting import csv_bytes, insight_export

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
st.set_page_config(page_title='ConsumerLens | Consumer Intelligence', page_icon='◉', layout='wide')
COLORS = {'Positive': '#168773', 'Neutral': '#a8b5c8', 'Negative': '#de735d', 'Unknown': '#d9dee7'}
IMPACT_HELP = ('A 0–10 investigation priority index: 35% prevalence, 30% negative intensity, '
               '20% purchase relevance and 15% trend growth. Missing components are excluded '
               'and remaining weights rescaled. Not a revenue or ROI estimate.')
CSS = '''<style>
.stApp {background:#f5f7fb;color:#20334d;}
[data-testid="stHeader"] {background:rgba(245,247,251,.96);}
.block-container {max-width:1500px;padding-top:3.6rem;padding-bottom:3rem;}
h1,h2,h3 {color:#12243e;letter-spacing:-.035em;} h1 {font-weight:750!important;font-size:2rem!important;line-height:1.2!important;}
[data-testid="stSidebar"] {background:#fff;border-right:1px solid #e4e9f2;}
[data-testid="stMetric"] {background:#fff;border:1px solid #e1e7f0;border-radius:14px;padding:19px 21px;}
[data-testid="stMetricValue"] {font-weight:650;letter-spacing:-.04em;}
[data-testid="stMetricLabel"] {color:#64748b;font-size:.85rem;}
[data-testid="stVerticalBlockBorderWrapper"]>div {border-radius:14px!important;}
[data-baseweb="tab-list"] {gap:28px;border-bottom:1px solid #dfe6ef;margin:12px 0 18px;}
[data-baseweb="tab"] {font-weight:600;}
button[kind="primary"] {background:#3569dc!important;border-color:#3569dc!important;color:white!important;}
.stButton>button,.stDownloadButton>button {border-radius:9px;font-weight:600;min-height:42px;}
[data-testid="stExpander"] {background:white;border-radius:10px;}
.eyebrow {font-size:11px;letter-spacing:.14em;font-weight:700;color:#61738e;text-transform:uppercase;margin-bottom:10px;}
.brand {font-size:25px!important;font-weight:750;color:#162b49;letter-spacing:-1px;margin-bottom:0;}
.brand-mark {color:#3b6eea;margin-right:7px;}
.muted {color:#687a92;font-size:13px;line-height:1.6;}
.hero-title {font-size:clamp(34px,4vw,55px);line-height:1.12;letter-spacing:-.045em;font-weight:750;color:#142742;margin:8px 0 20px;max-width:790px;}
.hero-copy {font-size:18px;line-height:1.65;color:#61738e;max-width:720px;margin-bottom:25px;}
.feature {background:#fff;border:1px solid #e1e7f0;border-radius:14px;padding:25px;min-height:178px;margin-top:15px;}
.feature .step {color:#356ee6;font-size:12px;font-weight:750;letter-spacing:.1em;}
.feature h3 {font-size:20px;margin:12px 0 10px;}
.feature p {font-size:14px;line-height:1.6;color:#667891;margin:0;}
.badge {display:inline-block;border:1px solid #dce5f0;border-radius:50px;padding:5px 11px;font-size:11px;font-weight:700;letter-spacing:.045em;background:#fff;color:#526984;}
.synthetic {background:#fff6e7;border-color:#f1d8a2;color:#946013;}
.priority {background:#152b49;color:#fff;border-radius:15px;padding:25px 28px;margin:22px 0 25px;}
.priority .eyebrow {color:#9db8e5;margin-bottom:9px;}
.priority h3 {color:#fff;font-size:24px;letter-spacing:-.02em;margin:0 0 10px;}
.priority p {color:#c8d5e8;font-size:14px;line-height:1.65;margin:0;}
.chip {display:inline-block;padding:5px 10px;margin:3px 5px 3px 0;background:#eef3fa;border-radius:7px;color:#496381;font-size:12px;}
.action-tag {font-weight:750;font-size:11px;letter-spacing:.09em;padding:6px 10px;border-radius:6px;display:inline-block;margin-bottom:14px;}
.product {background:#eaf0ff;color:#315ecc;}.pricing {background:#e6f4ee;color:#217155;}.marketing {background:#f1eafa;color:#8050a7;}.signal {background:#fff1e5;color:#a0642c;}
.section-copy {color:#687a92;font-size:14px;margin-top:-10px;margin-bottom:18px;}
@media (max-width:760px) {.block-container {padding-left:1rem;padding-right:1rem;} .priority {padding:20px;} [data-baseweb="tab-list"] {gap:12px;}}
</style>'''
st.markdown(CSS, unsafe_allow_html=True)


def chart(fig, height=350):
    """Apply a consistent presentation without changing underlying metrics."""
    fig.update_layout(template='plotly_white', font=dict(family='Arial', color='#49617e', size=12),
                      height=height, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                      margin=dict(l=10, r=15, t=25, b=20), legend_title_text='',
                      hoverlabel=dict(bgcolor='#152b49', font_color='white'))
    st.plotly_chart(fig, width='stretch', config={'displaylogo': False, 'toImageButtonOptions': {'scale': 2}})


@st.cache_data(show_spinner=False, max_entries=4, ttl=3600)
def run_analysis(payload: bytes, mode: str, text_sentiment: bool):
    return analyze(payload, mode, text_sentiment)


@st.cache_data(show_spinner=False)
def load_demo(snapshot_bytes: bytes, sample_bytes: bytes, label_bytes: bytes):
    """Load a computed semantic snapshot only when it matches the bundled CSV."""
    saved = json.loads(snapshot_bytes)
    if saved['schema_version'] != 1 or saved['sample_sha256'] != hashlib.sha256(sample_bytes).hexdigest():
        raise ValueError('Demo snapshot does not match sample data. Rebuild it with evaluation/build_demo.py.')
    data = pd.DataFrame(saved['data'])
    data['date'] = pd.to_datetime(data.date, utc=True)
    display_labels = json.loads(label_bytes)
    topics = {int(k): {**v, 'model_label': v['label'], 'label': display_labels.get(v['label'], v['label'])} for k, v in saved['topics'].items()}
    data['topic'] = data.topic_id.map(lambda t: topics[t]['label'])
    scores = pd.DataFrame(saved['scores'])
    scores['topic'] = scores.topic_id.map(lambda t: topics[t]['label'])
    return (data, saved['quality'], topics, saved['topic_meta'], saved['sentiment_meta'],
            pd.DataFrame(saved['trends']), saved['trend_meta'], scores)


def review_card(row):
    source = str(row.source) if pd.notna(row.source) else 'Source unavailable'
    st.caption(f'{row.review_id}  ·  {row.sentiment_label}  ·  {source}')
    st.text(row.review_text)


def section(title, caption):
    st.subheader(title)
    st.caption(caption)


with st.sidebar:
    st.markdown('<p class="brand"><span class="brand-mark">◉</span>ConsumerLens</p><p class="muted">Consumer intelligence workspace</p>', unsafe_allow_html=True)
    st.divider()
    demo_sidebar = st.button('Try Demo Data', key='demo_sidebar', type='primary', width='stretch')
    st.caption('400 synthetic reviews · No API key needed')
    st.divider()
    st.markdown('**Analyze your reviews**')
    upload = st.file_uploader('Upload CSV', type=['csv'], help='Required: review_text (or review, text, comment, content). Optional: rating, date, product, source.')
    with st.expander('Analysis settings'):
        mode_label = st.selectbox('Topic discovery', ['Semantic · BERTopic + fallback', 'Semantic · KMeans', 'Offline · TF-IDF + KMeans'])
        use_text = st.checkbox('Use text sentiment model', value=True)
        st.caption('Semantic analysis downloads public models on first use. For offline analysis, choose TF-IDF and turn off text sentiment.')
    mode = {'Semantic · BERTopic + fallback': 'auto', 'Semantic · KMeans': 'kmeans', 'Offline · TF-IDF + KMeans': 'offline'}[mode_label]
    run = st.button('Analyze uploaded CSV', disabled=upload is None, width='stretch')
    st.caption('Your data stays local unless you explicitly enable the optional AI API.')
    with st.expander('Demo dataset'):
        st.write('Aster Slate 10 is a fictional tablet. Reviews are synthetic and do not represent market research.')
        st.download_button('Download sample CSV', (ROOT / 'data/sample_reviews.csv').read_bytes(), 'sample_reviews.csv', 'text/csv', width='stretch')

landing_demo = False
if 'result' not in st.session_state:
    st.markdown('<div class="eyebrow">AI-powered consumer intelligence</div><div class="hero-title">From customer reviews<br>to your next product decision.</div><div class="hero-copy">Discover what customers care about, prioritize the issues that deserve attention, and trace every recommendation back to review evidence.</div>', unsafe_allow_html=True)
    cta, _ = st.columns([1, 2])
    with cta:
        landing_demo = st.button('Try Demo Data', key='demo_landing', type='primary', width='stretch')
    st.caption('Instant demo · 400 synthetic reviews · No upload or API key required')
    for col, step, title, copy in zip(st.columns(3), ['01 / DISCOVER', '02 / PRIORITIZE', '03 / ACT'],
            ['Find the recurring themes', 'See what matters most', 'Make an evidence-backed call'],
            ['Semantic AI groups reviews into topics and measures how customers feel.', 'A transparent 0–10 index combines reach, negative intensity, purchase relevance and growth.', 'Explore Product, Pricing and Marketing actions with the exact supporting reviews.']):
        with col:
            st.markdown(f'<div class="feature"><div class="step">{step}</div><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
    st.markdown('<br><span class="badge synthetic">SYNTHETIC DEMO</span> <span class="muted">Fictional product. Real analytical pipeline. No claims about real consumers.</span>', unsafe_allow_html=True)
    with st.expander('How the analysis works'):
        st.write('Validate reviews → discover semantic topics → measure sentiment → rank business priorities → link actions to evidence. All counts and scores are calculated in Python. The optional AI API cannot invent metrics.')

if demo_sidebar or landing_demo or run:
    st.session_state.pop('result', None)
    st.session_state.pop('insights', None)
    try:
        demo = demo_sidebar or landing_demo
        with st.spinner('Opening the demo…' if demo else 'Analyzing customer reviews…'):
            if demo:
                result = load_demo((ROOT / 'data/demo_analysis.json').read_bytes(), (ROOT / 'data/sample_reviews.csv').read_bytes(), (ROOT / 'data/demo_labels.json').read_bytes())
                dataset = 'Aster Slate 10 · Synthetic demo'
            else:
                result = run_analysis(upload.getvalue(), mode, use_text)
                dataset = upload.name
            st.session_state.result = result
            st.session_state.dataset_name = dataset
            st.session_state.is_demo = bool(demo)
            st.session_state.insights = generate_insights(result[0], result[-1], result[2], False)
        st.rerun()
    except Exception as exc:
        st.error(f'Unable to complete analysis: {exc}. Check the CSV columns or try offline analysis.')
if 'result' not in st.session_state:
    st.stop()

data, quality, topics, topic_meta, sentiment_meta, trends, trend_meta, scores = st.session_state.result
if 'insights' not in st.session_state:
    st.session_state.insights = generate_insights(data, scores, topics, False)
brief = st.session_state.insights
synthetic = st.session_state.get('is_demo', False) or data['product'].astype(str).str.contains('SYNTHETIC', case=False).any()
head, export = st.columns([3, 1], vertical_alignment='bottom')
with head:
    st.markdown('<div class="eyebrow">ConsumerLens / Executive workspace</div>', unsafe_allow_html=True)
    st.title('Customer voices. Clear priorities.')
    st.caption(st.session_state.dataset_name)
with export:
    st.download_button('Export Insights', insight_export(brief, data, st.session_state.dataset_name, bool(synthetic)), 'consumerlens_insights.csv', 'text/csv', type='primary', width='stretch', help='Download findings, actions, calculated evidence and supporting review citations as CSV.')
if synthetic:
    st.markdown('<span class="badge synthetic">SYNTHETIC DEMO</span> <span class="muted">Fictional product and reviews. These findings are a demonstration, not market research.</span>', unsafe_allow_html=True)
    st.write('')
for message in topic_meta['warnings'] + sentiment_meta['warnings']:
    st.warning(message)

metrics = st.columns(4)
metrics[0].metric('Reviews analyzed', f'{len(data):,}', help=f"{quality['uploaded']:,} uploaded; empty, duplicate and noise-only reviews removed.")
metrics[1].metric('Topics identified', len(scores), help='Review groups identified by the active topic method. Includes the unassigned / mixed group when present.')
metrics[2].metric('Positive reviews', f"{data.sentiment_label.eq('Positive').mean()*100:.1f}%", help='Share of all valid reviews classified positive. Unknown reviews remain in the denominator.')
metrics[3].metric('Negative reviews', f"{data.sentiment_label.eq('Negative').mean()*100:.1f}%", help='Share of all valid reviews classified negative. Review-level sentiment, not aspect sentiment.')

coverage = data.sentiment_label.ne('Unknown').mean()*100
st.caption(f'{coverage:.0f}% sentiment coverage  ·  {topic_meta["method"]}  ·  {sentiment_meta["method"]}' + ('  ·  Precomputed demo result' if st.session_state.get('is_demo') else ''))
overview, explorer, recommendations, methodology = st.tabs(['Overview', 'Topic Explorer', 'Recommendations', 'Methodology'])

with overview:
    top = scores.iloc[0]
    st.markdown(f'<div class="priority"><div class="eyebrow">#1 Investigation Priority</div><h3>{html.escape(top.topic)}</h3><p><b>Rank 1 of {len(scores)} topics</b> &nbsp; · &nbsp; Impact Score: {top.impact_score:.2f} / 10 &nbsp; · &nbsp; {int(top.review_count)} reviews &nbsp; · &nbsp; {top.negative_pct:.1f}% negative.<br>This is the highest relative priority in this dataset; the score does not imply high absolute business impact.</p></div>', unsafe_allow_html=True)
    left, right = st.columns([1, 1.6], gap='large')
    with left:
        with st.container(border=True):
            section('How customers feel', 'Sentiment across all valid reviews')
            dist = data.sentiment_label.value_counts().rename_axis('Sentiment').reset_index(name='Reviews')
            fig = px.pie(dist, names='Sentiment', values='Reviews', color='Sentiment', color_discrete_map=COLORS, hole=.72)
            fig.update_traces(textinfo='percent', textposition='outside', hovertemplate='%{label}: %{value} reviews (%{percent})<extra></extra>')
            fig.update_layout(legend=dict(orientation='h', y=-.13, x=.5, xanchor='center'), annotations=[dict(text=f'<b>{len(data):,}</b><br>reviews', x=.5, y=.5, showarrow=False, font_size=20)])
            chart(fig)
    with right:
        with st.container(border=True):
            section('What customers talk about', 'Most discussed topics · share of all reviews')
            most = scores.nlargest(8, 'share_pct').sort_values('share_pct').copy()
            most['display_topic'] = most.apply(lambda r: f'{r.topic_id} · {textwrap.shorten(r.topic, width=31, placeholder="…")}', axis=1)
            fig = px.bar(most, x='share_pct', y='display_topic', orientation='h', text=most.share_pct.map(lambda n: f'{n:.1f}%'), custom_data=['topic', 'review_count'], color_discrete_sequence=['#4775da'], labels={'share_pct': 'Share of reviews (%)', 'display_topic': ''})
            fig.update_traces(textposition='outside', cliponaxis=False, hovertemplate='%{customdata[0]}<br>%{customdata[1]} reviews · %{x:.1f}% share<extra></extra>')
            fig.update_layout(xaxis_range=[0, max(most.share_pct)*1.2])
            chart(fig)
    section('Prioritize the next investigation', 'Ranked by Business Impact Score. Higher means a stronger signal to investigate, not guaranteed financial impact.')
    st.dataframe(scores[['topic', 'review_count', 'share_pct', 'negative_pct', 'impact_score', 'trend', 'change_pp']], hide_index=True, width='stretch', column_config={
        'topic': st.column_config.TextColumn('Consumer topic', width='large'),
        'review_count': st.column_config.NumberColumn('Reviews', format='%d'),
        'share_pct': st.column_config.NumberColumn('Share of voice', format='%.1f%%', help='Topic reviews divided by all valid reviews.'),
        'negative_pct': st.column_config.NumberColumn('Negative', format='%.1f%%'),
        'impact_score': st.column_config.ProgressColumn('Business Impact', min_value=0, max_value=10, format='%.2f / 10', help=IMPACT_HELP),
        'trend': st.column_config.TextColumn('Direction'),
        'change_pp': st.column_config.NumberColumn('Share change (pp)', format='%+.1f', help='Later minus earlier topic share among dated reviews, in percentage points.')})
    with st.container(border=True):
        section('Where reach meets dissatisfaction', 'Topics toward the upper right affect more reviews and attract more negative sentiment. Bubble size shows impact score.')
        fig = px.scatter(scores, x='share_pct', y='negative_pct', size='impact_score', color='impact_score', hover_name='topic', custom_data=['review_count'], size_max=44, range_color=[0, 10], color_continuous_scale=['#bcd9f0', '#2f5fcb'], labels={'share_pct': 'Share of all reviews (%)', 'negative_pct': 'Negative reviews within topic (%)', 'impact_score': 'Impact / 10'})
        fig.update_traces(text=[textwrap.shorten(r.topic, width=27, placeholder='…') if i<3 else '' for i,r in scores.iterrows()], mode='markers+text', textposition='top center', textfont_size=11)
        fig.update_layout(yaxis_range=[-5, 115])
        chart(fig, 390)
    if trend_meta['available']:
        section('Signals gaining attention', 'Growth in topic share across two comparable time windows; increasing attention does not prove a worsening experience.')
        st.caption(f"Earlier: {trend_meta['earlier_n']} dated reviews · Later: {trend_meta['later_n']} · Split date: {trend_meta['split'][:10]} · Emerging ≥ +2 percentage points")
        if trend_meta['low_sample']:
            st.warning('One period has fewer than 30 reviews. Treat these signals as provisional.')
        rising = scores[(scores.change_pp>=2) & (scores.negative_pct>0)].nlargest(3, 'change_pp')
        if rising.empty:
            st.info('No topic with negative reviews grew by at least 2 percentage points in this comparison.')
        else:
            for col, (_, row) in zip(st.columns(len(rising)), rising.iterrows()):
                col.metric(textwrap.shorten(row.topic, width=34, placeholder='…'), f'{row.change_pp:+.1f} pp', help=f'{row.topic} — later minus earlier share among dated reviews.')
        focus_ids = scores.head(5).topic_id
        long = trends[trends.topic_id.isin(focus_ids)].merge(scores[['topic_id','topic']],on='topic_id').melt(id_vars=['topic_id','topic'],value_vars=['earlier_share','later_share'],var_name='Period',value_name='Share of dated reviews (%)')
        long['Period'] = long.Period.map({'earlier_share': 'Earlier period', 'later_share': 'Later period'})
        fig = px.line(long, x='Period', y='Share of dated reviews (%)', color='topic', markers=True, labels={'topic': 'Consumer topic'})
        fig.update_layout(legend=dict(orientation='h', y=-.2, font_size=10))
        chart(fig, 410)
        st.caption('Trend chart shows the five highest-priority groups. All topic trends are included in the topic analysis export.')
    st.divider()
    c1, c2, _ = st.columns([1,1,2])
    c1.download_button('Export topic analysis', csv_bytes(scores), 'consumerlens_topics.csv', 'text/csv', width='stretch')
    c2.download_button('Export annotated reviews', csv_bytes(data), 'consumerlens_reviews.csv', 'text/csv', width='stretch')

with explorer:
    section('Explore the story behind a topic', 'Inspect the score, sentiment mix and representative reviews before making a decision.')
    topic_options = {f"{t} · {topics[t]['label']}": t for t in scores.topic_id.tolist()}
    tid = topic_options[st.selectbox('Select a topic', list(topic_options))]
    selected = scores[scores.topic_id.eq(tid)].iloc[0]
    if 'model_label' in topics[tid]:
        st.caption('Original model keywords: '+topics[tid]['model_label'])
    chips = ''.join(f'<span class="chip">{html.escape(k)}</span>' for k in topics[tid]['keywords'])
    st.markdown(chips, unsafe_allow_html=True)
    st.write('')
    topic_rank = int(scores.reset_index(drop=True).index[scores.reset_index(drop=True).topic_id.eq(tid)][0]) + 1
    cols = st.columns(5)
    cols[0].metric('Investigation rank', f'{topic_rank} of {len(scores)}', help='Relative position after sorting topics by Business Impact Score within this dataset.')
    cols[1].metric('Business Impact Score', f'{selected.impact_score:.2f} / 10', help=IMPACT_HELP)
    cols[2].metric('Share of voice', f'{selected.share_pct:.1f}%', help=f'{int(selected.review_count)} of {len(data)} valid reviews.')
    cols[3].metric('Negative reviews', f'{selected.negative_pct:.1f}%', help='Percentage of all reviews in this topic labeled negative.')
    cols[4].metric('Share change', f'{selected.change_pp:+.1f} pp' if pd.notna(selected.change_pp) else 'Unavailable', help='Later minus earlier share among dated reviews. No estimate when date evidence is insufficient.')
    left, right = st.columns([1, 1.25], gap='large')
    with left:
        with st.container(border=True):
            section('What drives this score?', 'Each component is measured on a 0–10 scale.')
            components = pd.DataFrame([{'Component': label, 'Score': selected[key], 'Weight': f'{WEIGHTS[key]:.0%}'} for key,label in [('prevalence_score','Prevalence'),('negative_intensity_score','Negative intensity'),('purchase_relevance_score','Purchase relevance'),('trend_growth_score','Trend growth')]])
            st.dataframe(components, hide_index=True, width='stretch', column_config={'Score':st.column_config.ProgressColumn('Score / 10',min_value=0,max_value=10,format='%.2f',help=IMPACT_HELP)})
            st.caption(f'Available weight: {selected.available_weight:.0%} · Sentiment coverage: {selected.sentiment_coverage:.0f}%')
            st.caption('Unavailable components are excluded and remaining weights are rescaled. See Methodology for exact definitions.')
        with st.container(border=True):
            st.markdown('**Sentiment within this topic**')
            sentiments = pd.DataFrame({'Sentiment':list(COLORS), 'Share (%)':[selected[f'{s.lower()}_pct'] for s in COLORS]})
            sentiments['Topic'] = 'Reviews'
            fig = px.bar(sentiments, x='Share (%)', y='Topic', color='Sentiment', orientation='h', color_discrete_map=COLORS, text=sentiments['Share (%)'].map(lambda n:f'{n:.0f}%' if n>=5 else ''))
            fig.update_layout(barmode='stack',xaxis_range=[0,100],yaxis_title=None,legend=dict(orientation='h',y=-.55))
            chart(fig, 190)
    with right:
        with st.container(border=True):
            section('The voice of the customer', 'Representative reviews closest to the topic center; examples are not an exhaustive sample.')
            for i, (_, row) in enumerate(data[data.review_id.isin(topics[tid]['representative_ids'])].iterrows()):
                if i: st.divider()
                review_card(row)

with recommendations:
    section('Turn evidence into action', 'Investigation proposals for Product, Pricing and Marketing. Validate these hypotheses before making business changes.')
    with st.container(border=True):
        st.markdown('**Executive summary**')
        st.write(brief['summary'])
        st.markdown(f"**Recommendation mode:** {brief['method']}")
        if brief.get('warning'):
            st.warning(brief['warning'])
    for category, css in [('Product','product'),('Pricing','pricing'),('Marketing','marketing')]:
        items = [i for i in brief['insights'] if i['category']==category]
        with st.container(border=True):
            st.markdown(f'<span class="action-tag {css}">{category.upper()} ACTION</span>', unsafe_allow_html=True)
            if not items:
                st.caption('No action selected for this category. Evidence may be insufficient; no recommendation is inferred.')
            for item in items:
                st.markdown(f"**Topic:** {item['topic']}  ·  Rank {item['rank']} of {len(scores)}")
                st.info(item['evidence'])
                st.markdown('**Finding**')
                st.write(item['finding'])
                st.markdown('**Business implication**')
                st.write(item['implication'])
                st.markdown('**Recommended action**')
                st.write(item['action'])
                with st.expander(f"View supporting evidence · {category} · {len(item['review_ids'])} reviews"):
                    for rid in item['review_ids']:
                        review_card(data[data.review_id.eq(rid)].iloc[0])
    others = [i for i in brief['insights'] if i['category'] not in ('Product','Pricing','Marketing')]
    if others:
        with st.expander('Additional pain points & emerging signals'):
            for item in others:
                st.markdown('**'+item['category']+' — '+item['finding']+'**')
                st.write(item['action'])
                st.caption(item['implication'])
                with st.expander(f"Review evidence · {item['id']}"):
                    st.write(item['evidence'])
                    for rid in item['review_ids']: review_card(data[data.review_id.eq(rid)].iloc[0])
    with st.expander('Optional LLM recommendation layer'):
        st.caption('The deterministic brief works without an API key. Enable the optional LLM to synthesize one Product, Pricing, and Marketing action from structured topic metrics and a limited review-evidence pool. Python validates topic IDs and citations before display.')
        remote = st.checkbox('Use configured LLM API', value=False)
        if st.button('Generate recommendation brief'):
            with st.spinner('Preparing the evidence brief…'):
                st.session_state.insights = generate_insights(data, scores, topics, remote)
            st.rerun()

with methodology:
    section('Evidence before interpretation', 'A short guide to what is measured, how it is calculated, and where judgment is still needed.')
    for col, title, copy in zip(st.columns(3), ['01 · Validate & group','02 · Measure & rank','03 · Recommend & verify'],
            ['Remove empty, noisy and duplicate reviews. Semantic embeddings group similar language into topics.',
             'Calculate topic share, sentiment and date-based growth. Combine four visible components into a priority index.',
             'Attach review citations to every recommendation. Python controls the numbers; the optional LLM synthesizes actions from validated structured evidence.']):
        with col:
            with st.container(border=True):
                st.markdown('**'+title+'**'); st.write(copy)
    st.markdown('**Business Impact Score**')
    st.info('35% Prevalence + 30% Negative Intensity + 20% Purchase Relevance + 15% Trend Growth. Each component ranges from 0 to 10. This is an investigation priority index, not an estimate of revenue impact.')
    with st.expander('Exact score definitions & trend rules'):
        st.markdown('''- **Prevalence:** 10 × topic share of all valid reviews.
- **Negative intensity:** 10 × average negative polarity magnitude among known sentiment, including zeros for non-negative reviews.
- **Purchase relevance:** 10 × share matching commercial terms such as price, value, refund and quality.
- **Trend growth:** positive later-minus-earlier share change in percentage points, capped at 10. Declines score zero.
- **Missing evidence:** omit unavailable components and rescale remaining weights. Check sentiment coverage before comparing.
- **Trends:** split at the midpoint of the date range. Each period uses its own dated-review denominator. ±2 pp defines emerging/declining; this is not a significance test.''')
    st.caption(f"Topic method: {topic_meta['method']} · Sentiment method: {sentiment_meta['method']}")
    if st.session_state.get('is_demo'):
        st.caption('The instant demo is a saved run of the same pipeline against the bundled synthetic CSV. Rebuild with evaluation/build_demo.py, or upload the sample to run it again with your settings. Demo display names are shortened after reviewing representative evidence; original model labels remain visible in Topic Explorer. Assignments and numbers are unchanged.')
    with st.expander('Data quality report'):
        cols = st.columns(4)
        for col, label, key in zip(cols, ['Uploaded reviews','Valid reviews','Duplicates removed','Empty reviews removed'], ['uploaded','valid','duplicates_removed','missing_removed']):
            col.metric(label, quality[key])
        st.caption(f"Average review length: {quality['average_length']:.1f} words · Noise removed: {quality['noise_removed']} · Invalid dates: {quality['invalid_dates']} · Invalid ratings: {quality['invalid_ratings']}")
    st.markdown('**Read these results responsibly**')
    st.write('Review samples may be biased. Clusters and sentiment labels can be wrong, and one review may discuss several topics. Impact weights are heuristics. Synthetic demo results cannot establish real-world accuracy or market demand.')
    with st.expander('Optional AI topic labels'):
        st.caption('Send keywords and representative excerpts to your configured API for shorter topic labels. Assignments and measured scores stay unchanged.')
        if st.button('Rename topics with configured API'):
            try:
                from src.topic_modeling import rename_topics_with_api
                renamed = rename_topics_with_api(data, topics)
                data = data.copy(); scores = scores.copy()
                data['topic'] = data.topic_id.map(lambda t:renamed[t]['label'])
                scores['topic'] = scores.topic_id.map(lambda t:renamed[t]['label'])
                st.session_state.result = (data,quality,renamed,topic_meta,sentiment_meta,trends,trend_meta,scores)
                st.session_state.pop('insights',None)
                st.rerun()
            except Exception:
                st.warning('Topic renaming unavailable or response invalid. Original labels retained.')
st.divider()
st.caption('ConsumerLens  /  AI-powered consumer intelligence  ·  Every priority starts with evidence.')
