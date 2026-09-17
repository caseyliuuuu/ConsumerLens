"""Topic prevalence trends; dated-only denominators and percentage points."""
import pandas as pd

def analyze_trends(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dated=df[df.date.notna()].copy()
    if len(dated)<4 or dated.date.nunique()<2:
        return pd.DataFrame(), {'available':False,'reason':'At least four dated reviews across two dates are required.'}
    midpoint=dated.date.min()+(dated.date.max()-dated.date.min())/2
    dated['period']=dated.date.map(lambda d:'Earlier' if d<=midpoint else 'Later')
    totals=dated.groupby('period').size()
    if len(totals)<2: return pd.DataFrame(),{'available':False,'reason':'Two populated periods required.'}
    counts=pd.crosstab(dated.topic_id,dated.period).reindex(index=sorted(df.topic_id.unique()),columns=['Earlier','Later'],fill_value=0)
    shares=counts.div(totals,axis=1)*100
    result=shares.rename(columns={'Earlier':'earlier_share','Later':'later_share'}).reset_index()
    result['change_pp']=result.later_share-result.earlier_share
    result['trend']=result.change_pp.map(lambda v:'Emerging' if v>=2 else 'Declining' if v<=-2 else 'Stable')
    return result,{'available':True,'dated_reviews':len(dated),'earlier_n':int(totals['Earlier']),'later_n':int(totals['Later']),'split':midpoint.isoformat(),'low_sample':bool(totals.min()<30)}
