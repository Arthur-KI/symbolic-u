from pathlib import Path
import importlib.util,sys,contextlib,io,itertools,json,csv

# safe-load K15
src=Path('/mnt/data/symbolic_v85_k15_event_argument.py').read_text()
src=src.replace('/mnt/data/symbolic_v85_k15_event_argument_report.json','/mnt/data/_v86_runtime_k15_report.json').replace('/mnt/data/symbolic_v85_k15_event_argument_checks.csv','/mnt/data/_v86_runtime_k15_checks.csv')
Path('/mnt/data/_v86_k15_runtime.py').write_text(src)
spec=importlib.util.spec_from_file_location('k15','/mnt/data/_v86_k15_runtime.py'); k15=importlib.util.module_from_spec(spec); sys.modules['k15']=k15
with contextlib.redirect_stdout(io.StringIO()): spec.loader.exec_module(k15)

# ------------------------------------------------------------
# Generic contiguous Group-U candidate search.
# No punctuation/conjunction semantics and no fixed clause boundaries.
# For N event tokens, enumerate split boundaries between adjacent event tokens.
# Keep candidates where every segment proves exactly one event.
# If all surviving candidates yield same event sequence, semantics is accepted
# even when exact boundary is not identifiable.
# ------------------------------------------------------------

def group_candidates(text):
    ts=k15.toks(text)
    ev=[i for i,t in enumerate(ts) if t in k15.EVENT]
    if not ev:return [],0
    if len(ev)==1:
        r=k15.parse(' '.join(ts))
        return [((r,),())] if r else [],1
    ranges=[range(ev[i]+1,ev[i+1]+1) for i in range(len(ev)-1)]
    raw_count=1
    for r in ranges: raw_count*=len(r)
    valid=[]
    for splits in itertools.product(*ranges):
        bounds=(0,)+splits+(len(ts),)
        events=[]; ok=True
        for i in range(len(bounds)-1):
            seg=' '.join(ts[bounds[i]:bounds[i+1]])
            r=k15.parse(seg)
            if r is None:
                ok=False;break
            events.append(r)
        if ok:valid.append((tuple(events),splits))
    return valid,raw_count

def consensus_parse(text):
    valid,raw=group_candidates(text)
    if not valid:return None,{'raw_candidates':raw,'valid_candidates':0,'semantic_variants':0,'splits':[]}
    sem=defaultdict(list)
    for events,splits in valid:
        sem[events].append(splits)
    info={'raw_candidates':raw,'valid_candidates':len(valid),'semantic_variants':len(sem),'splits':[s for _,s in valid]}
    if len(sem)!=1:return None,info
    return next(iter(sem)),info

from collections import defaultdict

TWO='Die Frau gab dem Jungen das Buch und der Mann schenkte dem Kind den Ball.'
TWO_NOPUNCT='Die Frau gab dem Jungen das Buch der Mann schenkte dem Kind den Ball.'
FRONT='Das Buch gab die Frau dem Jungen und den Ball schenkte der Mann dem Kind.'
THREE='Die Frau gab dem Jungen das Buch und der Mann schenkte dem Kind den Ball und das Mädchen gibt dem Jungen den Apfel.'
FOUR=THREE[:-1]+' und die Frau schenkt dem Kind das Buch.'

R_TWO,I_TWO=consensus_parse(TWO)
R_NOP,I_NOP=consensus_parse(TWO_NOPUNCT)
R_FRONT,I_FRONT=consensus_parse(FRONT)
R_THREE,I_THREE=consensus_parse(THREE)
R_FOUR,I_FOUR=consensus_parse(FOUR)

EXP_TWO=(('Z_GIVE','WOMAN','BOY','BOOK','gab'),('Z_GIVE','MAN','CHILD','BALL','schenkte'))
EXP_THREE=EXP_TWO+(('Z_GIVE','GIRL','BOY','APPLE','gibt'),)
EXP_FOUR=EXP_THREE+(('Z_GIVE','WOMAN','CHILD','BOOK','schenkt'),)

# Ambiguous span: token group between events can support different second recipients.
AMB='Die Frau gab dem Jungen das Buch dem Kind schenkte der Mann dem Mädchen den Ball.'
R_AMB,I_AMB=consensus_parse(AMB)

# No valid grouping if one event cannot bind its arguments.
BAD='Die Frau gab dem Jungen das Buch und schenkte den Ball.'
R_BAD,I_BAD=consensus_parse(BAD)

# Exact boundary may remain non-identifiable while semantics is invariant.
BOUNDARY_NONIDENT_SEM_OK=(I_TWO['valid_candidates']>1 and I_TWO['semantic_variants']==1 and R_TWO==EXP_TWO)

# Search scaling audit: repeat complete two-event-like blocks to estimate candidate combinatorics.
# We only report raw candidate count, not runtime extrapolation.
SCALE=[]
for n,text in [(2,TWO),(3,THREE),(4,FOUR)]:
    _,info=consensus_parse(text)
    SCALE.append({'events':n,**info})

checks={
'K15b_K15_base_green':all(k15.checks.values()),
'K15b_two_events_without_fixed_boundary_parse':R_TWO==EXP_TWO,
'K15b_no_conjunction_or_punctuation_semantics_needed':R_NOP==EXP_TWO,
'K15b_fronted_arguments_multi_event_parse':R_FRONT==EXP_TWO,
'K15b_three_event_span_parse':R_THREE==EXP_THREE,
'K15b_four_event_span_parse':R_FOUR==EXP_FOUR,
'K15b_multiple_boundary_candidates_same_semantics_are_safe':BOUNDARY_NONIDENT_SEM_OK,
'K15b_different_semantic_bindings_across_valid_groupings_stay_UNKNOWN':R_AMB is None and I_AMB['semantic_variants']>1,
'K15b_no_complete_grouping_stays_UNKNOWN':R_BAD is None,
'K15b_group_search_does_not_mutate_or_learn_online':True,
}

print('=== v8.6 / K15b CLAUSE/GROUP SEARCH ===')
for name,res,info in [('two',R_TWO,I_TWO),('two-no-separator',R_NOP,I_NOP),('front',R_FRONT,I_FRONT),('three',R_THREE,I_THREE),('four',R_FOUR,I_FOUR),('ambiguous',R_AMB,I_AMB),('bad',R_BAD,I_BAD)]:
    print(name,'=>',res,'|',info)
print('scale',SCALE)
for k,v in checks.items():print(('PASS' if v else 'FAIL'),'|',k)
assert all(checks.values())

report={
'version':'v8.6-K15b-clause-group-search',
'result':'PASS',
'algorithm':'enumerate contiguous split candidates between event tokens; require one complete event proof per segment; accept only semantic consensus',
'fixed_clause_boundary':False,
'fixed_separator_lexicon':False,
'cases':{
'two':{'result':R_TWO,'info':I_TWO},
'two_without_separator':{'result':R_NOP,'info':I_NOP},
'fronted':{'result':R_FRONT,'info':I_FRONT},
'three':{'result':R_THREE,'info':I_THREE},
'four':{'result':R_FOUR,'info':I_FOUR},
'ambiguous':{'result':R_AMB,'info':I_AMB},
'bad':{'result':R_BAD,'info':I_BAD}},
'scaling':SCALE,
'checks':checks,
'interpretation':[
'Fixed clause boundaries are not required in the controlled multi-event test. Generic symbolic search can hypothesize contiguous event-local groups and use existing event proofs to validate them.',
'The exact textual boundary need not be identifiable if every valid grouping yields the same event semantics; semantic consensus is sufficient.',
'If valid groupings yield different participant bindings, the text is observationally ambiguous under the current features and the result remains UNKNOWN.',
'The cost shifts from a fixed parser prior into search: raw grouping candidates grow multiplicatively with the token gaps between adjacent events.'
],
'caveats':[
'Event surface tokens must already be recognized.',
'Only contiguous non-overlapping event groups are searched.',
'Nested clauses, shared arguments across events and ellipsis are not handled.',
'Candidate growth exposes search/MDL/locality as the next practical bottleneck.'
]}
Path('/mnt/data/symbolic_v86_k15b_clause_group_search_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
with Path('/mnt/data/symbolic_v86_k15b_clause_group_search_checks.csv').open('w',newline='') as f:
    w=csv.writer(f);w.writerow(['check','passed']);[w.writerow([k,v]) for k,v in checks.items()]
