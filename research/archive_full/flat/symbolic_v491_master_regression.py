from __future__ import annotations
from pathlib import Path
import argparse, csv, json, re, subprocess, sys, time

BASE=Path('/mnt/data')

TESTS=[
    dict(name='TERNARY_U_KEY', category='core', file='ternary_u_key_negative_full.py', timeout=15,
         markers=['All ternary U/Key assertions passed.']),
    dict(name='TEMPORAL_STATE', category='core', file='temporal_mini_lm_v02.py', timeout=15,
         markers=['Temporal Mini-LM v0.2 assertions passed.']),
    dict(name='CORE_V1', category='core', file='symbolic_mini_lm_v1.py', timeout=20,
         markers=['ALL v1 ASSERTIONS PASSED']),
    dict(name='DICTIONARY_ONTOLOGY', category='language', file='symbolic_mini_lm_v11_dictionary_test.py', timeout=20,
         markers=['ALL v1.1 ASSERTIONS PASSED']),
    dict(name='CONTROLLED_E2E_V12', category='language', file='symbolic_mini_lm_v12_end_to_end.py', timeout=20,
         markers=['Passed 14/14','ALL v1.2 END-TO-END ASSERTIONS PASSED']),
    dict(name='REFERENCE_U', category='language', file='symbolic_mini_lm_v13_reference_u.py', timeout=20,
         markers=['Passed 12 / 12','ALL v1.3 REFERENCE-U ASSERTIONS PASSED']),
    dict(name='CLAUSE_U', category='language', file='symbolic_v27_clause_u_ablation.py', timeout=20,
         markers=['V2_SAFE_CLAUSE_U        9/9 false_commit=0 wrong_entity=0']),
    dict(name='RAW_SWEET_PORRIDGE', category='language', file='symbolic_v32_full_raw_end_to_end.py', timeout=25,
         markers=['TOTAL 30/30 = 100.0%','wrong committed answers: 0']),
    dict(name='CLAIM_WORLD_CONTEXT', category='language', file='symbolic_v38_context_clause_ports.py', timeout=20,
         markers=['Context-aware Wolf benchmark: 20/20','Adversarial context checks:   5/5']),
    dict(name='QUOTE_BINDING', category='language', file='frau_holle_v35_quote_binding_test.py', timeout=20,
         markers=['v3.5 speaker-quote binding passed']),
    dict(name='SHARED_RECURSIVE_V39', category='reasoning', file='symbolic_v39_shared_recursive_u.py', timeout=25,
         markers=['24/24 passed','without Recursive-U: 0','without Standard-U number ontology: 0']),
    dict(name='MATH_OPT_V42', category='reasoning', file='symbolic_math_v42_solver_optimization.py', timeout=50,
         markers=['baseline: 12 / 12','optimized: 12 / 12','30 / 30','optimized duplicate U signatures: 0']),
    dict(name='REC_SUB', category='recursive_verifier', file='symbolic_v44b_recursive_u_verifier.py', args=['SUB'], timeout=45,
         json_checks={'target':'SUB','conflict':0,'unseen_passed':12,'unseen_n':12}),
    dict(name='REC_MUL', category='recursive_verifier', file='symbolic_v44b_recursive_u_verifier.py', args=['MUL'], timeout=75,
         json_checks={'target':'MUL','conflict':0,'unseen_passed':10,'unseen_n':10}),
    dict(name='REC_PATH', category='recursive_verifier', file='symbolic_v44b_recursive_u_verifier.py', args=['PATH'], timeout=25,
         json_checks={'target':'PATH','conflict':0,'unseen_passed':168,'unseen_n':168}),
    dict(name='REC_PATH_COST', category='recursive_verifier', file='symbolic_v44b_recursive_u_verifier.py', args=['PATH_COST'], timeout=35,
         json_checks={'target':'PATH_COST','conflict':0,'unseen_passed':80,'unseen_n':80}),
    dict(name='ADAPTIVE_LIBRARY_V45', category='learning', file='symbolic_v45_adaptive_u_library.py', timeout=45,
         markers=['PASS | query_without_training_is_not_evidence','PASS | poisoned_training_rejected']),
    dict(name='NESTED_LEARNING_V46', category='learning', file='symbolic_v46_nested_on_demand.py', timeout=50,
         markers=['PASS | child_installed_before_parent','PASS | cycle_detected_no_install','PASS | failed_child_aborts_parent']),
    dict(name='AUTO_DEPENDENCY_V47', category='learning', file='symbolic_v47_automatic_dependency_discovery.py', timeout=50,
         markers=['PASS | dependency_discovered_from_candidate_body','PASS | poisoned_child_blocks_parent','PASS | candidate_dependency_cycle_aborts']),
    dict(name='ABSTRACTION_INVENTION_V48', category='learning', file='symbolic_v48_abstraction_invention.py', timeout=70,
         markers=['PASS | anonymous_relation_invented','PASS | second_domain_reuses_same_abstraction','PASS | failed_parent_does_not_commit_child']),
    dict(name='VERSIONED_REVISION_V49', category='learning', file='symbolic_v49_versioned_revision.py', timeout=40,
         markers=['PASS | revision_found_symbolic_specialization','PASS | failed_v3_rolls_back_transactionally','PASS | unrepairable_challenge_disables_unsafe_active_version']),
]

FAST_NAMES={
    'TERNARY_U_KEY','TEMPORAL_STATE','CORE_V1','REFERENCE_U','RAW_SWEET_PORRIDGE',
    'CLAIM_WORLD_CONTEXT','SHARED_RECURSIVE_V39','REC_SUB','REC_PATH','ADAPTIVE_LIBRARY_V45',
    'VERSIONED_REVISION_V49'
}

def extract_json(stdout:str):
    # v4.4b prints one JSON object. Be tolerant of harmless leading lines.
    i=stdout.find('{')
    j=stdout.rfind('}')
    if i<0 or j<i:
        return None
    try:
        return json.loads(stdout[i:j+1])
    except Exception:
        return None

def run_one(spec):
    cmd=[sys.executable,str(BASE/spec['file']),*spec.get('args',[])]
    t0=time.perf_counter()
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,cwd=str(BASE),timeout=spec['timeout'])
        rc=p.returncode
        out=p.stdout or ''
        err=p.stderr or ''
        timed_out=False
    except subprocess.TimeoutExpired as e:
        rc=None
        out=e.stdout.decode(errors='replace') if isinstance(e.stdout,bytes) else (e.stdout or '')
        err=e.stderr.decode(errors='replace') if isinstance(e.stderr,bytes) else (e.stderr or '')
        timed_out=True

    marker_fail=[]
    for marker in spec.get('markers',[]):
        if marker not in out:
            marker_fail.append(marker)

    json_fail=[]
    parsed=None
    if spec.get('json_checks'):
        parsed=extract_json(out)
        if parsed is None:
            json_fail.append('valid JSON result')
        else:
            for k,v in spec['json_checks'].items():
                if parsed.get(k)!=v:
                    json_fail.append(f'{k}={v!r} (got {parsed.get(k)!r})')

    passed=(not timed_out and rc==0 and not marker_fail and not json_fail)
    status='PASS' if passed else ('TIMEOUT' if timed_out else 'FAIL')
    return {
        'name':spec['name'],'category':spec['category'],'file':spec['file'],
        'args':spec.get('args',[]),'status':status,'passed':passed,'returncode':rc,
        'seconds':round(time.perf_counter()-t0,3),'timeout_seconds':spec['timeout'],
        'missing_markers':marker_fail,'json_check_failures':json_fail,
        'parsed_result':parsed,
        'stdout_tail':out.strip().splitlines()[-20:],
        'stderr_tail':err.strip().splitlines()[-12:],
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fast',action='store_true',help='run a representative subset')
    ap.add_argument('--start',type=int,default=1,help='1-based inclusive test index')
    ap.add_argument('--end',type=int,default=None,help='1-based inclusive test index')
    ap.add_argument('--suffix',default='',help='suffix for shard report filenames')
    args=ap.parse_args()
    specs=[s for s in TESTS if (not args.fast or s['name'] in FAST_NAMES)]
    end=args.end or len(specs)
    specs=specs[max(0,args.start-1):end]

    rows=[]
    start=time.perf_counter()
    for i,s in enumerate(specs,1):
        print(f'[{i:02}/{len(specs):02}] {s["name"]} ...',flush=True)
        r=run_one(s); rows.append(r)
        print(f'      {r["status"]}  {r["seconds"]:.3f}s',flush=True)

    cat={}
    for r in rows:
        c=cat.setdefault(r['category'],{'passed':0,'n':0,'failed':[]})
        c['n']+=1
        if r['passed']: c['passed']+=1
        else: c['failed'].append(r['name'])

    total_pass=sum(r['passed'] for r in rows)
    overall=(total_pass==len(rows))
    report={
        'version':'v4.9.1-master-regression-freeze',
        'mode':'fast' if args.fast else 'full',
        'result':'PASS' if overall else 'FAIL',
        'passed':total_pass,'n':len(rows),
        'seconds':round(time.perf_counter()-start,3),
        'categories':cat,
        'tests':rows,
        'frozen_invariants':[
            'KEY truth and U state remain separate; U=-1 does not imply KEY=-1.',
            'Explicit negation is required for KEY=-1; contradiction remains UNKNOWN plus contradiction state.',
            'Query matching is not evidence; unknown remains unknown.',
            'Story/time state is local; stale historical state does not override latest state.',
            'Mention/Reference/Clause role resolution preserves ambiguity instead of recency false commits.',
            'CLAIM and HYPOTHETICAL propositions do not leak into WORLD facts.',
            'Standard-U and Recursive-U exchange ordinary Keys; wrong arithmetic outputs remain UNKNOWN.',
            'Recursive-U verifier enforces well-founded numeric decrease or finite structural traversal.',
            'On-demand learning requires independent training/self-tests and rejects poisoned evidence.',
            'Nested/dependency learning is child-first and transactional; cycles abort without partial install.',
            'Anonymous relations can be invented from typed structural holes and reused across domains.',
            'Versioned U knowledge can be challenged, specialized, rolled back, superseded or quarantined.'
        ],
        'important_scope_note':(
            'This freeze re-runs multiple historically frozen codepaths. It proves regression compatibility of the components, '
            'not yet one unified v5 runtime. v3.x language/context and v4.x adaptive learning are still separate implementations.'
        ),
        'performance_note':(
            'Generic recursive induction is substantially heavier than proof reuse; REC_MUL is deliberately given the largest timeout. '
            'The freeze records runtime separately so future changes can detect search regressions without conflating them with logical correctness.'
        )
    }
    suffix=(('_'+args.suffix) if args.suffix else '')
    (BASE/f'symbolic_v491_master_regression_report{suffix}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (BASE/f'symbolic_v491_master_regression_summary{suffix}.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['name','category','status','seconds','file'])
        for r in rows: w.writerow([r['name'],r['category'],r['status'],r['seconds'],r['file']])

    print('\n=== MASTER REGRESSION ===')
    print(f'{total_pass}/{len(rows)} tests passed')
    for k,v in cat.items():
        print(f'{k:20} {v["passed"]}/{v["n"]}',('FAIL '+','.join(v['failed'])) if v['failed'] else '')
    print('RESULT:',report['result'])
    print('Saved report/summary.')
    raise SystemExit(0 if overall else 1)

if __name__=='__main__':
    main()
