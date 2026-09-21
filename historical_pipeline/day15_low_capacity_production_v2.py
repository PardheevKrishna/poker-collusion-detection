"""One bound depth-three exploratory exception; never upload or select.

Both original eligibility gates remain false. Every compute stage requires the
exact separate exception decision, unchanged candidate artifacts, independent
validation integrity, and a production prefit review of this source/protocol.
This does not unlock any other candidate or alter the original guarded script.
Invoke stages in fresh processes so numerical eval extraction uses NUMBA4.
"""
import os
for _key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','POLARS_MAX_THREADS'):
    os.environ[_key] = '2'
os.environ['NUMBA_NUM_THREADS'] = '4'
import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import pickle
import time
import numpy as np
from private_audit import sha, write_json, save_npz

OUT = Path('models/day15_low_capacity_production')
CACHE = Path('cache/day15_low_capacity_production')
BASE = Path('submissions/submission_v50_soft_top5_consensus.csv')
BASE_SHA = 'b6196e8d05e445a1ab1f8abcbb77b9d8324a3dd7bd3678b243011b7efcf2952a'
OUTPUT = Path('submissions/submission_v53_low_capacity_soft_map.csv')
PRIMARY = Path('models/day15_low_capacity_soft_map_v2')
PRIMARY_REVIEW = Path('models/day15_low_capacity_soft_map_v2_review/integrity.json')
POLICY = Path('DAY15_LOW_CAPACITY_PROBE_POLICY.md')
DECISION = Path('models/day15_depth3_probe_decision.json')
EXCEPTION_DOC = Path('DAY15_DEPTH3_EXPLORATORY_EXCEPTION.md')
EXCEPTION_REVIEW = Path('DAY15_DEPTH3_EXCEPTION_REVIEW.md')
EXACT_LATE_GAIN = -4.40917107583667e-06
EXACT_VALIDATION_SHA = '8fcf530c37a4f27cc246cabb2f6511e98f86fb41d3ad2a0b6b82c8589959161c'
EXACT_DECISION_SHA = '020da0699e72e2479b4061f67409589bc4986877e24cb369b0811973db024214'
PRECISE = Path('cache/day15_precision/features.npy')
SOURCE_BATCHES = Path('cache/day13_shared_soft_top5_consensus_family_2')
MODEL = OUT/'soft_full.ubj'
CUE_OUT = OUT/'cues'
OLD_MODELS = [Path('models/day14_repaired_map/shared_map_chrono_full.pkl'),
              Path('models/day13_shared_retrieval/shared_lambda_chrono_full.pkl')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def exact_file(path, digest):
    assert sha(path) == digest, str(path)


def atomic_npy(path, a):
    path = Path(path); temp = path.with_name(path.name+'.tmp')
    with temp.open('wb') as stream:
        np.save(stream, a); stream.flush(); os.fsync(stream.fileno())
    assert np.array_equal(np.load(temp), a)
    temp.replace(path)


def array_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def checked_ranker(model):
    booster = model.get_booster()
    assert model.n_features_in_ == 2655 and booster.num_boosted_rounds() == 650
    assert json.loads(booster.save_config())['learner']['objective']['name'] == 'rank:map'
    def walk(node, depth=0):
        assert depth <= 3
        if 'children' in node:
            for child in node['children']: walk(child, depth+1)
    for tree in booster.get_dump(dump_format='json'): walk(json.loads(tree))


def unchanged(definition):
    for path, digest in definition['source_hashes'].items(): exact_file(path,digest)


def guards():
    exact_file(BASE, BASE_SHA)
    exact_file(DECISION, EXACT_DECISION_SHA)
    decision = read(DECISION)
    assert decision['exception_authorized'] is True
    assert decision['exploratory_probe_eligible'] is False
    assert decision['original_screen_passed'] is False
    assert decision['upload_limit'] == 1 and decision['family'] == 2 and decision['eligible_pairs'] == 660
    assert decision['recipe'] == 'soft_all_query_25'
    assert decision['expected_daily_used_before'] == 2
    assert decision['no_post_public_weight_seed_scope_rescue'] is True
    assert decision['exact_late_gain'] == EXACT_LATE_GAIN
    required = [Path('day15_low_capacity_soft_map_v2.py'),PRIMARY/'protocol.json',PRIMARY/'validation.json',
                PRIMARY_REVIEW,POLICY,EXCEPTION_DOC,EXCEPTION_REVIEW]
    bound = {str(Path(path).resolve()).casefold():digest for path,digest in decision['bound_hashes'].items()}
    assert len(bound) == len(decision['bound_hashes']), 'Duplicate normalized decision input path'
    for path in required:
        key = str(path.resolve()).casefold(); assert key in bound, str(path)
        exact_file(path,bound[key])
    for path,digest in decision['bound_hashes'].items(): exact_file(path,digest)
    exact_file(PRIMARY/'validation.json', EXACT_VALIDATION_SHA)
    primary = read(PRIMARY/'validation.json'); review = read(PRIMARY_REVIEW)
    assert primary['exploratory_probe_eligible'] is False
    assert review['status'] == 'passed' and review['exploratory_probe_eligible'] is False
    exact_file(PRIMARY/'validation.json', review['validation_sha256'])
    exact_file(PRIMARY/'protocol.json', review['protocol_sha256'])
    exact_file(POLICY, primary['decision_policy_sha256'])
    exact_file('day15_low_capacity_soft_map_v2.py', review['runner_sha256'])
    for path,digest in read(PRIMARY/'protocol.json')['definition']['source_hashes'].items(): exact_file(path,digest)
    original = primary['gates']['candidate_2']['mean_screen_passed']
    assert original is False and review['original_screen_passed'] is False
    rows = primary['results']
    views = ['full','random_short_a','random_short_b','early_two_thirds','late_two_thirds']
    assert len(rows) == 5 and [r['view'] for r in rows] == views
    assert all(r['family'] == 2 and r['recipe'] == 'candidate' for r in rows)
    assert all(r['gain'] > 0 for r in rows[:4])
    assert rows[4]['gain'] == decision['exact_late_gain'] == EXACT_LATE_GAIN
    assert rows[4]['gain'] < 0, 'The documented negative mean is not rounded into a pass'
    assert all(r['pool_bootstrap_gain_95'][1] >= 0 for r in rows[1:])
    omitted = primary['full_leave_one_fold_out']
    assert len(omitted) == 4 and sorted(r['omitted_fold'] for r in omitted) == list(range(4))
    assert all(r['mean_gain'] > 0 for r in omitted)
    recomputed_strict = rows[0]['gain'] > 0 and all(r['gain'] >= 0 for r in rows) and sum(g >= 0 for g in rows[0]['fold_gains']) >= 3
    assert recomputed_strict is False
    for path,digest in review.get('output_hashes',{}).items(): exact_file(path,digest)
    return primary, review


def source_paths():
    paths = [Path(__file__).resolve(), BASE, PRIMARY/'protocol.json', PRIMARY/'validation.json',
             PRIMARY_REVIEW, POLICY, DECISION, EXCEPTION_DOC, EXCEPTION_REVIEW,
             Path('day15_low_capacity_production.py'), Path('DAY15_LOW_CAPACITY_REVIEW.md'),
             Path('day15_low_capacity_soft_map_v2.py'), Path('day15_precise_specialist_map.py'),
             Path('day15_precision_cues.py'), Path('day15_precision_features_v2.py'),
             Path('rank_cue_history.py'), Path('private_audit.py'), Path('private_audit_evidence.py'),
             Path('rank_episodes.py'), Path('rank_policy.py'), Path('rank_card_history.py'),
             Path('query_context.py'), Path('detailed_actions.py'), Path('decision_policy.py'), Path('rank_rich.py'),
             Path('day14_repaired_validation.py'), Path('day13_shared_retrieval.py'), Path('train.py'),
             Path('day13_infer_shared.py'), Path('day14_infer_consensus.py'), Path('day14_map_retrieval.py'),
             Path('day9_retrieval_research.py'), Path('features.py'), Path('mechanism_actions.py'),
             Path('decision_equity.py'), Path('equity_gpu.py'), Path('decision_value_features.py'),
             Path('models/day15_precision_cues/protocol.json'),
             Path('models/day15_precision_features/feature_report.json'), PRECISE,
             Path('audits/private_20260911/split_manifest.npz'),
             Path('data/development_labels.csv'), Path('data/development_evidence.csv'),
             Path('models/day10_history_stress/window_masks.npz'), *OLD_MODELS]
    names = ['h','sp','sc','s','board','a','offsets','train_X','train_events','train_rich_policy',
             'train_mechanisms','train_countercards','train_decision_values','train_mechanism_actions',
             'train_rich_detail','train_rich_query','train_mechanism_signs','train_mechanism_direction','train_card_history',
             'hand_oof','nested_cues/outer_4','eval_events','day3_known_scope','countercards_selected',
             'decision_queries','decision_action_queries','decision_equities','decision_equities_computed',
             'policy_inputs','policy_probs','rich_policy_probs','aug_0_mask','aug_1_mask',
             'aug_0_usable','aug_1_usable']
    paths += [Path('cache')/(name+('.npz' if name=='day3_known_scope' else '.npy')) for name in names]
    paths += [Path('cache/hands.parquet'),Path('cache/development_labels.parquet')]
    paths += [Path(p) for p in read(PRIMARY/'protocol.json')['definition']['source_hashes']]
    paths += [Path(f'models/cue_outer4_inner{j}.ubj') for j in (0,1)]
    paths += sorted(SOURCE_BATCHES.glob('*.npz'))
    assert len(list(SOURCE_BATCHES.glob('*.npz'))) == 46
    # Freeze exactly the existing batches touched by the unchanged soft scope.
    for p in sorted(SOURCE_BATCHES.glob('*.npz')):
        lo = p.stem
        paths += [Path(f'cache/eval_batches/{lo}.parquet'), Path(f'cache/eval_batches/{lo}_rich_events.npy'),
                  Path(f'cache/final_features/{lo}.npy'), Path(f'cache/day2_features/{lo}_mechanism.npy')]
    return list(dict.fromkeys(paths))


def prepare():
    primary,_ = guards(); OUT.mkdir(exist_ok=True); CACHE.mkdir(exist_ok=True)
    path = OUT/'protocol.json'
    params = read(PRIMARY/'protocol.json')['definition']['parameters']
    assert params['max_depth'] == 3 and params['n_estimators'] == 650
    assert params['objective'] == 'rank:map'
    definition = dict(
        recipe='.75 actual v50 + .25 whole-query rankmatched depth3 precise soft MAP; soft scope only.',
        output=str(OUTPUT), base_sha256=BASE_SHA, ranker_parameters=params,
        cue_parameters=read('models/day15_precision_cues/protocol.json')['definition']['params'],
        cue_full_recipe='Exactly original production two-way pool split. RNG3224 shuffles sorted present pools; second half gets inner1. Continue same RNG for inner0 then inner1 normal samples capped50000. Seeds3258/3259. Change only final428 cue inputs to precise8192 direct values. All allowed positive-pair hands; released evidence weight1, other hands weight.15.',
        cue_prediction='Each train row uses only the model excluding its complete inner pool partition. Eval uses mean of the two precise cue models. Never average the12 outer-audit models.',
        inherited_distinction='Validation uses3 held-inner models averaging on the excluded outer; full production retains the original2-way random-pool construction. Training fraction about50% in both. This distinction is inherited, not proof of calibration equality.',
        ranker='One650-tree depth3 soft MAP, same parameters and seed91925, all132 known soft pairs. Views0/3/4 weights1/.35/.35. Replace428 direct columns2074:2502 and153 cue-history columns2502:2655. Drop4 constant family flags.',
        eval_precision='8192 draws seed1776 only for preflop queries of existing660 soft scope pairs. All exact postflop queries and other policies stay unchanged. NUMBA4 extraction. Original512 queries and coarseDV replay required.',
        gating='Exact separate depth3 exception decision required; both original policy and strict screen remain false. Independent audit passed. New prefit_review.json must bind this source and protocol before any compute stage.',
        exploratory_probe_eligible=False, exception_authorized=True, exception_decision_sha256=sha(DECISION),
        decision_policy_sha256=sha(POLICY), original_gate_passed=primary['gates']['candidate_2']['mean_screen_passed'], private_rank_confirmed=False, production_fits=3,
        source_hashes={str(p):sha(p) for p in source_paths()})
    if path.exists(): assert read(path)['definition'] == definition
    else: write_json(path, dict(created_at_utc=datetime.now(timezone.utc).isoformat(),definition=definition))
    print('Precise production prepared', sha(path), flush=True)


def checked():
    guards(); protocol = read(OUT/'protocol.json'); definition = protocol['definition']
    unchanged(definition)
    prefit = read(OUT/'prefit_review.json')
    assert prefit['status'] == 'passed'
    exact_file(OUT/'protocol.json',prefit['protocol_sha256'])
    exact_file(Path(__file__).resolve(),prefit['runner_sha256'])
    return sha(OUT/'protocol.json'), definition


def cue_layouts(d):
    # Literal reproduction of rank_cue_history.nested_cues(...,outer=4).
    pools = d.h[d.e[:,0],5].astype('int'); groups = np.unique(pools)
    rng = np.random.default_rng(3224); rng.shuffle(groups)
    assignment = np.zeros(400,int); assignment[groups[len(groups)//2:]] = 1
    inner = assignment[pools]; result = []
    assert np.array_equal(pools,d.epool)
    for j in (0,1):
        available = np.flatnonzero(inner != j)
        normal = available[d.yp[available] == 0]; positive = available[d.yp[available] > 0]
        normal = rng.choice(normal,min(len(normal),50000),replace=False)
        tr = np.r_[normal,positive]; va = np.flatnonzero(inner == j)
        weight = np.where(d.yp[tr] == 0,.15,np.where(d.y[tr] > 0,1.,.15))
        assert not np.intersect1d(pools[tr],pools[va]).size
        result.append(dict(indices=tr,labels=d.y[tr],weights=weight,held_indices=va))
    return inner, result


def fit_cues():
    ph, definition = checked()
    import xgboost as xgb
    from day13_shared_retrieval import dataset
    CUE_OUT.mkdir(exist_ok=True)
    final = CUE_OUT/'manifest.json'
    if final.exists():
        m = read(final); assert m['protocol_sha256'] == ph
        for p,h in m['artifact_hashes'].items(): exact_file(p,h)
        print('Existing precise full cues verified',flush=True); return
    d = dataset()
    inner, layouts = cue_layouts(d)
    CR = np.column_stack([d.X[:,[j for j in range(320) if j not in (6,7)]],
        np.load('cache/train_rich_policy.npy',mmap_mode='r'),
        np.load('cache/train_mechanisms.npy',mmap_mode='r')[:,:708],
        np.load('cache/train_countercards.npy',mmap_mode='r'),np.load(PRECISE,mmap_mode='r')]).astype('float32')
    assert CR.shape == (len(d.e),1557) and np.isfinite(CR).all()
    old_dv = np.load('cache/train_decision_values.npy',mmap_mode='r')
    old_cp = np.load('cache/nested_cues/outer_4.npy',mmap_mode='r')
    cp = np.zeros((len(d.e),4),np.float32); coverage = np.zeros(len(d.e),np.int8); records = []
    for j, layout in enumerate(layouts):
        stem = CUE_OUT/f'inner_{j}'; mp = stem.with_suffix('.ubj'); lp = stem.with_suffix('.layout.npz'); jp = stem.with_suffix('.training.json')
        tr, va = layout['indices'],layout['held_indices']
        sample = va[np.unique(np.linspace(0,len(va)-1,min(256,len(va)),dtype=int))]
        coarse_R = CR[sample].copy(); coarse_R[:,-428:] = old_dv[sample]
        old_model = xgb.Booster(); old_model.load_model(f'models/cue_outer4_inner{j}.ubj'); old_model.set_param({'device':'cpu','nthread':2})
        old_pred = old_model.predict(xgb.DMatrix(coarse_R,nthread=2))
        difference = float(np.max(abs(old_pred-old_cp[sample])))
        assert difference <= 1e-7, ('original full cue replay',j,difference)
        del old_model,coarse_R,old_pred
        if lp.exists():
            with np.load(lp) as prior:
                assert set(prior.files) == set(layout)
                for key,value in layout.items(): assert np.array_equal(prior[key],value),key
        else: save_npz(lp,**layout)
        if mp.exists() or jp.exists():
            assert mp.exists() and jp.exists(), 'Partial cue model requires review; no overwrite.'
            record = read(jp); assert record['protocol_sha256'] == ph
            exact_file(mp,record['model_sha256']); exact_file(lp,record['layout_sha256'])
        else:
            model = xgb.XGBClassifier(**definition['cue_parameters'],random_state=3258+j)
            start = time.time(); model.fit(CR[tr],d.y[tr],sample_weight=layout['weights']); seconds = time.time()-start
            model.set_params(device='cpu'); temp = mp.with_name(mp.stem+'.tmp.ubj'); model.save_model(temp)
            with temp.open('r+b') as stream: stream.flush(); os.fsync(stream.fileno())
            reload = xgb.Booster(); reload.load_model(temp); reload.set_param({'device':'cpu','nthread':2})
            assert reload.num_features() == 1557 and reload.num_boosted_rounds() == 650
            assert np.array_equal(model.predict_proba(CR[tr[:128]]),reload.predict(xgb.DMatrix(CR[tr[:128]],nthread=2)))
            temp.replace(mp)
            record = dict(protocol_sha256=ph,inner=j,seed=3258+j,parameters=definition['cue_parameters'],
                rows=len(tr),training_pairs=np.unique(d.e[tr,1]).tolist(),training_pools=np.unique(d.epool[tr]).tolist(),
                held_pools=np.unique(d.epool[va]).tolist(),pool_overlap=0,features=1557,rounds=650,seconds=seconds,
                model_sha256=sha(mp),layout_sha256=sha(lp),coarse_cue_replay_rows=len(sample),coarse_cue_replay_max_difference=difference)
            write_json(jp,record); del model,reload; gc.collect()
        model = xgb.Booster(); model.load_model(mp); model.set_param({'device':'cpu','nthread':2})
        for lo in range(0,len(va),16384):
            ix = va[lo:lo+16384]; cp[ix] = model.predict(xgb.DMatrix(CR[ix],nthread=2)); coverage[ix] += 1
        records.append(record); del model; gc.collect()
        print('Production precise cue',j,'complete',flush=True)
    assert (coverage == 1).all() and np.isfinite(cp).all() and ((cp>=0)&(cp<=1)).all()
    assert np.allclose(cp.sum(1),1,atol=1e-5)
    unchanged(definition)
    atomic_npy(CUE_OUT/'oof_probabilities.npy',cp); atomic_npy(CUE_OUT/'inner_assignment.npy',inner)
    artifacts = [CUE_OUT/'oof_probabilities.npy',CUE_OUT/'inner_assignment.npy']
    for j in (0,1): artifacts += [CUE_OUT/f'inner_{j}{ext}' for ext in ('.ubj','.layout.npz','.training.json')]
    write_json(final,dict(status='passed',protocol_sha256=ph,records=records,rows=len(cp),oof_predictions_per_row=1,
        eval_models_to_average=2,artifact_hashes={str(p):sha(p) for p in artifacts},precise_features_sha256=sha(PRECISE)))


def checked_cues(ph):
    m = read(CUE_OUT/'manifest.json'); assert m['status'] == 'passed' and m['protocol_sha256'] == ph
    for p,h in m['artifact_hashes'].items(): exact_file(p,h)
    return m


def fit_ranker():
    ph, definition = checked(); checked_cues(ph)
    import xgboost as xgb
    from day13_shared_retrieval import dataset
    from private_audit_evidence import EvidenceData
    from day14_repaired_validation import make_views
    from day15_low_capacity_soft_map_v2 import layout
    from rank_cue_history import history_features
    from threadpoolctl import threadpool_limits
    # This stage uses only cached direct values; import may set NUMBA2. Eval is
    # an independent process and explicitly requires NUMBA4 before extraction.
    manifest = OUT/'soft_full.training.json'
    if MODEL.exists() or manifest.exists():
        assert MODEL.exists() and manifest.exists(), 'Partial full ranker requires review.'
        m = read(manifest); assert m['protocol_sha256'] == ph
        exact_file(MODEL,m['model_sha256']); exact_file(OUT/'soft_full.layout.npz',m['layout_sha256'])
        exact_file(CUE_OUT/'manifest.json',m['cue_manifest_sha256']); return
    d = dataset(); ed = EvidenceData(d); vv = make_views(d,ed,4,chronological_training=True)
    precise = np.load(PRECISE,mmap_mode='r'); old = np.load('cache/train_decision_values.npy',mmap_mode='r')
    cp = np.load(CUE_OUT/'oof_probabilities.npy'); checks = []
    for view,z in enumerate(vv):
        ix = z['indices']; assert np.array_equal(z['R'][:,2074:2502],old[ix])
        prefix = array_sha(z['R'][:,:2074]); flags = array_sha(z['R'][:,2655:])
        with threadpool_limits(limits=1): ctx = history_features(d.X[ix],z['events'],cp[ix])
        assert ctx.shape == (len(ix),153) and np.isfinite(ctx).all()
        z['R'][:,2074:2502] = precise[ix]; z['R'][:,2502:2655] = ctx
        assert array_sha(z['R'][:,:2074]) == prefix and array_sha(z['R'][:,2655:]) == flags
        assert np.array_equal(z['R'][:,2074:2502],precise[ix]) and np.array_equal(z['R'][:,2502:2655],ctx)
        checks.append(dict(view=view,rows=len(ix),other_columns_preserved=True,precise_blocks_exact=True))
    R, training, pairs = layout(d,vv,4,2)
    assert len(pairs) == 132 and np.array_equal(pairs,np.flatnonzero(d.py==2))
    assert R.shape[1] == 2655 and np.isfinite(R).all()
    lp = OUT/'soft_full.layout.npz'
    if lp.exists():
        with np.load(lp) as prior:
            for key,value in training.items(): assert np.array_equal(prior[key],value),key
    else: save_npz(lp,**training)
    model = xgb.XGBRanker(**definition['ranker_parameters'],random_state=91925)
    start = time.time(); model.fit(R,training['labels'],group=training['group_counts'],sample_weight=training['query_weights']); seconds = time.time()-start
    model.set_params(device='cpu'); temp = MODEL.with_name(MODEL.stem+'.tmp.ubj'); model.save_model(temp)
    with temp.open('r+b') as stream: stream.flush(); os.fsync(stream.fileno())
    reload = xgb.XGBRanker(); reload.load_model(temp); reload.set_params(device='cpu',n_jobs=2)
    checked_ranker(reload)
    assert np.array_equal(model.predict(R[:256]),reload.predict(R[:256])); temp.replace(MODEL)
    unchanged(definition)
    write_json(manifest,dict(protocol_sha256=ph,source_sha256=sha(__file__),model_sha256=sha(MODEL),layout_sha256=sha(lp),
        cue_manifest_sha256=sha(CUE_OUT/'manifest.json'),precise_features_sha256=sha(PRECISE),
        training_pairs=pairs.tolist(),training_pools=np.unique(d.pool[pairs]).tolist(),parameters=definition['ranker_parameters'],
        trees=650,features=2655,seed=91925,training_views=[0,3,4],view_weights=[1.,.35,.35],feature_checks=checks,seconds=seconds))
    print('Precise production soft MAP fit complete',flush=True)


def scope_events():
    import pandas as pd
    sub = pd.read_csv(BASE,dtype=str); scope = np.load('cache/day3_known_scope.npz')
    wanted = scope['wanted'] & (scope['family']==1) & (sub.predicted_behavior.to_numpy()=='soft_play')
    assert len(sub) == 112540 and wanted.sum() == 660
    ev = np.load('cache/eval_events.npy',mmap_mode='r'); selected = np.load('cache/countercards_selected.npy')
    assert selected[wanted].all()
    starts = np.r_[0,np.cumsum(np.bincount(ev[:,1],minlength=len(sub)))]; batches = []
    for source in sorted(SOURCE_BATCHES.glob('*.npz')):
        lo = int(source.stem); be = ev[starts[lo]:starts[min(lo+2500,len(sub))]]; mask = wanted[be[:,1]]; e = be[mask]
        with np.load(source) as z: assert np.array_equal(e,z['events'])
        se = be[selected[be[:,1]]]; sm = wanted[se[:,1]]; assert np.array_equal(e,se[sm])
        batches.append((lo,e,mask,sm))
    assert len(batches)==46 and sum(len(z[1]) for z in batches)==43423
    assert np.array_equal(np.unique(np.concatenate([z[1][:,1] for z in batches])),np.flatnonzero(wanted))
    return sub,wanted,batches


def eval_equities():
    ph, definition = checked()
    from numba import cuda, get_num_threads
    from decision_equity import run
    from features import load_arrays
    assert get_num_threads() == 4
    _,_,batches = scope_events(); events = np.concatenate([z[1] for z in batches]); hands = np.unique(events[:,0])
    arrays = load_arrays(); _,_,sc,_,board,_,_ = arrays
    queries = np.load('cache/decision_queries.npy',mmap_mode='r')
    old = np.load('cache/decision_equities.npy',mmap_mode='r'); done = np.load('cache/decision_equities_computed.npy',mmap_mode='r')
    chunks = []
    for hand in hands:
        lo = np.searchsorted(queries[:,0],hand); hi = np.searchsorted(queries[:,0],hand,side='right')
        assert done[lo:hi].all(); local = np.arange(lo,hi); chunks.append(local[queries[lo:hi,1]==0])
    ix = np.concatenate(chunks); assert len(ix)==308560 and len(hands)==40149
    eqdir = CACHE/'equity'; eqdir.mkdir(exist_ok=True); target = eqdir/'manifest.json'
    if target.exists():
        m = read(target); assert m['protocol_sha256']==ph
        for p,h in m['artifact_hashes'].items(): exact_file(p,h)
        assert np.array_equal(np.load(eqdir/'query_indices.npy'),ix); return
    dh = cuda.to_device(np.asarray(sc,dtype='int32')); db = cuda.to_device(np.asarray(board,dtype='int32'))
    values = np.zeros((len(ix),6),np.float32); records = []
    for lo in range(0,len(ix),20000):
        qi = ix[lo:lo+20000]; path = eqdir/f'{lo:06d}.npz'; jp = path.with_suffix('.json')
        if path.exists() or jp.exists():
            assert path.exists() and jp.exists(), 'Partial equity checkpoint requires review.'
            r = read(jp); assert r['protocol_sha256']==ph; exact_file(path,r['sha256'])
            with np.load(path) as z:
                assert np.array_equal(z['indices'],qi); pred = z['values']
        else:
            pred = run(dh,db,queries[qi],8192)
            assert pred.shape == (len(qi),6) and np.isfinite(pred).all() and np.allclose(pred.sum(1),1,atol=2e-6)
            save_npz(path,indices=qi,values=pred)
            r = dict(protocol_sha256=ph,sha256=sha(path),rows=len(qi)); write_json(jp,r)
        values[lo:lo+len(qi)] = pred; records.append(r)
        print('Precise eval queries',lo+len(qi),len(ix),flush=True)
    sample = np.unique(np.linspace(0,len(ix)-1,256,dtype=int))
    assert np.array_equal(run(dh,db,queries[ix[sample]],512),old[ix[sample]])
    assert np.array_equal(run(dh,db,queries[ix[sample]],8192),values[sample])
    # Enforce the extractor's original numerical configuration before inference.
    from decision_value_features import extract
    train_e = np.load('cache/train_events.npy',mmap_mode='r'); train_dv = np.load('cache/train_decision_values.npy',mmap_mode='r')
    ti = np.sort(np.random.default_rng(9206208).choice(len(train_e),128,replace=False))
    P = np.load('cache/policy_probs.npy',mmap_mode='r'); AQ = np.load('cache/decision_action_queries.npy',mmap_mode='r')
    anchors = np.load('cache/train_mechanism_actions.npy',mmap_mode='r')
    assert np.array_equal(extract(train_e[ti],*arrays,P,AQ,old,anchors[ti]),train_dv[ti])
    unchanged(definition)
    atomic_npy(eqdir/'query_indices.npy',ix); atomic_npy(eqdir/'preflop_equities.npy',values)
    paths = [eqdir/'query_indices.npy',eqdir/'preflop_equities.npy']
    write_json(target,dict(status='passed',protocol_sha256=ph,rows=len(ix),hands=len(hands),draws=8192,seed=1776,
        old512_query_exact_replays=len(sample),new8192_query_exact_replays=len(sample),coarse_train_dv_exact_replays=len(ti),
        numba_threads=4,chunks=records,artifact_hashes={str(p):sha(p) for p in paths},postflop_modified=False))


class Unpickler(pickle.Unpickler):
    def find_class(self,module,name):
        if module=='__main__' and name=='Predictor':
            from day14_map_retrieval import Predictor
            return Predictor
        return super().find_class(module,name)


def infer():
    ph, definition = checked(); checked_cues(ph)
    import xgboost as xgb
    import polars as pl
    from numba import get_num_threads
    from features import load_arrays
    from mechanism_actions import build as mechanism_build
    from decision_value_features import extract
    from rank_cue_history import history_features
    from threadpoolctl import threadpool_limits
    from day9_retrieval_research import matched
    from day14_infer_consensus import proposal
    assert get_num_threads()==4
    training = read(OUT/'soft_full.training.json'); assert training['protocol_sha256']==ph
    exact_file(MODEL,training['model_sha256']); exact_file(CUE_OUT/'manifest.json',training['cue_manifest_sha256'])
    eqpath = CACHE/'equity'/'manifest.json'; eq = read(eqpath); assert eq['status']=='passed' and eq['protocol_sha256']==ph
    for p,h in eq['artifact_hashes'].items(): exact_file(p,h)
    qi = np.load(CACHE/'equity'/'query_indices.npy'); precise_values = np.load(CACHE/'equity'/'preflop_equities.npy')
    oldQ = np.load('cache/decision_equities.npy',mmap_mode='r'); Q = np.array(oldQ); Q[qi] = precise_values
    query_meta = np.load('cache/decision_queries.npy',mmap_mode='r'); assert (query_meta[qi,1]==0).all()
    # All other rows are a literal copy; only selected preflop indices are written.
    arrays = load_arrays(); I,P,RP,AQ = [np.load('cache/'+n+'.npy',mmap_mode='r') for n in ('policy_inputs','policy_probs','rich_policy_probs','decision_action_queries')]
    sub,wanted,batches = scope_events(); original = sub.copy()
    hands = pl.read_parquet('cache/hands.parquet',columns=['hand_id'])['hand_id'].to_numpy()
    model = xgb.XGBRanker(); model.load_model(MODEL); model.set_params(device='cpu',n_jobs=2)
    checked_ranker(model)
    references = [Unpickler(p.open('rb')).load() for p in OLD_MODELS]; references[0].model.set_params(device='cpu',n_jobs=2)
    old_cues=[]; new_cues=[]
    for j in (0,1):
        for dest,path in ((old_cues,Path(f'models/cue_outer4_inner{j}.ubj')),(new_cues,CUE_OUT/f'inner_{j}.ubj')):
            m=xgb.Booster();m.load_model(path);m.set_param({'device':'cpu','nthread':2});dest.append(m)
    dest=CACHE/'batches';dest.mkdir(exist_ok=True); checks=[]; total=changed=membership_changed=0
    stage_sources={str(p):sha(p) for p in (MODEL,OUT/'soft_full.training.json',CUE_OUT/'manifest.json',eqpath)}
    for lo,e,mask,sm in batches:
        source=SOURCE_BATCHES/f'{lo:06d}.npz'; z=np.load(source)
        path=dest/source.name; jp=path.with_suffix('.json')
        if path.exists() or jp.exists():
            assert path.exists() and jp.exists(), 'Partial inference batch requires review.'
            check=read(jp);assert check['protocol_sha256']==ph and check['stage_sources']==stage_sources
            exact_file(path,check['output_sha256']);exact_file(source,check['source_sha256'])
            with np.load(path) as saved:
                assert np.array_equal(e,saved['events']) and np.array_equal(z['candidate'],saved['baseline_v50'])
                raw=saved['raw'];candidate=saved['candidate']
                assert np.array_equal(candidate,.75*z['candidate']+.25*matched(e,z['candidate'],raw))
        else:
            X=pl.read_parquet(f'cache/eval_batches/{lo:06d}.parquet').to_numpy()[mask]
            R=np.column_stack([np.load(f'cache/final_features/{lo:06d}.npy',mmap_mode='r')[sm],np.load(f'cache/day2_features/{lo:06d}_mechanism.npy',mmap_mode='r')[sm]])
            assert R.shape==(len(e),2074)
            _,anchors,_,_=mechanism_build(e,*arrays,RP,I)
            coarseDV=extract(e,*arrays,P,AQ,oldQ,anchors)
            PE=np.load(f'cache/eval_batches/{lo:06d}_rich_events.npy',mmap_mode='r')[mask]
            CR=np.column_stack([X[:,[j for j in range(320) if j not in (6,7)]],PE,R[:,1352:2060],R[:,1312:1352],coarseDV])
            coarse_cp=sum(m.predict(xgb.DMatrix(CR,nthread=2)) for m in old_cues)/2
            with threadpool_limits(limits=1): coarse_ctx=history_features(X,e,coarse_cp)
            RR=np.column_stack([R,coarseDV,coarse_ctx,np.eye(4,dtype='float32')[np.full(len(e),2,int)]]).astype('float32')
            old_raw=np.column_stack([m.predict(RR,num_threads=2) for m in references])
            replay=float(np.max(abs(old_raw-z['raw'])));assert replay<=1e-7,(lo,replay)
            actual=.75*z['baseline']+.25*proposal(e,z['baseline'],old_raw)
            assert np.array_equal(actual,z['candidate']), 'Actual v50 scores must replay before feature replacement.'
            prefix=array_sha(RR[:,:2074]); flags=array_sha(RR[:,2655:])
            DV=extract(e,*arrays,P,AQ,Q,anchors);assert DV.shape==(len(e),428) and np.isfinite(DV).all()
            # Check deterministic row order and pair-seat invariance on a small
            # per-batch sample without changing any untouched feature block.
            sample=np.unique(np.linspace(0,len(e)-1,min(8,len(e)),dtype=int))
            assert np.array_equal(extract(e[sample[::-1]],*arrays,P,AQ,Q,anchors[sample[::-1]]),DV[sample[::-1]])
            swapped=e[sample].copy();swapped[:,[2,3]]=swapped[:,[3,2]]
            assert np.array_equal(extract(swapped,*arrays,P,AQ,Q,anchors[sample]),DV[sample])
            CR[:,-428:]=DV
            cp=sum(m.predict(xgb.DMatrix(CR,nthread=2)) for m in new_cues)/2
            assert np.isfinite(cp).all() and ((cp>=0)&(cp<=1)).all() and np.allclose(cp.sum(1),1,atol=1e-5)
            with threadpool_limits(limits=1): ctx=history_features(X,e,cp)
            RR[:,2074:2502]=DV;RR[:,2502:2655]=ctx
            assert array_sha(RR[:,:2074])==prefix and array_sha(RR[:,2655:])==flags
            assert np.array_equal(RR[:,2074:2502],DV) and np.array_equal(RR[:,2502:2655],ctx) and np.isfinite(RR).all()
            raw=model.predict(RR[:,:2655]).astype('float64');candidate=.75*actual+.25*matched(e,actual,raw)
            save_npz(path,events=e,baseline_v50=actual,raw=raw,candidate=candidate,original_raw=old_raw,precise_cue_probabilities=cp)
            check=dict(protocol_sha256=ph,stage_sources=stage_sources,output_sha256=sha(path),source_sha256=sha(source),rows=len(e),
                original_raw_max_difference=replay,v50_scores_exact=True,untouched_prefix_sha256=prefix,family_flags_sha256=flags,
                feature_matrix_sha256=array_sha(RR[:,:2655]),direct_values_sha256=array_sha(DV),cue_history_sha256=array_sha(ctx),
                replaced_blocks_exact=True,other_columns_exact=True,numba_threads=4,order_and_seat_replays=len(sample))
            write_json(jp,check)
            del X,R,anchors,coarseDV,PE,CR,coarse_cp,coarse_ctx,RR,old_raw,actual,DV,cp,ctx;gc.collect()
        for pair in np.unique(e[:,1]):
            ii=np.flatnonzero(e[:,1]==pair);old=ii[np.argsort(-z['candidate'][ii],kind='stable')[:5]];new=ii[np.argsort(-candidate[ii],kind='stable')[:5]]
            assert len(new)==5 and len(set(hands[e[new,0]]))==5
            assert np.array_equal(hands[e[old,0]],original.iloc[pair,3:].to_numpy()),int(pair)
            changed+=not np.array_equal(old,new);membership_changed+=set(old)!=set(new);sub.iloc[pair,3:]=hands[e[new,0]];total+=1
        checks.append(check);z.close();print('Precise soft inference',lo,'pairs',total,'changed',changed,flush=True)
    assert total==660 and sum(c['rows'] for c in checks)==43423
    assert sub.iloc[:,:3].equals(original.iloc[:,:3]) and sub.loc[~wanted].equals(original.loc[~wanted])
    unchanged(definition)
    payload=sub.to_csv(index=False).encode('utf-8');digest=hashlib.sha256(payload).hexdigest()
    cfgpath=OUT/'configuration.json'
    if OUTPUT.exists():
        assert cfgpath.exists();exact_file(OUTPUT,digest);assert read(cfgpath)['sha256']==digest
        from validate_submission import validate
        validate(OUTPUT)
        print('Existing output identical; no overwrite',flush=True);return
    assert not OUTPUT.with_suffix('.upload_intent.json').exists()
    temporary = OUTPUT.with_name(OUTPUT.name+'.tmp')
    with temporary.open('wb') as stream:
        stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    exact_file(temporary,digest);temporary.replace(OUTPUT)
    cfg=dict(output=str(OUTPUT),sha256=digest,base=str(BASE),base_sha256=BASE_SHA,protocol_sha256=ph,
        validation=str(PRIMARY/'validation.json'),validation_sha256=sha(PRIMARY/'validation.json'),
        independent_validation_review=str(PRIMARY_REVIEW),independent_validation_review_sha256=sha(PRIMARY_REVIEW),
        model_sha256=sha(MODEL),cue_manifest_sha256=sha(CUE_OUT/'manifest.json'),equity_manifest_sha256=sha(eqpath),
        created_at_utc=datetime.now(timezone.utc).isoformat(),batches=checks,eligible_pairs=total,baseline_lists_reproduced=total,
        changed_evidence_rows=changed,membership_changed_rows=membership_changed,risk_and_behavior_preserved=True,
        outside_scope_unchanged=True,all_evidence_memberships_preserved=(membership_changed==0),
        original_validation_gate_passed=read(PRIMARY/'validation.json')['gates']['candidate_2']['mean_screen_passed'],
        exploratory_probe_eligible=False,exception_authorized=True,exception_decision_sha256=sha(DECISION),
        exception_decision=str(DECISION),exception_document_sha256=sha(EXCEPTION_DOC),
        exception_review_sha256=sha(EXCEPTION_REVIEW),decision_policy_sha256=sha(POLICY),
        exploratory_exception=not read(PRIMARY/'validation.json')['gates']['candidate_2']['mean_screen_passed'],
        new_production_fits=3,private_rank_confirmed=False,submitted=False,
        limitations='One openly adaptive exception after the original exploratory policy and strict screen failed. Exact negative late gain remains documented; no tolerance or validation pass claim. Reused known-family labels and inherited3-versus2 cue averaging distinction. No public/private score or rank guarantee.')
    write_json(cfgpath,cfg)
    from validate_submission import validate
    validate(OUTPUT)
    print('Precise soft candidate complete',str(OUTPUT),digest,'changed',changed,'membership_changed',membership_changed,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=['prepare','fit-cues','fit-ranker','eval-equities','infer'])
    args=parser.parse_args()
    {'prepare':prepare,'fit-cues':fit_cues,'fit-ranker':fit_ranker,'eval-equities':eval_equities,'infer':infer}[args.stage]()


if __name__=='__main__': main()
