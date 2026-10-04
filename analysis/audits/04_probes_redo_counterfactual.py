import random
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from world import Move, Put, Undo, Redo, Swap, Merge, Split

# --- (#7) redo-validity label constancy
labels = Counter(); n_fail = 0
for seed in range(1000):
    try:
        ops, st, hist, cont, meta = build_redo_validity_example(random.Random(seed), 2, 6, [Move, Swap, Undo, Redo], 3)
        labels[meta["would_be_valid"]] += 1
    except Exception as e:
        n_fail += 1; last = repr(e)[:90]
print("redo-validity labels over 1000 seeds:", dict(labels), "build failures:", n_fail, (last if n_fail else ""))

# --- counterfactual probe sensitivity == 'last target Move'?
def S(fam,T,D=0,E=1,C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D)
CFG = {"basic_chain":(1,0),"revision":(1,0),"interleaved_chain":(3,4),"split_chain":(2,0),"merge_chain":(2,0),"swap_chain":(2,0),"undo_chain":(1,0),"undo_redo_chain":(1,0)}
print("\ncounterfactual probes (T=8): is answer_changed==True exactly when the removed op is the last target-Move (or the Put)?")
for fam,(E,D) in CFG.items():
    agree = tot = 0; sens_ops = Counter(); n_none = n_all = 0
    for seed in range(200):
        r = build_trajectory(random.Random(seed), S(fam,8,D,E))
        q = LocationQuery(r.target_obj)
        lm = max(i for i,o in enumerate(r.ops) if isinstance(o,Move) and o.obj_id==r.target_obj)
        _, fs, _ = __import__("world").replay_trace(r.ops, r.containers); orig = q.read(fs)
        for idx in range(len(r.ops)):
            cf = counterfactual_gold(r.ops, r.containers, idx, q)
            n_all += 1
            if cf is None: n_none += 1; continue
            changed = cf != orig
            pred = (idx == lm)
            tot += 1; agree += (changed == pred)
            if changed: sens_ops[type(r.ops[idx]).__name__] += 1
    print(f"  {fam:18s} valid-replay probes={tot:5d}  agree(changed == 'is last target Move')={100*agree/tot:5.1f}%   removals giving invalid replay (cf=None): {100*n_none/n_all:4.1f}%   sensitive op types: {dict(sens_ops)}")
