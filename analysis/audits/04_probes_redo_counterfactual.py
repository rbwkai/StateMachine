import random
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from world import Move, Put, Undo, Redo, Swap, Merge, Split

# --- (#7) redo-validity label constancy
labels = Counter(); n_fail = 0
for seed in range(500):
    try:
        examples, _ = build_redo_validity_examples(random.Random(seed), 2, 6, [Move, Swap, Undo, Redo], 3, n_per_class=1)
        for ex in examples:
            labels[ex.would_be_valid] += 1
    except Exception as e:
        n_fail += 1; last = repr(e)[:90]
print("redo-validity labels over 1000 seeds:", dict(labels), "build failures:", n_fail, (last if n_fail else ""))

# --- counterfactual probe sensitivity == 'last target-affecting op'?
def S(fam,T,D=0,E=1,C=3):
    # Families requiring count query for causal validity
    count_families = {"split_chain", "merge_chain", "swap_chain", "undo_chain", "undo_redo_chain"}
    query_type = "count" if fam in count_families else "location"
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D, query_type=query_type)

# Families where the last target-affecting op is not a Move
NON_MOVE_FINAL_FAMILIES = {"swap_chain", "undo_chain", "undo_redo_chain"}

# Use E=1 for single-entity families, E=2 for multi-entity families
# Use D=0 for families without distractors in this audit
CFG = {"basic_chain":(1,0),"revision":(1,0),"interleaved_chain":(3,4),"split_chain":(2,0),"merge_chain":(2,0),"swap_chain":(2,0),"undo_chain":(1,0),"undo_redo_chain":(1,0)}
print("\ncounterfactual probes (T=8): is answer_changed==True exactly when the removed op is the last target-affecting op?")
for fam,(E,D) in CFG.items():
    agree = tot = 0; sens_ops = Counter(); n_none = n_all = 0
    for seed in range(200):
        r = build_trajectory(random.Random(seed), S(fam,8,D,E))
        q = LocationQuery(r.target_obj)
        # Find last target-affecting operation
        target_ops = []
        for i, o in enumerate(r.ops):
            obj_id = getattr(o, 'obj_id', None)
            if obj_id is None:
                obj_id = getattr(o, 'source_obj_id', None)
            if obj_id is None:
                obj_id = getattr(o, 'new_obj_id', None)
            if obj_id == r.target_obj:
                target_ops.append(i)
        if not target_ops:
            continue
        lm = target_ops[-1]
        _, fs, _ = __import__("world").replay_trace(r.ops, r.containers); orig = q.read(fs)
        for idx in range(len(r.ops)):
            cf = counterfactual_gold(r.ops, r.containers, idx, q)
            n_all += 1
            if cf is None: n_none += 1; continue
            changed = cf != orig
            pred = (idx == lm)
            tot += 1; agree += (changed == pred)
            if changed: sens_ops[type(r.ops[idx]).__name__] += 1
    print(f"  {fam:18s} valid-replay probes={tot:5d}  agree(changed == 'is last target-affecting op')={100*agree/tot:5.1f}%   removals giving invalid replay (cf=None): {100*n_none/n_all:4.1f}%   sensitive op types: {dict(sens_ops)}")
