import random
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from world import Move, Put, Split, Merge, Swap, Undo, Redo, replay_trace, InvalidOperation

def spec_for(fam, T, D=0, E=None, C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C,
        total_updates=T+D, target_updates=T, distractor_updates=D)

CFG = {  # family -> (E, D) used for the check
 "basic_chain":(1,0), "revision":(1,0), "interleaved_chain":(3,4),
 "split_chain":(2,0), "merge_chain":(2,0), "swap_chain":(2,0),
 "undo_chain":(1,0), "undo_redo_chain":(1,0),
}
STRUCT = (Split, Merge, Swap, Undo, Redo)

def final_loc(ops, containers, tgt):
    _, st, _ = replay_trace(ops, containers)
    return st.location.get(tgt)

def run(fam, T, n=500):
    E, D = CFG[fam]
    eq_last_move = 0; built = 0; changed = 0; testable = 0; invalid = 0
    for seed in range(n):
        try:
            r = build_trajectory(random.Random(seed), spec_for(fam, T, D, E))
        except Exception as e:
            continue
        built += 1
        tgt = r.target_obj
        gold = final_loc(r.ops, r.containers, tgt)
        mv = [op for op in r.ops if isinstance(op, Move) and op.obj_id == tgt]
        if mv and mv[-1].dst == gold: eq_last_move += 1
        # structural op ablation
        sidx = [i for i,op in enumerate(r.ops) if isinstance(op, STRUCT)]
        if sidx:
            testable += 1
            any_change = False
            for i in sidx:
                ab = r.ops[:i] + r.ops[i+1:]
                try:
                    g2 = final_loc(ab, r.containers, tgt)
                except InvalidOperation:
                    invalid += 1; continue
                if g2 != gold: any_change = True
            changed += any_change
    return built, eq_last_move, testable, changed, invalid

print(f"{'family':18s} {'T':>3s} {'built':>5s} {'ans==lastMoveDst':>17s} {'struct-ablation changes ans':>28s} {'ablation invalid':>17s}")
for fam in CFG:
    for T in (2,3,4,6,8,12,16):
        b, eq, tst, ch, inv = run(fam, T)
        if b == 0: continue
        s = f"{ch}/{tst} ({100*ch/tst:.0f}%)" if tst else "n/a"
        print(f"{fam:18s} {T:3d} {b:5d} {100*eq/b:16.1f}% {s:>28s} {inv:17d}")
