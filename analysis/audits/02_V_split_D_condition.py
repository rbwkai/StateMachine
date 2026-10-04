import random, statistics as st
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from generator.dataset_spec import Condition, Experiment
from world import Move, Put, Split, Merge, Swap, Undo, Redo

def S(fam,T,D=0,E=1,C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D)

print("== odd/even T for swap and undo_redo: does answer == last Move dst?")
for fam,E in (("swap_chain",2),("undo_redo_chain",1),("undo_chain",1),("split_chain",2),("merge_chain",2)):
    row=[]
    for T in range(2,13):
        ok=tot=0
        for seed in range(200):
            try: r=build_trajectory(random.Random(seed), S(fam,T,0,E))
            except Exception: continue
            tot+=1
            mv=[o for o in r.ops if isinstance(o,Move) and o.obj_id==r.target_obj]
            ok += bool(mv) and mv[-1].dst==r.final_state.location[r.target_obj]
        row.append(f"T{T}:{100*ok//tot if tot else '-'}")
    print(f"  {fam:16s}", " ".join(row))

print("\n== V (measured revisits) basic vs revision, 3 containers")
for T in (4,6,8,12,16):
    vb=[build_trajectory(random.Random(s),S("basic_chain",T)).measured_factors.V_actual for s in range(300)]
    vr=[build_trajectory(random.Random(s),S("revision",T)).measured_factors.V_actual for s in range(300)]
    print(f"  T={T:2d}  basic mean V={st.mean(vb):.2f} (min {min(vb)}, max {max(vb)})   revision mean V={st.mean(vr):.2f} (min {min(vr)}, max {max(vr)})   T-2={T-2}")
# does revision_count change anything?
a=build_trajectory(random.Random(3),TrajectorySpec(family="revision",entity_count=1,total_updates=8,target_updates=8,revision_count=0))
b=build_trajectory(random.Random(3),TrajectorySpec(family="revision",entity_count=1,total_updates=8,target_updates=8,revision_count=5))
print("  revision_count=0 vs 5 identical ops:", a.ops==b.ops)
# 4 containers
for C in (3,4,5):
    vb=[build_trajectory(random.Random(s),S("basic_chain",12,C=C)).measured_factors.V_actual for s in range(200)]
    vr=[build_trajectory(random.Random(s),S("revision",12,C=C)).measured_factors.V_actual for s in range(200)]
    print(f"  C={C} T=12: basic V={st.mean(vb):.2f}  revision V={st.mean(vr):.2f}")

print("\n== split_chain: requested vs measured and verify_factors")
for T in (3,4,8):
    r=build_trajectory(random.Random(0),S("split_chain",T,0,2)); m=r.measured_factors
    print(f"  requested T={T} D=0 E=2 -> measured T={m.T_actual} D={m.D_actual} E={m.E_actual}", end="  ")
    try: verify_factors(2,T,0,m,"split_chain"); print("verify OK")
    except AssertionError as e: print("verify raises:", str(e)[:70])
r=build_trajectory(random.Random(0),S("split_chain",2,0,2)); print("  T=2 ops:", [type(o).__name__ for o in r.ops])

print("\n== undo/undo_redo: requested D ignored")
for fam in ("undo_chain","undo_redo_chain"):
    r=build_trajectory(random.Random(0),S(fam,6,4,1)); m=r.measured_factors
    print(f"  {fam}: requested D=4 -> measured D={m.D_actual}; ops={[type(o).__name__ for o in r.ops]}")

print("\n== Condition D>=T")
for T,D in ((8,0),(8,4),(8,8),(8,16),(16,16),(4,8)):
    try: Condition(family="interleaved_chain",T=T,E=3,D=D,experiment=list(Experiment)[0]); print(f"  T={T} D={D}: ok")
    except ValueError as e: print(f"  T={T} D={D}: ValueError({e})".replace("\n"," "))
print("  Experiment members:", [e.name for e in Experiment])
