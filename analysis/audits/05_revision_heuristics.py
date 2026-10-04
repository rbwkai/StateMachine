import random
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from world import Move, Put

def S(fam,T,D=0,E=1,C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D)

def heur(r):
    tgt=r.target_obj
    put=[o for o in r.ops if isinstance(o,Put) and o.obj_id==tgt][0]
    mv=[o for o in r.ops if isinstance(o,Move) and o.obj_id==tgt]
    gold=r.final_state.location[tgt]
    locs=[put.container]+[m.dst for m in mv]
    # source of last move = previous location in the move chain
    freq=Counter(locs).most_common()
    top=[c for c,n in freq if n==freq[0][1]]
    return {
      "last_dst": mv[-1].dst==gold,
      "last_src(=prev loc)": locs[-2]==gold,
      "start(Put)": put.container==gold,
      "modal loc": (len(top)==1 and top[0]==gold),
      "unseen-in-last-3": False,
    }

print("share of instances where the cheap heuristic equals the gold answer (3 containers; chance=33%)")
print(f"{'family':10s} {'T':>3s} " + " ".join(f"{k:>20s}" for k in ("last_dst","last_src(=prev loc)","start(Put)","modal loc")))
for fam in ("basic_chain","revision"):
    for T in (4,6,8,12,16):
        agg=Counter(); n=300
        for s in range(n):
            r=build_trajectory(random.Random(s),S(fam,T))
            for k,v in heur(r).items(): agg[k]+=v
        print(f"{fam[:9]:10s} {T:3d} "+" ".join(f"{100*agg[k]/n:19.0f}%" for k in ("last_dst","last_src(=prev loc)","start(Put)","modal loc")))
