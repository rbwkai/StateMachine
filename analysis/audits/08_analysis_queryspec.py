import random, statistics as st
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from analysis import QuerySpec, analyze_trajectory
from world import Move, Put, replay_trace

def S(fam,T,D=0,E=1,C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D)
CFG = {"basic_chain":(1,0),"revision":(1,0),"interleaved_chain":(3,4),"split_chain":(2,0),"merge_chain":(2,0),"swap_chain":(2,0),"undo_chain":(1,0),"undo_redo_chain":(1,0)}

print("== A. CountQuery on a trajectory containing a Move")
r = build_trajectory(random.Random(0), S("basic_chain",4))
try:
    a = analyze_trajectory(r.ops, r.containers, CountQuery("c0","key")); print("   ok", a.relevant_count)
except Exception as e:
    print("   analyze_trajectory(CountQuery) ->", type(e).__name__+":", e)
try:
    qs = QuerySpec(query_type="count")
    select_query(random.Random(1), r.ops, r.final_state, r.containers, qs)
except Exception as e:
    print("   select_query(count) ->", type(e).__name__+":", str(e)[:100])

print("\n== B. LocationQuery metrics vs measured T (T=8)")
print(f"   {'family':18s} {'T_actual':>8s} {'relevant_count':>14s} {'state_change_count':>18s} {'revision_count':>14s} {'answer_changed':>14s} {'dep_depth':>9s} {'len(ops)':>8s} {'interleave':>10s}")
for fam,(E,D) in CFG.items():
    rows=[]
    for s in range(100):
        r=build_trajectory(random.Random(s),S(fam,8,D,E))
        a=analyze_trajectory(r.ops,r.containers,LocationQuery(r.target_obj))
        m=r.measured_factors
        rows.append((m.T_actual,a.relevant_count,a.state_change_count,a.revision_count,a.answer_changed,a.dependency_depth,len(r.ops),a.interleaving_score))
    c=lambda i: st.mean(x[i] for x in rows)
    print(f"   {fam:18s} {c(0):8.1f} {c(1):14.1f} {c(2):18.1f} {c(3):14.2f} {str(all(x[4] for x in rows)):>14s} {c(5):9.1f} {c(6):8.1f} {c(7):10.2f}")

print("\n== C. revision family: can require_revision ever be satisfied?  (QuerySpec(require_revision=True))")
for fam in ("revision","basic_chain"):
    ok=0
    for s in range(200):
        r=build_trajectory(random.Random(s),S(fam,8))
        a=analyze_trajectory(r.ops,r.containers,LocationQuery(r.target_obj))
        ok+=QuerySpec(require_revision=True).matches(a)
    print(f"   {fam}: matches {ok}/200 (measure_factors V_actual on same data: {st.mean(build_trajectory(random.Random(s),S(fam,8)).measured_factors.V_actual for s in range(50)):.1f})")

