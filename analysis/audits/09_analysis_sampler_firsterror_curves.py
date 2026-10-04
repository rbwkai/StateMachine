import random, statistics as st, math
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from analysis import QuerySpec, analyze_trajectory, best_fitting_curve, compare_curves, fit_sigmoid, fit_exponential, compute_failure_onset, analyze_first_error
from world import Put, Move, Remove, Split, Merge, Swap, Undo, Redo, replay_trace, GenerationError

print("== D. sampler -> select_query path with the real QuerySpec (location, defaults)")
enabled=[Put,Move,Remove,Split,Merge,Swap,Undo,Redo]
ok=fail=0; ans_changed=Counter(); sel_none_initial=0; ncont=Counter(); sc_vs_nonput=[]
for s in range(400):
    rng=random.Random(s)
    try:
        ops,fs,h,c=sample_sequence(rng,3,8,enabled)
        q,a=select_query(rng,ops,fs,c,QuerySpec(query_type="location"))
    except GenerationError as e:
        fail+=1; continue
    ok+=1; ans_changed[a.answer_changed]+=1; ncont[len(c)]+=1
    sc_vs_nonput.append((a.state_change_count, sum(not isinstance(o,Put) for o in ops)))
print(f"   built={ok} GenerationError={fail}; answer_changed={dict(ans_changed)}; containers={dict(ncont)}")
print(f"   mean state_change_count={st.mean(x[0] for x in sc_vs_nonput):.2f}  vs mean non-Put ops={st.mean(x[1] for x in sc_vs_nonput):.2f}")

print("\n   must_change_from_initial=False can ever match? ", end="")
m=0
for s in range(400):
    rng=random.Random(s)
    try:
        ops,fs,h,c=sample_sequence(rng,3,8,enabled)
        for q in candidate_queries(rng,ops,fs):
            a=analyze_trajectory(ops,c,q)
            if QuerySpec(must_change_from_initial=False).matches(a): m+=1; break
    except GenerationError: pass
print(f"{m}/400 trajectories have any object satisfying it")

print("\n   min_undo=1 : fraction of sampled trajectories where a query exists:", end=" ")
hit=0;n=0
for s in range(300):
    rng=random.Random(s)
    try:
        ops,fs,h,c=sample_sequence(rng,3,8,enabled); n+=1
        select_query(rng,ops,fs,c,QuerySpec(min_undo=1)); hit+=1
    except GenerationError: pass
print(f"{hit}/{n}")

print("\n== E. first_error + step_wise_gold alignment")
S=lambda fam,T,D,E: TrajectorySpec(family=fam,entity_count=E,num_containers=3,total_updates=T+D,target_updates=T,distractor_updates=D)
r=build_trajectory(random.Random(4),S("interleaved_chain",4,4,3))
q=LocationQuery(r.target_obj)
g=step_wise_gold(r.ops,r.containers,q)
print("   ops:",len(r.ops),"| step_wise_gold len:",len(g),"| first 4 entries:",g[:4])
print("   -> entries are container IDs ('c1') and per-op (None until the target is Put);")
print("      narration/prompt asks for one 'Step k' per EVENT with display names ('the red bag').")
a=analyze_first_error(["c0","c1","c1","c2"],["the a","the b","the b","the c"])
print("   analyze_first_error(ids vs display names) ->", a.error_type.name, "first_error_step", a.first_error_step, "(every step 'wrong' unless names are mapped to IDs upstream; no such mapping exists in eval/analysis)")
print("   gold[0] for a trajectory whose first op is not the target's Put:", step_wise_gold([o for o in r.ops],r.containers,LocationQuery(r.target_obj))[0], "| any None in gold:", any(x is None for x in g))

print("\n== F. curve fitting")
xs=[2,4,6,8,12]  # 5 points
# F1: sigmoid floor
true=lambda x:0.33+(0.98-0.33)/(1+math.exp(0.6*(x-8)))
ys=[true(x) for x in xs]
fits=compare_curves(xs,ys)
print("   noise-free sigmoid with chance floor c=0.33 (a=0.98,b=0.6,x0=8):")
for k,v in fits.items(): print(f"      {k:11s} R2={v.r_squared:.3f} AIC={v.aic:7.2f} params={ {a:round(b,2) for a,b in v.params.items()} }")
print("   -> grid caps c at 0.2 (sigmoid) / 0.25 (exponential), so a 1/3 chance floor is unreachable; fitted params are biased")
# F2: model selection under AIC with n=5 (k=4 for sigmoid)
random.seed(0); sel=Counter()
for _ in range(2000):
    ys=[min(1,max(0,0.95-0.04*x+random.gauss(0,0.03))) for x in xs]   # truly linear + noise
    sel[best_fitting_curve(xs,ys)[0]]+=1
print("   truly LINEAR data + N(0,.03) noise, n=5 points; model chosen by AIC over 2000 draws:", dict(sel))
sel=Counter()
for _ in range(2000):
    ys=[min(1,max(0,0.95-0.04*x+random.gauss(0,0.03))) for x in [4,8,12,16]]
    sel[best_fitting_curve([4,8,12,16],ys)[0]]+=1
print("   same, n=4 points (sigmoid has 4 params -> can interpolate):", dict(sel))
# F3: onset on a non-monotonic curve (revision's T mod 6 pattern)
print("   compute_failure_onset on sawtooth accuracies [.95,.40,.92,.90,.35] over T=[3,4,6,8,12] ->", compute_failure_onset([3,4,6,8,12],[.95,.40,.92,.90,.35]), "(first dip only; ignores recovery)")
