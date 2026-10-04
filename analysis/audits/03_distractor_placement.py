import random, statistics as st
from generator import *
from generator.trajectory_specs import TrajectorySpec
from world import Move, Put, Split, Merge, Swap, Undo, Redo

def S(fam,T,D=0,E=1,C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D)

def tail_frac(r):
    """fraction of non-target 'distractor' ops that occur after the last op that touches the target's location"""
    # target-affecting op indices measured by replay
    from world import apply_op, History, WorldState
    state=WorldState({}, {}, set(r.containers), 0); h=History(); last=-1; dis=[]
    for i,op in enumerate(r.ops):
        if isinstance(op,Put): state=apply_op(op,state,h); continue
        before=state.location.get(r.target_obj); state=apply_op(op,state,h)
        if before!=state.location.get(r.target_obj): last=i
        else: dis.append(i)
    return (sum(1 for i in dis if i>last)/len(dis)) if dis else None

for fam,E in (("interleaved_chain",3),("merge_chain",3),("swap_chain",3),("split_chain",2)):
    print("==",fam)
    for T,D in ((4,2),(4,4),(8,4),(8,8),(8,16),(4,16),(12,4),(16,16)):
        fr=[];  fail=0
        for s in range(300):
            try: r=build_trajectory(random.Random(s),S(fam,T,D,E)); 
            except Exception as e: fail+=1; msg=str(e)[:60]; continue
            f=tail_frac(r)
            if f is not None: fr.append(f)
        print(f"   T={T:2d} D={D:2d}: builds={300-fail:3d}/300 " + (f"mean fraction of distractors AFTER last target change = {st.mean(fr):.2f}" if fr else f"({msg if fail else 'no distractors'})"))
