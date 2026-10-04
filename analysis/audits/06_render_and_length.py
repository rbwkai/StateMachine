import random, re
from collections import Counter
from generator import *
from generator.trajectory_specs import TrajectorySpec
from render import NameRegistry, render_narrative, make_distractor_sentences, splice_distractors, question_location
from world import Move, Put

def S(fam,T,D=0,E=1,C=3):
    return TrajectorySpec(family=fam, entity_count=E, num_containers=C, total_updates=T+D, target_updates=T, distractor_updates=D)

def make(fam,T,D,E,seed,N=0):
    rng=random.Random(seed)
    r=build_trajectory(rng,S(fam,T,D,E))
    names=NameRegistry(rng,r.containers)
    sents,final=render_narrative(r.ops,r.containers,names)
    if N:
        used=[final.object_type[o] for o in final.object_type]
        ds=make_distractor_sentences(rng,N,names,used)
        sents=splice_distractors(rng,sents,ds)
    q=question_location(r.target_obj,final,names)
    gold=names.container(final.location[r.target_obj])
    return r,names,sents,q,gold

for fam,E,D in (("basic_chain",1,0),("revision",1,0),("split_chain",2,0),("merge_chain",3,0),("swap_chain",3,0),("undo_chain",1,0),("undo_redo_chain",1,0),("interleaved_chain",3,4)):
    r,names,sents,q,gold=make(fam,6,D,E,5)
    print(f"--- {fam}  (gold: {gold})"); print("   "+"\n   ".join(sents)); print("   Q:",q)

# (1) ordinal bug + naming of same-type distinct objects
bad=0; dupl=0; tot=0
for seed in range(400):
    r,names,sents,q,gold=make("interleaved_chain",8,6,5,seed)
    tot+=1
    txt=" ".join(sents)+" "+q
    if re.search(r"the \d+th ",txt): bad+=1
    if "the duplicate " in txt or "the original " in txt: dupl+=1
print(f"\ninterleaved E=5: narratives with malformed ordinal like 'the 3th': {bad}/{tot}; narratives using 'original/duplicate' wording for independently Put objects: {dupl}/{tot}")

# (2) length: words & CoT steps needed
print("\nlength check (words in narrative; events = sentences incl. distractors)")
for T,D,N in ((8,0,0),(16,0,0),(16,16,0),(16,16,16),(12,8,8)):
    fam="interleaved_chain" if D else "basic_chain"; E=3 if D else 1
    try:
        r,names,sents,q,gold=make(fam,T,D,E,1,N)
    except Exception as e: print("  ",T,D,N,"ERR",e); continue
    words=sum(len(s.split()) for s in sents)
    print(f"   T={T:2d} D={D:2d} N={N:2d}: sentences={len(sents):3d} words={words:4d}  (~{int(words*1.35)} tokens)   CoT steps required=1 per event => ~{len(sents)*11} output tokens vs max_new_tokens=256 (config) / 128 (engine default)")
