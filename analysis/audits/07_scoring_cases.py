from eval.scoring import score_prediction, extract_instance_answer, candidate_answers
from eval.eval_harness import evaluate_predictions

inst = {"instance_id":"i1","family":"basic_chain","requested_factors":{"T":4,"D":0},
        "gold_answer":"the red bag",
        "step_wise_gold_answers":["the dusty shelf","the red bag","the metal chest","the red bag","the metal chest"][:4]+["the red bag"],
        "final_state":{"container_names":{"c0":"the dusty shelf","c1":"the red bag","c2":"the metal chest"}}}
print("candidates:", candidate_answers(inst))

cases = {
 "direct: proper 'Final Answer:' line":            "Final Answer: the red bag",
 "direct: bare answer, no prefix":                 "the red bag",
 "direct: sentence answer, no prefix":             "The token is now in the red bag.",
 "CoT: complete, correct":                         "Step 1: the dusty shelf\nStep 2: the red bag\nFinal Answer: the red bag",
 "CoT: truncated (no Final Answer), last step ok": "Step 1: the dusty shelf\nStep 2: the red bag\nStep 3: the metal chest\nStep 4: the red bag",
 "CoT: correct final but no 'Step k' lines":       "The token ends up in the red bag.\nFinal Answer: the red bag",
 "echoes template placeholder":                    "Final Answer: <name of the container>",
 "correct answer then base-model continuation":    "Final Answer: the red bag\n\nNarrative:\nA pen was placed in the old box.\n\nQuestion:\nWhere is the pen now?\n\nFinal Answer: the old box",
 "markdown bold":                                  "**Final Answer:** the red bag",
 "answer names two containers":                    "Final Answer: moved from the metal chest to the red bag",
 "hedged: mentions wrong+right":                   "Final Answer: the red bag (not the metal chest)",
 "answer on next line":                            "Final Answer:\nthe red bag",
 "wrong container":                                "Final Answer: the metal chest",
}
print(f"\n{'case':50s} {'extracted':14s} {'method':14s} {'semantic':>8s} {'strict(=is_correct)':>20s}")
for name, raw in cases.items():
    for cot in (False,):
        s = score_prediction({"raw_prediction": raw}, inst, chain_of_thought=cot)
        print(f"{name:50s} {s['extracted_answer'][:13]:14s} {s['extraction_method']:14s} {str(s['semantic_correct']):>8s} {str(s['strict_correct']):>20s}")

# evaluate_predictions: is chain_of_thought honoured? missing preds? duplicate ids?
print("\nevaluate_predictions():")
r = evaluate_predictions([inst], [{"instance_id":"i1","raw_prediction":"The token is in the red bag.\nFinal Answer: the red bag"}])
print("  CoT-style output w/o Step lines ->", r["instance_results"][0]["is_correct"], "(chain_of_thought flag is never passed, so Step-line compliance is not enforced)")
r = evaluate_predictions([inst], [])
print("  missing prediction -> counted as incorrect:", r["overall_total"], r["overall_correct"], "(no error / no exclusion)")
r = evaluate_predictions([inst],[{"instance_id":"i1","raw_prediction":"Final Answer: the red bag"},{"instance_id":"i1","raw_prediction":"Final Answer: the metal chest"}])
print("  duplicate instance_id (e.g. two models/modes in one list): last wins ->", r["instance_results"][0]["is_correct"])
inst2 = dict(inst); inst2.pop("step_wise_gold_answers")
import logging; logging.disable(logging.CRITICAL)
print("  candidates when step_wise_gold_answers absent (dataset_context):", candidate_answers(inst2, dataset_context=[inst2,{"family":"basic_chain","requested_factors":{"T":4},"gold_answer":"the old box"},{"family":"basic_chain","requested_factors":{"T":4},"gold_answer":"the red bag"}]))
