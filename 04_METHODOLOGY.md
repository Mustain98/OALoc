# Methodology — Quality-Adaptive LLM Bug Localization

> The method end to end: what it does, why each step exists, and how the experiment
> proves the claim. This is the version to be able to say out loud in the defense.

---

## The hypothesis the method tests

One sentence underneath everything: **localization effort should scale with how much signal
the bug report carries.** A clear report (stack trace, code, repro steps) needs little
searching; a vague one needs more. Current systems ignore this and spend a fixed amount on
every report. So the method is a controlled way to test:

> *Does making effort a function of report quality buy accuracy on hard reports and/or save
> cost on easy ones?*

Everything in the pipeline exists to make one variable — **effort** — respond to one signal
— **report quality** — while holding everything else constant.

---

## Pipeline overview

```
bug report ─► (1) QUALITY SCORER ─► score 0..4
                                        │
                                        ▼
                              (2) BUDGET POLICY ─► budget {candidates, hops, samples}
                                        │
repo @ commit ─► (3) CODE GRAPH ────────┤
                                        ▼
                              (4) LOCALIZER AGENT LOOP ─► ranked files & functions (+cost)
                                        │
                                        ▼
                              (5) EVALUATION ─► Acc@k, MRR, cost  →  accuracy-vs-cost plot
```

---

## The pipeline, stage by stage

### Stage 1 — Quality scoring
The bug report text goes to a scorer that returns an integer **0–4**. It counts objective,
presence-based features:

- stack trace / traceback present?
- code snippet present?
- names a class or method explicitly?
- steps-to-reproduce / expected-vs-actual behaviour?

Each present feature adds a point. It is deliberately **rule-based and cheap** — no model
call, fully deterministic — because the score is the load-bearing part of the novelty and it
must be *objective and defensible* (features come straight from the report-quality
literature: Fang et al., Chaparro et al.). An optional LLM-based scorer is kept as a
comparison point, but the rule-based one is the default.

*Why it exists:* it turns "report quality," which sounds subjective, into a number a
committee can't argue with.

### Stage 2 — Budget policy
The score maps to a **budget** — three dials: how many candidate entities to keep, how many
graph hops to traverse, how many agent samples to run.

| Report quality | max_candidates | max_hops | max_samples |
|---|---|---|---|
| low (0–1)      | 50 | 3 | 3 |
| medium (2)     | 25 | 2 | 2 |
| high (3–4)     | 10 | 1 | 1 |

Low score → big budget; high score → small budget. The mapping is a lookup table in
`config.yaml`, not buried in code, so it can be tuned and its sensitivity reported.

*Why it exists:* this is the mechanism that converts the quality signal into actual compute.
It is the genuinely new part — turning AutoFL's *emergent* effort variation into an
*explicit, signal-driven* policy.

### Stage 3 — Graph construction
Independently, the target repository (checked out at the exact **pre-fix commit**) is parsed
into a code graph: nodes are files/classes/functions; edges are
`contain` / `import` / `invoke` / `inherit`. This is the LocAgent backbone — it lets the
localizer navigate a repo far too large for a context window, and follow *dependency* links
rather than just text matches. Built once per repository.

*Why it exists:* it is how the system reaches buggy code the report doesn't literally name —
the code a few call-graph hops away from the symptom.

### Stage 4 — Localizer agent loop
The report plus its budget drive the search. **One pass** =

1. the LLM extracts keywords from the report;
2. those seed `search_entity` (BM25 over graph nodes);
3. `traverse_graph` expands from the seeds out to `max_hops`, rendered as a readable tree;
4. the LLM reads report + tree and ranks the most suspicious `file:function` entities.

The loop repeats **`max_samples`** times and merges runs by **reciprocal-rank voting** (an
entity ranked high across several runs floats to the top). Output: ranked files and
functions, plus token/dollar cost.

The three budget dials (candidates, hops, samples) **are** the effort knobs. On a low-quality
report the loop searches wider, walks farther, and votes across more runs; on a high-quality
one it does the minimum. **That difference is the entire method.**

*Why the multi-sample step:* it improves stability and is itself a knob — running a vague
report 3× and aggregating is one concrete way "more effort" cashes out. Borrowed from
AutoFL's confidence aggregation (honest to cite).

---

## The experimental design (what a committee actually grades)

Run the **same** system twice on the **same** instances, changing exactly one thing:

- **Baseline (control):** every report gets a **fixed** budget (Top-15, 2 hops, 1 sample) —
  what LocAgent / BLAgent effectively do.
- **Adaptive (treatment):** each report's budget is **chosen by its quality score**.

The graph, tools, model, prompts, and dataset are identical between the two runs, so any
difference in accuracy or cost is attributable to the adaptation and nothing else. That is
what makes it a **controlled experiment**, not a demo — and it is the answer to "how do you
know the effect is real?"

---

## How success is measured

Two standard metrics, at file level (primary) and function level (secondary):

- **Acc@k** — did *all* gold files land in the top-k (strict, matches LocAgent)? Report
  k = 1, 3, 5.
- **MRR** — how high the first correct file ranks.
- **Cost** — average tokens and USD per instance.

**Ground truth** comes from the fixing patch in SWE-bench Lite: the files/functions the real
fix touched are the gold locations. The patch is used **only** to derive gold labels in the
evaluator — it never touches the localizer.

**Headline result:** a single **accuracy-vs-cost plot** — baseline point vs adaptive point.
**The money result:** the **breakdown by quality bucket** — adaptive should show its biggest
accuracy gain on the **low-quality** reports, the failure class current tools ignore. If that
bucket moves, the thesis is demonstrated; if only cost drops, you have the efficiency half.

---

## Honest methodological points (say these before they're asked)

- **What's held constant vs varied.** One variable (the budget-selection rule); everything
  else fixed. State it explicitly — it is the validity argument.
- **The scorer is the risk.** If the quality score is noisy, reports route to the wrong
  budget and you lose on both axes. So the method's first internal check is: *does the score
  correlate with localization difficulty on our data?* Validate that **before** claiming the
  adaptive policy works — it is step one of the results, not an assumption.
- **This is a controlled comparison, not a SOTA chase.** The agent loop is intentionally
  simple. The claim is not "we beat LocAgent's 92.7%"; it is that *within one system*,
  quality-adaptive effort dominates fixed effort on the accuracy–cost trade-off. Scoped that
  way, it is defensible.

---

## Relation to prior work (one line each)

- **LocAgent (ACL 2025):** supplies the graph + tools backbone; spends fixed effort per
  report — the gap this method fills.
- **AutoFL (FSE 2024):** shows effort *already* varies with difficulty, but only emergently;
  this method makes it explicit and signal-driven.
- **Fang et al. (Soft Computing 2021):** conditions on report quality by *gating* (skip bad
  reports); this method instead *adapts effort to recover* them — opposite direction.
