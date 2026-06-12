# 02: Bottleneck selection (queue gradient)

## The problem

Which stage is the real bottleneck? The intuitive answer ("the slowest one by
measured speed") has a trap. A stage **downstream of the true bottleneck** is
starved of input, so it completes few tasks and its measured speed looks low.
A speed-argmin selector then crowns the *victim* as the bottleneck and grows or
protects it, while the real producer in front of it never grows. The pipeline
deadlocks against its own feedback loop.

![Pipeline S0 to S3 where measured speed labels S3 as slowest; a red arrow marks S3 as the naive pick while a teal arrow marks S2 as the real constraint that is not feeding S3.](assets/02-speed-argmin-trap.png)

*The speed-argmin trap: by measured speed the starved consumer S3 looks slowest,
so a naive selector crowns the victim. The real constraint is S2, the producer
that is not feeding it.*

Low speed at a starved stage is a **supply symptom**, not a service limit.

## What we do

Pick the bottleneck from **where supply meets demand**, observed directly from
queue occupancy on both sides of each inter-stage hop. Each stage gets a queue
state (`qstate`):

| `qstate` | Meaning |
|---|---|
| `starved` | this stage's own input is below one batch: under-fed, not slow; cannot be the bottleneck. |
| `bottleneck` | this stage has at least one batch **and** its consumer has less than one batch: the supply cliff. |
| `buffered` | this stage and its consumer both have input: upstream of the cliff, not the cliff itself. |
| `balanced` | terminal stage with input and ready capacity, or no cliff is visible. |

![Four pipeline stages S0 to S3, each with an input-buffer bar measured against a shared threshold of one batch. S0, S1, and S2 sit above the threshold (qin 6, 5, 5) and are labeled buffered, buffered, and BOTTLENECK; S3 is empty (qin 0) and starved. A supply-cliff marker sits between S2 and S3, with a callout noting S2 is the deepest full stage whose consumer is below threshold.](assets/02-queue-cliff.png)

*The bottleneck is the **deepest** stage that still has a full input buffer while
its consumer is under-fed: the supply cliff. The starved stage past the cliff is
a victim of low supply, not the constraint.*

![Animated four-stage queue gradient. As the input buffers fill left to right past the one-batch threshold while S3 stays empty, the supply-cliff marker and BOTTLENECK label march from S0 to S1 to S2; the final frame settles on S2 as the deepest full stage feeding an empty S3.](assets/02-queue-gradient.gif)

*The useful signal is the **drop** between adjacent queues. When S2 has work and
S3 does not, S2 owns the cliff; S3's idleness is evidence that supply stopped
upstream, not a reason to grow S3.*

Selection is deliberately simple:

- choose the **deepest** `bottleneck` stage as the current rate-source
  candidate;
- if no stage sits at such a cliff (none has a full input queue feeding an
  under-fed consumer), fall back to the slowest measured stage by smoothed
  `cap_src` so the rate stays stable. That happens when the pipeline is still
  cold (queues have not filled yet) or fully balanced (every buffer is
  populated, so queue depth never drops sharply between a stage and its
  consumer);
- the **sizing rate** (`bottleneck_rate`) is always the slowest *measured*
  `cap_src`; a cold candidate (`cap_src = 0`) is excluded so its `0` cannot
  collapse the whole pipeline.

The bottleneck **identity** is sticky: a challenger must be decisively slower
(by `hysteresis_margin`) for a couple of confirm cycles before it becomes the
growth owner, so a one-cycle dip cannot flap it. But if the current bottleneck
goes `starved` it is replaced immediately; a starved stage is never a valid
bottleneck. Stickiness governs *identity* only; the *rate* is re-measured every
cycle.

![Flowchart of the bottleneck selection algorithm. Classify each stage by queue depth (qin vs the one-batch threshold). If any stage is at a cliff, the candidate is the deepest cliff stage; otherwise it is the slowest measured cap_src (cold or fully balanced). The bottleneck identity is sticky: it switches only when the current owner starves or a challenger stays slower for switch_confirm cycles. The sizing rate is the slowest measured cap_src, with cold zero-capacity stages excluded.](assets/02-selection-flow.png)

*The whole selection algorithm in one view: classify every stage by queue depth,
take the **deepest** cliff (or the slowest measured `cap_src` when no cliff
exists), apply the sticky-identity guard, then size from the slowest measured
`cap_src`.*

Picking the producer directly from the queue gradient sidesteps the
speed-argmin trap by construction, so the design needs no extra machinery to
correct for it: normal growth, ramp, floor, and FRAG placement handle the rest.

The division of labor is the point: **queues select** the stage to act on, and
the per-worker **speed sizes** it ([01](01-capacity-model.md)). Speed enters
*selection* only as the fallback above - when no cliff is visible at all; a low
speed at a populated-input stage never makes it the bottleneck.

## Why downstream stages idle by design

The steady-state contract is: every inter-stage buffer stays populated, the
**bottleneck runs near 100%**, and **non-bottleneck stages run below 100%**.
A stage downstream of the bottleneck can only go as fast as the bottleneck feeds it, so it will sit with `qstate=starved`, an empty input queue, and low
utilization.

This is correct. Adding workers to a starved downstream stage cannot raise
throughput; there is nothing for them to do. The scheduler instead grows the
**producer** that is the real constraint. So when you see an expensive GPU stage
idle while an upstream stage is the selected bottleneck, investigate the
upstream feeder's `cap_src` and `qstate`, not the idle stage.

## Trade-offs

| Cost | Benefit |
|---|---|
| Needs queue depth on both sides of each hop, not just per-stage speed. | Eliminates the starvation feedback trap that mislabels the victim. |
| Sticky identity delays a genuine bottleneck handoff by a couple of cycles. | The growth owner cannot flap on a one-cycle `cap_src` dip. |
| A starved expensive stage looks idle in dashboards. | That idle is the *signal* to fix the upstream feeder, not the stage. |

## Implementation pointer

- `capacity.py::classify_stages`: per-stage `qstate` from both sides of each
  hop.
- `capacity.py::_select_bottleneck_by_queue`: deepest-cliff selection, sticky
  growth-owner identity, per-regime confirm streak, immediate replacement of
  the current bottleneck when it starves.
- Decision snapshot fields: `qstate`, `bottleneck`, `bottleneck_candidate`,
  `bottleneck_candidate_rate`.
