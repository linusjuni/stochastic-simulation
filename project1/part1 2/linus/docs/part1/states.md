# The states of the model

The model follows a woman month by month after breast tumor surgery. At any
month she is in exactly one of 5 states.

| # | Name               | Meaning                                                  |
|---|--------------------|----------------------------------------------------------|
| 1 | NED                | No evidence of disease — cancer-free since surgery       |
| 2 | Local recurrence   | Cancer reappeared near the original surgery site         |
| 3 | Distant metastasis | Cancer reappeared elsewhere in the body                  |
| 4 | Local + distant    | Both local recurrence and distant metastasis present     |
| 5 | Death              | Absorbing — once reached, never left                     |

## Transition diagram

Edge labels are the monthly transition probabilities from `P` (self-loops,
i.e. "stay in the same state", are omitted for readability).

```mermaid
stateDiagram-v2
    state "1: NED" as S1
    state "2: Local recurrence" as S2
    state "3: Distant metastasis" as S3
    state "4: Local + distant" as S4
    state "5: Death" as S5

    [*] --> S1
    S1 --> S2 : 0.005
    S1 --> S3 : 0.0025
    S1 --> S5 : 0.001
    S2 --> S3 : 0.005
    S2 --> S4 : 0.004
    S2 --> S5 : 0.005
    S3 --> S4 : 0.003
    S3 --> S5 : 0.005
    S4 --> S5 : 0.009
    S5 --> [*]
```

## Why state 4 is separate from 2 and 3

Health only gets worse in this model, never better. The allowed transitions
are 1→2, 1→3, 2→3, 2→4, 3→4, and any state →5. There is no direct 1→4
transition (`P[1,4] = 0`): a woman can only reach "both" by first having one
of local recurrence or distant metastasis, and then acquiring the other.

## State 5 is absorbing

Row 5 of `P` is `0, 0, 0, 0, 1` — death transitions to death with
probability 1. This is what makes simulation terminate: once a simulated
woman reaches state 5, she stays there, so we stop simulating her and record
the number of months it took as her lifetime.
