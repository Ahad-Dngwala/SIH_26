# tools/parity/

Cross-implementation parity fixture for the UKF fusion core.

## Why this exists

The same filter math is going to exist in three places: the Python
prototype (`fusion_core/python_prototype/ukf.py`), the C++ core
(`fusion_core/cpp/`), and the Kotlin demo app. Three implementations of
a numerically delicate filter **diverge silently**. Silently is the
important word - a subtly wrong Kalman filter does not crash, it
produces plausible trajectories that are quietly wrong, and you find
out on stage.

This fixture is the guardrail that makes "reimplement it in Kotlin" a
reasonable decision instead of a reckless one.

## What's in the fixture

`fixture_ukf_reference.json` holds, for a 60 s synthetic route with a
25 s GNSS blackout:

- `scenario` / `blackout` / `fusion_config` - everything needed to
  reproduce the run from scratch
- `initial_state` - where the filter starts
- `cycles[].input` - per-cycle `dt`, gyro yaw rate, body accel, GNSS
  availability and position
- `cycles[].expected` - the Python prototype's resulting position,
  velocity and heading for that cycle
- `tolerances` - how far a conforming implementation may deviate

Both velocity channels are absent throughout. This fixture pins **the
filter's own math**, not a channel's; a velocity channel is a separate
implementation with its own parity question.

## How another implementation uses it

1. Load the JSON.
2. Construct your filter from `initial_state` and `fusion_config`.
3. For each entry in `cycles`, feed `input` to your step function.
4. Compare your output to `expected` using `tolerances`.

Tolerances are generous enough to absorb float32-versus-float64 and
platform math-library differences, tight enough that a real logic
divergence cannot hide underneath them.

## The rule

The Python prototype is the reference **by convention**, because it is
the implementation with unit tests and measured benchmark numbers
behind it - not because it is known to be correct. If C++ or Kotlin
disagrees, go find out which one is wrong.

**Do not regenerate the fixture to make a failing port pass.** That
converts the one thing protecting three implementations into a rubber
stamp. If the Python filter changes deliberately, regenerate it in its
own commit, with the behaviour change explained in that commit message,
so the diff is reviewable as a behaviour change rather than buried.

## Commands

```
python -m tools.parity.generate_fixture   # regenerate (deliberately, see above)
python -m tools.parity.verify_fixture     # replay and check (Python reference)
tools/parity/run_kotlin_parity.sh         # replay and check (Kotlin port)
```

## The Kotlin port

`android_app/app/src/main/java/org/sih26/deadreckoning/fusion/` is checked by
`run_kotlin_parity.sh`, which compiles the fusion core with `kotlinc`, replays
all 600 cycles and exits non-zero on divergence. Measured on the committed
fixture:

| quantity | worst divergence | tolerance |
|---|---|---|
| position | 5.4e-12 m | 0.5 m |
| velocity | 3.5e-13 m/s | 0.1 m/s |
| heading | 2.5e-14 rad | 0.01 rad |

That is floating-point round-off over 600 cycles, not agreement within
tolerance. The tolerances exist to absorb platform math differences; the port
does not need them, and if a change ever starts consuming them, that is a
signal worth chasing rather than a pass.

The gate deliberately avoids Gradle. It needs `kotlinc`, a JVM and Python, so it
runs in seconds on any machine without the Android SDK. Run it before and after
touching any filter code in either language.

`export_fixture_csv.py` flattens the JSON into a CSV the harness reads, because
hand-rolling a JSON parser in Kotlin would add a second thing that can be subtly
wrong to a tool whose job is catching things that are subtly wrong. The CSV is
generated into a temporary directory at run time and never committed; the JSON
remains the single source of truth.
