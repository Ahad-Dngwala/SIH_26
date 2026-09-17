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
python -m tools.parity.verify_fixture     # replay and check
```
