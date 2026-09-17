# tools/phone_replay/

Turns a session recorded by the Kotlin app into the project's first
real-world drift measurement.

## Why this is a deliverable and not debug plumbing

PS 26168's Performance Benchmark is stated over real GNSS-denied
driving: positional drift under 10% of the distance travelled during the
blackout. Every drift number this repo produced before this module was
synthetic, on a route the repo generated itself. `data/processed/` is
empty and IO-VNBD is not loaded.

A phone logging raw IMU and raw GNSS on a real road, with the blackout
applied *in software* so the withheld fixes survive as ground truth, is
a complete benchmark rig for the cost of one drive. That is the cheapest
real number available to this team, and the log format is what makes it
possible, so the format is part of the deliverable.

The measurement is computed offline from the raw log, never from what
the app displayed. A bug in the app's UI therefore cannot flatter the
result.

## Usage

```
# Make a synthetic session in the real format (tests the chain, proves nothing about accuracy)
python -m tools.phone_replay.synth_session

# Measure a session
python -m tools.phone_replay.run --session tools/phone_replay/sessions/synthetic.jsonl \
    --blackout-start 60 --blackout-end 90 --channel-a physics

# Tests
python -m pytest tools/phone_replay/test_phone_replay.py -v
```

With no `--blackout-*` arguments the tool measures whatever window the
app actually withheld, per the `withheld` flags in the log. That is the
honest default: it is the window the live filter really ran blind
through. The override exists so one recording can be re-cut into several
windows offline.

## Log format

Full schema, with the reasoning behind each field, is in
`session.py`'s module docstring. The four things that will otherwise go
wrong, in order of how often they will:

1. **Raw sensors only.** `TYPE_ACCELEROMETER` with gravity included, not
   `TYPE_LINEAR_ACCELERATION`. Mount leveling estimates the gravity
   vector and cannot recover it once a vendor filter has removed it.
   `PhoneSession.sanity_report()` detects this and there is a test for
   it, because it is the most likely app-side mistake.
2. **Raw GNSS only.** `LocationManager.GPS_PROVIDER`, never
   `FusedLocationProviderClient`. The fused provider already blends IMU,
   WiFi and cell into its position, so measuring our dead reckoning
   against it would be measuring Google's dead reckoning against ours.
3. **One clock.** Both streams timestamped from the monotonic
   elapsedRealtime base, never wall time.
4. **Start parked.** A few seconds stationary at the top of every
   session gives mount leveling a clean gravity window. There is a
   whole-session-mean fallback and the conversion report says when it
   was used, but the fallback is worse.

## Findings from building it

**Path length is the wrong denominator on real logs, and wrong in our
favour.** Distance travelled can be measured two ways: summing distances
between GNSS fixes, or integrating GNSS speed-over-ground. Independent
per-fix position noise inflates the summed path length, and a larger
denominator makes our own drift percentage *smaller*. On the synthetic
session, with 3 m per-fix noise at 1 Hz and roughly 13 m/s, the two
disagree by 12.5% - path length reads 398 m where the speed integral
reads 349 m. Android's speed-over-ground is Doppler-derived rather than
differenced from positions, so it does not share that bias.

`run.py` reports both and warns when they disagree by more than 5%, and
uses the speed integral for the headline number. A test pins the
direction of the bias so the denominator does not get "simplified" later.

**Channel P matters much more on a speed-varying route than the
synthetic benchmark suggested.** On the phone-format session, blackout
over 60-90 s:

| configuration | drift | verdict |
|---|---|---|
| no velocity channel | 36.39% | FAIL |
| Channel P | 0.46% | PASS |
| Channel P + NHC | 0.52% | PASS |

The no-channel case fails the PS bar outright, because the CTCV process
model coasts at constant speed and this route changes speed constantly.
That is a starker version of the same effect stream 1 measured on the
`varying_speed` synthetic route (10.35% versus 3.91%) and it is the
clearest evidence yet that Channel P is what carries the system today.

NHC being roughly neutral here is consistent with stream 1's root-cause
finding: with a velocity channel active, the channel's speed is already
rotated by the current heading, so NHC asserts a constraint the filter
is already applying.

**All of those numbers are from a synthetic session and prove only that
the plumbing works.** The IMU in `synth_session.py` is clean physics plus
Gaussian noise: no engine harmonics, no pothole shocks, no mount slop, no
temperature drift, and no vibration difference between parked and moving.
Those are exactly the effects PS 26168 names as the hard part, and
exactly what a real recording will introduce. Expect the real numbers to
be substantially worse, and report the real ones.

## Ground truth has a floor

The truth track is the withheld GNSS, carrying a few metres of per-fix
error. Any claimed error near or below that is not measurable with this
rig. Over a 30-60 s blackout the drift under measurement should be much
larger than the floor, so this does not invalidate the experiment - but
it does mean a sub-5-metre claim from this setup would be meaningless,
and `run.py` prints the mean reported accuracy next to the result so the
floor is always visible alongside the number.
