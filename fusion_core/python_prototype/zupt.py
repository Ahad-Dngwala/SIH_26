"""Zero-velocity update (ZUPT) detection.

Standard land-vehicle INS: when the vehicle is stopped, its velocity is
exactly zero, and that is a free, perfectly accurate measurement. It
costs nothing to apply and it stops an integrated-speed channel from
wandering while the vehicle sits at a traffic light - which on a city
demo route is a large fraction of the blackout.

Design note on the gate, because getting this wrong makes the filter
lie to itself: **detection is gated on raw IMU, never on the filter's
own speed estimate.** Gating on estimated speed creates a feedback loop
- the filter thinks it is slow, so ZUPT fires, so the filter becomes
more confident it is slow, so ZUPT keeps firing - and a vehicle that is
genuinely moving at 3 m/s can be pinned to a standstill by its own
estimate. The accelerometer and gyro do not care what the filter
believes, which is exactly why they are the right input.

**The hard limit this detector runs into, stated up front.** An
accelerometer cannot distinguish rest from constant velocity. That is
Galilean invariance, not a tuning problem, and no threshold fixes it.
Real-world ZUPT gets away with it because a real stationary vehicle has
a *different vibration signature* from a moving one - engine idle
versus road-induced vibration - and that difference lives in the
high-frequency content a variance test picks up.

The benchmark tool's synthetic IMU has no vibration model at all. It is
ground-truth motion plus white Gaussian noise of a fixed standard
deviation, which means its variance is identical whether the vehicle is
parked or cruising. Measured on the synthetic `varying_speed` route:
stopped gives accel variance 0.0014 and gyro variance 0.00013, while
cruising gives 0.0025 and 0.00009 over the same 1 s window - the gyro
variance is actually *lower* while moving. A variance-only detector
therefore fires continuously on synthetic data, and enabling it drove
benchmark drift from 3.91% to 59.70%.

So this detector uses two conditions, and only one of them is
trustworthy here:

* **Variance gate** - the real discriminator on real hardware,
  unverifiable on synthetic data for the reason above.
* **Magnitude gate** - horizontal specific force must be near zero. A
  stopped vehicle has no centripetal or longitudinal acceleration; a
  turning or accelerating one does. This is what makes the detector
  usable on the synthetic benchmark, but note carefully that it does
  *not* catch a vehicle cruising in a straight line at constant speed,
  which is exactly the Galilean case above.

Consequence: `zupt.enabled` defaults to **false** in the benchmark
config. ZUPT is standard, correct, cheap, and easy to defend to judges,
but it cannot be honestly validated against a synthetic IMU. Before
enabling it on the phone, record the demo vehicle actually stopped with
the engine running, measure the real variance floor, and set
`accel_var_threshold` / `gyro_var_threshold` from that recording. Ship
it enabled only once those numbers exist.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class ZuptConfig:
    """Thresholds for the zero-velocity detector.

    Defaults are tuned against the benchmark tool's synthetic IMU noise
    (`accel_noise_std: 0.05`, `gyro_noise_std: 0.01`) and are a
    starting point, not a calibrated result. Real phone IMUs in a real
    vehicle idle noisier than this; re-measure on a recording of the
    demo vehicle actually stopped before trusting these on hardware.
    """

    window_n: int = 10  # samples, 1.0 s at 10 Hz
    accel_var_threshold: float = 0.02  # (m/s^2)^2
    gyro_var_threshold: float = 0.002  # (rad/s)^2
    # Mean horizontal specific force must also be below this. See the
    # module docstring: on synthetic data this is the only condition
    # doing real work, and it does not catch straight-line cruise.
    accel_magnitude_threshold: float = 0.3  # m/s^2
    # A tight R, because a detected stop really is a near-exact
    # measurement - but not zero, because the detector can be wrong.
    r_mps: float = 0.05


class ZuptDetector:
    """Rolling-window zero-velocity detector.

    Feed it raw IMU each cycle; it reports whether the vehicle appears
    stopped. Stateful across a route - construct one per run.
    """

    def __init__(self, config: ZuptConfig | None = None) -> None:
        self.config = config or ZuptConfig()
        self._accel: deque[float] = deque(maxlen=self.config.window_n)
        self._gyro: deque[float] = deque(maxlen=self.config.window_n)

    def reset(self) -> None:
        self._accel.clear()
        self._gyro.clear()

    def update(self, accel_body: np.ndarray, gyro_yaw: float) -> bool:
        """Push one cycle of IMU and return whether the vehicle is
        currently detected as stopped.

        `accel_body` may be 2-axis or 3-axis; its magnitude is what
        gets tracked, so the axis count does not matter.
        """
        self._accel.append(float(np.linalg.norm(accel_body)))
        self._gyro.append(float(gyro_yaw))

        if len(self._accel) < self.config.window_n:
            # Not enough history to judge. Returning False - "not
            # detected as stopped" - is the safe answer: a spurious
            # early ZUPT would brake a moving filter, whereas a missed
            # one merely forgoes a correction.
            return False

        accel_var = float(np.var(self._accel))
        gyro_var = float(np.var(self._gyro))
        accel_mean = float(np.mean(self._accel))
        return (
            accel_var < self.config.accel_var_threshold
            and gyro_var < self.config.gyro_var_threshold
            and accel_mean < self.config.accel_magnitude_threshold
        )
