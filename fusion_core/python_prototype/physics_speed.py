"""Channel P - physics-based forward-speed estimator.

This is **not** Channel A. Channel A (MIP Section 4.2) is a learned TCN
regressor whose slot in the UKF's sequential update order stays exactly
where it is; this module occupies the same *slot* with a classical,
untrained estimator so the filter has some velocity evidence during a
GNSS blackout instead of none at all.

Why it exists
-------------
`ukf.fx` is a CTCV process model. It holds speed magnitude constant
across the prediction step and only rotates it into the new
gyro-propagated heading. It never integrates accelerometer. Every
velocity *change* therefore has to arrive through a measurement update.
With Channel A and Channel B absent - which is the honest state of this
repo, see `models/README.md` - the filter has zero velocity evidence
during a blackout and simply coasts at whatever speed it last saw.

Channel P gives it something. The chain is:

1. `MountLeveling.from_stationary_window` estimates the gravity
   direction from a short stationary window at session start. This
   stands in for the untrained `alignment_net` (MIP Section 4.1) and
   makes no claim to be as good as one.
2. `MountLeveling.horizontal` rotates raw 3-axis accelerometer into a
   gravity-aligned level frame and subtracts gravity, leaving the
   horizontal specific force.
3. `estimate_forward_axis` resolves the remaining yaw ambiguity - which
   horizontal direction is "forward" - by PCA over a driving window,
   since accelerate/brake events dominate the horizontal acceleration
   variance and lie along the vehicle's longitudinal axis.
4. `PhysicsSpeedChannel` integrates that longitudinal component into a
   scalar forward speed, reseeding from GNSS speed whenever GNSS is
   available.

Steps 1-3 are the front-end and are only needed when the caller has raw
3-axis accelerometer in an unknown mount orientation (i.e. a phone).
Callers that already have a body-frame longitudinal component - the
benchmark tool's synthetic routes, or anything downstream of a real
alignment net - can skip straight to `PhysicsSpeedChannel` and feed it
that scalar.

Honest statement of what this is
--------------------------------
This channel drifts. It is an open-loop single integration of a biased
accelerometer, so its speed error grows roughly linearly with the time
since the last GNSS reseed, and the resulting position error grows
roughly quadratically. That is expected and is not a bug to be tuned
away. It is also why `PhysicsSpeedConfig.r_mps` must be an empirically
measured number and not an aspirational one: see
`tools/measure_physics_channel_r.py`, which measures it against the
benchmark route rather than asserting it.

One caveat worth stating plainly, because it decides whether this
channel helps at all on a given route: a velocity channel can only
improve on the CTCV coast when the vehicle's speed actually *changes*
during the blackout. On a perfectly constant-speed route the coast
model is already exactly right, and any noisy velocity channel -
including this one - can only make things worse. See
`fusion_core/python_prototype/README.md` for the measured numbers on
both a constant-speed and a speed-varying synthetic route.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY_MPS2 = 9.80665


# --- front-end: mount leveling (stands in for the untrained alignment_net) ---


@dataclass
class MountLeveling:
    """Gravity-aligned rotation from the phone's (unknown) mount frame
    into a level frame whose z axis points up.

    Resolves pitch and roll only. The remaining yaw degree of freedom -
    which horizontal direction the vehicle actually faces - is *not*
    determined by gravity and is resolved separately by
    `estimate_forward_axis`.
    """

    rotation: np.ndarray  # (3, 3), body -> level
    gravity_magnitude: float

    @classmethod
    def from_stationary_window(cls, accel_xyz: np.ndarray) -> "MountLeveling":
        """Estimate leveling from a window of 3-axis accelerometer
        samples taken while the vehicle is stationary.

        `accel_xyz` is (N, 3) in m/s^2, raw mount frame. At rest the
        accelerometer measures specific force, which is the reaction to
        gravity - so its mean points *up* in the world frame, not down.
        """
        accel_xyz = np.asarray(accel_xyz, dtype=float)
        if accel_xyz.ndim != 2 or accel_xyz.shape[1] != 3:
            raise ValueError(f"expected (N, 3) accelerometer window, got {accel_xyz.shape}")
        if len(accel_xyz) < 2:
            raise ValueError("need at least 2 samples to estimate a gravity vector")

        mean = accel_xyz.mean(axis=0)
        magnitude = float(np.linalg.norm(mean))
        if magnitude < 1.0:
            raise ValueError(
                f"stationary-window mean |accel| is {magnitude:.3f} m/s^2, which is "
                "not a plausible gravity magnitude - the window is probably not "
                "stationary, or the accelerometer is already gravity-compensated "
                "(in which case skip MountLeveling entirely)."
            )

        up = mean / magnitude
        # Gram-Schmidt an arbitrary body axis against `up` to get a
        # level-frame x. Which horizontal direction this lands on is
        # arbitrary and is exactly what estimate_forward_axis resolves.
        seed = np.array([1.0, 0.0, 0.0])
        if abs(float(np.dot(seed, up))) > 0.9:
            seed = np.array([0.0, 1.0, 0.0])
        x_level = seed - np.dot(seed, up) * up
        x_level /= np.linalg.norm(x_level)
        y_level = np.cross(up, x_level)

        rotation = np.stack([x_level, y_level, up], axis=0)
        return cls(rotation=rotation, gravity_magnitude=magnitude)

    def horizontal(self, accel_xyz: np.ndarray) -> np.ndarray:
        """Rotate raw accelerometer into the level frame, remove
        gravity, and return the horizontal (x, y) components.

        Accepts a single (3,) sample or an (N, 3) window; returns (2,)
        or (N, 2) correspondingly.
        """
        accel_xyz = np.asarray(accel_xyz, dtype=float)
        single = accel_xyz.ndim == 1
        if single:
            accel_xyz = accel_xyz[None, :]
        leveled = accel_xyz @ self.rotation.T
        # The level-frame z component is gravity plus any genuine
        # vertical motion; horizontal is untouched by the subtraction,
        # so we simply drop z rather than subtracting from x/y.
        horizontal = leveled[:, :2]
        return horizontal[0] if single else horizontal


def estimate_forward_axis(
    horizontal_accel: np.ndarray,
    reference_speed_delta: np.ndarray | None = None,
    lateral_regressor: np.ndarray | None = None,
) -> np.ndarray:
    """Resolve which horizontal direction is vehicle-forward.

    `horizontal_accel` is (N, 2) level-frame horizontal specific force over a
    window of *driving* (not stationary).

    Three estimators, chosen by what references are available:

    **With both `reference_speed_delta` and `lateral_regressor`** (both (N,)):
    a joint least-squares fit of the 2D horizontal accel against the two 1D
    regressors, `[speed_delta, lateral_regressor]`. The first row of the
    resulting 2x2 coefficient matrix is the forward direction; the second is
    thrown away, and doing this jointly rather than as two separate
    correlations is exactly what makes it work when the two regressors are
    themselves correlated with each other, which they are whenever speed
    changes happen mostly during turns - see below.

    **With `reference_speed_delta` only**: the single-regressor correlation
    estimator this replaced PCA with (`centered.T @ delta`, normalised).

    **With neither**: first principal component, sign left to the caller.

    Why the joint estimator exists, measured rather than asserted
    -----------------------------------------------------------------
    The single-regressor correlation estimator is a large improvement on PCA
    (see the table below it used to carry, still true), but it degrades early
    in a session for a reason of its own: acceleration and cornering are
    correlated with each other whenever a route's early turns happen to
    coincide with its early speed changes, which is common because both cluster
    around junctions. When that happens, correlating horizontal accel against
    speed delta alone partially picks up the lateral component too, because
    lateral force is itself correlated with the thing being regressed against.

    `lateral_regressor` breaks that: it is `speed * yaw_rate`, the expected
    lateral specific force from a non-slipping turn, computed from GNSS speed
    and gyro yaw rate alone - no forward axis needed to compute it, since it
    lives entirely in the level frame before any forward/lateral split exists.
    Regressing against both regressors jointly, rather than against speed delta
    with a separate lateral correction, is what lets the two disentangle each
    other even while correlated: least squares apportions each regressor's own
    slice of the variance to it, which a single correlation cannot do.

    Measured against the known mount rotation of
    `tools/phone_replay/synth_session.py`, dot product with the true forward
    axis, over the first N moving samples (this is the CAUSAL, online case - an
    offline caller with the whole session already lands at 0.9999 either way):

    | samples | speed-delta only | joint |
    |---------|-------------------|--------|
    | 2107    | 1.0000            | 1.0000 |
    | 3606    | 0.9830            | 1.0000 |
    | 5107    | 0.9825            | 1.0000 |
    | 6607    | 0.9929            | 1.0000 |
    | 8107    | 0.9930            | 1.0000 |
    | 9504    | 0.9987            | 1.0000 |
    | 15504   | 0.9999            | 1.0000 |

    The single-regressor estimator's worst causal case (0.9825, an 11-degree
    error) is exactly where an online caller's blackout is most likely to
    start, since routes are commonly built with an early acceleration phase.
    That error leaked cornering force into Channel P and pushed a synthetic
    session's blackout drift from 1.50% to 29.8% before this was diagnosed.
    Nothing about `lateral_regressor` is Channel-P-specific or
    synthetic-data-specific: it is a real, always-available regressor
    (speed times yaw rate), so there is no honest reason to leave it out even
    though the single-regressor version clears the PS 26168 bar on the
    session that happened to expose this.
    """
    horizontal_accel = np.asarray(horizontal_accel, dtype=float)
    if horizontal_accel.ndim != 2 or horizontal_accel.shape[1] != 2:
        raise ValueError(f"expected (N, 2) horizontal accel, got {horizontal_accel.shape}")
    if len(horizontal_accel) < 3:
        raise ValueError("need at least 3 samples to estimate a forward axis")

    centered = horizontal_accel - horizontal_accel.mean(axis=0)

    if reference_speed_delta is not None and lateral_regressor is not None:
        reference_speed_delta = np.asarray(reference_speed_delta, dtype=float)
        lateral_regressor = np.asarray(lateral_regressor, dtype=float)
        n = min(len(centered), len(reference_speed_delta), len(lateral_regressor))
        d = reference_speed_delta[:n] - reference_speed_delta[:n].mean()
        q = lateral_regressor[:n] - lateral_regressor[:n].mean()
        design = np.stack([d, q], axis=1)
        # np.linalg.lstsq solves design @ coef = centered; coef's row 0 is the
        # forward-axis direction (the loading on the speed-delta regressor),
        # row 1 is the lateral loading and is discarded.
        try:
            coef, _, rank, _ = np.linalg.lstsq(design, centered[:n], rcond=None)
            forward_row = coef[0]
            magnitude = float(np.linalg.norm(forward_row))
            if rank == 2 and magnitude > 1e-9:
                return forward_row / magnitude
        except np.linalg.LinAlgError:
            pass
        # Degenerate design (e.g. speed delta and lateral regressor
        # collinear, or one of them all-zero): fall through to the
        # single-regressor path below rather than returning garbage.

    if reference_speed_delta is not None:
        reference_speed_delta = np.asarray(reference_speed_delta, dtype=float)
        n = min(len(centered), len(reference_speed_delta))
        correlation_vector = centered[:n].T @ reference_speed_delta[:n]
        magnitude = float(np.linalg.norm(correlation_vector))
        scale = float(
            np.linalg.norm(centered[:n]) * np.linalg.norm(reference_speed_delta[:n])
        )
        if magnitude > 1e-9 and magnitude > 1e-6 * max(scale, 1e-12):
            return correlation_vector / magnitude

    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    axis = axis / np.linalg.norm(axis)

    if reference_speed_delta is not None:
        projected = centered @ axis
        n = min(len(projected), len(reference_speed_delta))
        correlation = float(np.dot(projected[:n], reference_speed_delta[:n]))
        if correlation < 0:
            axis = -axis

    return axis


# --- the channel itself -------------------------------------------------------


@dataclass
class PhysicsSpeedConfig:
    """Tunables for Channel P.

    `r_mps` is the 1-sigma this channel should be handed to the UKF as.
    The default here is deliberately *not* a target or an aspiration -
    it is a placeholder that callers are expected to override with a
    measured number from `tools/measure_physics_channel_r.py`. Note
    also that this channel's error is a slow drift, not white noise, so
    the Kalman independence assumption behind R is violated; a measured
    RMSE is a pragmatic proxy, and a defensible one, but it is a proxy.
    """

    r_mps: float = 2.0
    max_speed_mps: float = 70.0
    # Optional first-order leak of the integrated speed toward the last
    # GNSS-confirmed speed, as a time constant in seconds. 0.0 disables
    # it. This is a bound on how far an integrated bias can run away
    # during a long blackout, at the cost of being wrong during genuine
    # sustained acceleration. Off by default - turn it on only with a
    # measured justification.
    leak_tau_s: float = 0.0


class PhysicsSpeedChannel:
    """Integrates longitudinal acceleration into a scalar forward
    speed, reseeded from GNSS whenever GNSS is available.

    Stateful across a route; construct one per run and call `update`
    once per cycle. Returns `None` until the first GNSS reseed, because
    before that the channel has no absolute reference and must not feed
    the filter a fabricated one.
    """

    def __init__(self, config: PhysicsSpeedConfig | None = None) -> None:
        self.config = config or PhysicsSpeedConfig()
        self._speed: float | None = None
        self._last_gnss_speed: float | None = None

    @property
    def speed(self) -> float | None:
        return self._speed

    def reset(self) -> None:
        self._speed = None
        self._last_gnss_speed = None

    def reseed(self, gnss_speed: float) -> float:
        """Snap the integrated speed to a GNSS speed measurement. This
        is what keeps the integration from being open-loop over the
        whole route rather than just over each blackout."""
        self._speed = float(max(0.0, gnss_speed))
        self._last_gnss_speed = self._speed
        return self._speed

    def apply_zupt(self) -> float:
        """Force the integrated speed to zero. Called when an external
        zero-velocity detector says the vehicle is stopped. Kept
        separate from `update` so the detector's decision and this
        channel's integration stay independently testable."""
        self._speed = 0.0
        return self._speed

    def update(
        self,
        dt: float,
        longitudinal_accel: float,
        gnss_speed: float | None = None,
    ) -> float | None:
        """Advance one cycle.

        `longitudinal_accel` is the forward-axis specific force in
        m/s^2, already leveled and gravity-free. `gnss_speed` is the
        GNSS speed-over-ground in m/s when available, else None.
        """
        if gnss_speed is not None:
            return self.reseed(gnss_speed)

        if self._speed is None:
            # No absolute reference yet. Returning None is the point:
            # the caller must skip the update rather than feed the
            # filter a number this channel cannot justify.
            return None

        speed = self._speed + float(longitudinal_accel) * dt

        tau = self.config.leak_tau_s
        if tau > 0.0 and self._last_gnss_speed is not None:
            alpha = min(1.0, dt / tau)
            speed = speed + alpha * (self._last_gnss_speed - speed)

        self._speed = float(np.clip(speed, 0.0, self.config.max_speed_mps))
        return self._speed
