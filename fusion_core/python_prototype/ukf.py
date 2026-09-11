"""Unscented Kalman Filter fusion core - MIP Section 5.

State vector (Section 5.1), 7 states:
    x = [pn, pe, vn, ve, psi, ba, bg]
    pn, pe   - position north/east, meters, local tangent frame
    vn, ve   - velocity north/east, m/s
    psi      - heading, radians
    ba       - residual accelerometer bias (reserved; not yet an
               input to the process model below, see module docstring
               note on Channel A/B already being bias-corrected)
    bg       - residual gyro bias, rad/s, subtracted from the raw yaw
               rate before it drives the heading propagation

Process model (Section 5.2): constant-velocity, constant-turn-rate
(CTCV) kinematic model. Heading is propagated from the (bias-
corrected) gyro yaw rate; speed magnitude is held constant across the
prediction step and only rotated into the new heading direction -
velocity *changes* come from the measurement updates (GNSS, Channel
A/B), not from integrating raw accelerometer here. This deliberately
avoids double-integrating accelerometer through the process model,
which is exactly the quadratic-drift failure mode HLD/main.tex Section
1 describes; that integration only happens in
tools/benchmark_replay/pipeline.py's *raw dead-reckoning baseline*,
never here.

Measurement/update sources (Section 5.3) - four independent evidence
sources, applied as sequential UKF updates each cycle (not one stacked
vector), since they arrive on different cadences and some are absent
on a given cycle:

1. GNSS position + velocity, when available.
2. Channel A forward-speed estimate, rotated into north/east using the
   filter's current heading estimate before being applied as a
   pseudo-measurement of (vn, ve).
3. Channel B forward-speed estimate, same rotation, higher R.
4. Road-signature anchor position (segment midpoint), applied as a
   soft position correction only when classifier confidence clears the
   threshold (Section 5.3 point 4 / Section 4.4's 0.85 deployment
   rule).

Trust weighting (Section 5.4): a simple gain-scheduling rule, not a
learned network, per the MIP's own instruction ("keep it that way for
the first version, it is easier to debug and defend to judges"). When
Channel A and Channel B disagree by more than
`channel_a_b_disagreement_factor` times Channel B's own typical RMSE,
Channel A's R is inflated for that cycle only.

GNSS re-admission ramp (Section 5.5 step 5): after GNSS reacquires
following a blackout, its measurement trust is ramped from near-zero
to full over `gnss_reacquire_ramp_s` seconds rather than snapping back
instantly, to avoid a visible position jump if the INS-only estimate
drifted during the blackout.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from filterpy.kalman import MerweScaledSigmaPoints, UnscentedKalmanFilter

N_STATES = 7
# Indices into the state vector, named for readability at call sites.
PN, PE, VN, VE, PSI, BA, BG = range(N_STATES)


def fx(x: np.ndarray, dt: float, gyro_yaw: float) -> np.ndarray:
    """Process model (Section 5.2): CTCV kinematic propagation.

    `gyro_yaw` is the raw gyro yaw-rate reading for this step; the
    filter's own bias estimate (state BG) is subtracted here so the
    bias term actually participates in the heading propagation it is
    meant to correct.
    """
    x_new = x.copy()
    yaw_rate = gyro_yaw - x[BG]
    psi_new = x[PSI] + yaw_rate * dt

    speed = float(np.hypot(x[VN], x[VE]))
    vn_new = speed * np.cos(psi_new)
    ve_new = speed * np.sin(psi_new)

    x_new[PN] = x[PN] + (x[VN] + vn_new) / 2.0 * dt
    x_new[PE] = x[PE] + (x[VE] + ve_new) / 2.0 * dt
    x_new[VN] = vn_new
    x_new[VE] = ve_new
    x_new[PSI] = psi_new
    # ba, bg: constant in the process model (random-walk drift is
    # handled by their entries in Q, not by anything here).
    return x_new


def hx_position(x: np.ndarray) -> np.ndarray:
    return np.array([x[PN], x[PE]])


def hx_position_velocity(x: np.ndarray) -> np.ndarray:
    return np.array([x[PN], x[PE], x[VN], x[VE]])


def hx_velocity(x: np.ndarray) -> np.ndarray:
    return np.array([x[VN], x[VE]])


def residual_angle_safe(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Default residual (plain subtraction) is fine for every
    measurement this filter uses (positions and velocities, never
    heading directly) - filterpy's default already does this, this
    helper exists only so a future heading-measurement source has an
    obvious place to add wraparound-safe handling without touching the
    other update calls."""
    return a - b


@dataclass
class FusionConfig:
    """Tunable parameters, defaults per Section 5.5 step 3 / Section
    5.4 / Section 5.5 step 5."""

    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    # Process noise (Section 5.2): tuned separately for GNSS-available
    # vs. blackout regimes - looser on position, tighter on bias terms
    # during blackout since we lean harder on the IMU-derived channels
    # then.
    q_pos_gnss: float = 0.05
    q_vel_gnss: float = 0.1
    q_pos_blackout: float = 0.5
    q_vel_blackout: float = 0.2
    q_psi: float = 1e-3
    q_ba: float = 1e-5
    q_bg: float = 1e-6

    # Measurement noise, nominal (Section 5.3).
    r_gnss_pos: float = 3.0  # meters, 1-sigma
    r_gnss_vel: float = 0.3  # m/s, 1-sigma
    r_channel_a: float = 0.5  # m/s, 1-sigma - Channel A target RMSE (Section 4.2)
    r_channel_b: float = 1.75  # m/s, 1-sigma - mid Section 4.3's 1.5-2.0 m/s target band

    # Trust weighting (Section 5.4).
    channel_a_b_disagreement_factor: float = 1.5
    channel_a_inflated_r_multiplier: float = 5.0

    # Road-signature anchor (Section 5.3 point 4 / Section 4.4).
    road_signature_confidence_threshold: float = 0.85

    # GNSS re-admission ramp (Section 5.5 step 5).
    gnss_reacquire_ramp_s: float = 2.5
    gnss_reacquire_r_multiplier_start: float = 50.0


@dataclass
class UkfState:
    pos: np.ndarray  # (2,) [north, east], meters
    vel: np.ndarray  # (2,) [v_north, v_east], m/s
    heading: float  # rad


class DualChannelUkf:
    """Wraps filterpy's UnscentedKalmanFilter with the CTCV process
    model and the four sequential update sources described in Section
    5.3. One instance per replay/run; call `step()` once per cycle.
    """

    def __init__(self, initial_state: UkfState, config: FusionConfig | None = None):
        self.config = config or FusionConfig()

        points = MerweScaledSigmaPoints(
            n=N_STATES,
            alpha=self.config.alpha,
            beta=self.config.beta,
            kappa=self.config.kappa,
        )
        self.ukf = UnscentedKalmanFilter(
            dim_x=N_STATES, dim_z=2, dt=0.1, fx=fx, hx=hx_position, points=points
        )

        x0 = np.zeros(N_STATES)
        x0[PN], x0[PE] = initial_state.pos
        x0[VN], x0[VE] = initial_state.vel
        x0[PSI] = initial_state.heading
        self.ukf.x = x0
        self.ukf.P = np.diag([1.0, 1.0, 0.5, 0.5, 0.05, 0.01, 1e-4])

        self._time_since_gnss_reacquired_s: float | None = None
        self._was_blacked_out = False

    # -- process noise selection -------------------------------------------------

    def _process_noise(self, dt: float, blackout: bool) -> np.ndarray:
        c = self.config
        q_pos = c.q_pos_blackout if blackout else c.q_pos_gnss
        q_vel = c.q_vel_blackout if blackout else c.q_vel_gnss
        diag = np.array([q_pos, q_pos, q_vel, q_vel, c.q_psi, c.q_ba, c.q_bg])
        return np.diag(diag) * dt

    # -- trust weighting (Section 5.4) -------------------------------------------

    def _channel_a_r(self, channel_a_speed: float, channel_b_speed: float) -> float:
        c = self.config
        disagreement = abs(channel_a_speed - channel_b_speed)
        if disagreement > c.channel_a_b_disagreement_factor * c.r_channel_b:
            return c.r_channel_a * c.channel_a_inflated_r_multiplier
        return c.r_channel_a

    # -- GNSS re-admission ramp (Section 5.5 step 5) -----------------------------

    def _gnss_r_multiplier(self, dt: float, gnss_available: bool) -> float:
        c = self.config
        if not gnss_available:
            self._was_blacked_out = True
            self._time_since_gnss_reacquired_s = None
            return 1.0  # unused when GNSS is absent this cycle

        if self._was_blacked_out:
            if self._time_since_gnss_reacquired_s is None:
                self._time_since_gnss_reacquired_s = 0.0
            else:
                self._time_since_gnss_reacquired_s += dt

            ramp_frac = min(
                1.0, self._time_since_gnss_reacquired_s / c.gnss_reacquire_ramp_s
            )
            if ramp_frac >= 1.0:
                self._was_blacked_out = False
                return 1.0
            # Linearly ramp the R multiplier down from "near-zero
            # trust" to nominal (1.0) as ramp_frac goes 0 -> 1.
            start = c.gnss_reacquire_r_multiplier_start
            return start + (1.0 - start) * ramp_frac

        return 1.0

    def _symmetrize_p(self) -> None:
        """Guard against covariance numerical drift from repeated
        sequential updates each cycle (Section 14's UKF numerical-
        stability note). Two things go wrong in practice with a
        7-state filter doing 3-5 sequential updates per cycle:
        asymmetry from floating-point noise, and one or more state
        variances (e.g. `ve` on a perfectly straight synthetic route)
        collapsing toward zero until Cholesky sees a tiny negative
        eigenvalue that is numerically indefinite even though it is
        conceptually zero. Symmetrizing fixes the first; a small
        diagonal jitter fixes the second without materially changing
        the filter's behavior anywhere P isn't already near-singular.
        """
        p = (self.ukf.P + self.ukf.P.T) / 2.0
        self.ukf.P = p + np.eye(N_STATES) * 1e-9

    def _refresh_sigmas(self) -> None:
        """filterpy's `UnscentedKalmanFilter.update()` reuses
        `self.sigmas_f` - the sigma points generated during the last
        `predict()` - for every subsequent `update()` call, since its
        normal use case is one predict then one update per cycle. This
        filter does one predict then up to four sequential updates per
        cycle (Section 5.3), so without this, the second/third/fourth
        update of a cycle would fold each source's correction into the
        *pre-update* sigma spread rather than the post-update one,
        which is mathematically inconsistent and reliably drives P
        non-positive-definite within a handful of cycles. filterpy's
        own `compute_process_sigmas` docstring names this exact
        multi-update-per-cycle case; regenerating sigma points from the
        current (just-updated) x/P with a zero-time identity fx is the
        documented fix. Call this between every pair of sequential
        updates in the same cycle (not needed right after `predict()`,
        which already leaves correct sigmas_f in place)."""
        self.ukf.compute_process_sigmas(dt=0.0, fx=lambda x, dt: x)

    # -- main step -----------------------------------------------------------------

    def step(
        self,
        dt: float,
        gyro_yaw: float,
        channel_a_speed: float,
        channel_b_speed: float,
        gnss_pos: np.ndarray | None,
        gnss_vel: np.ndarray | None,
        road_signature_pos: np.ndarray | None,
        road_signature_confidence: float,
    ) -> UkfState:
        """Advance the filter by one cycle. `gnss_pos`/`gnss_vel` are
        None when GNSS is unavailable this cycle (blackout). Road-
        signature args are None/0.0 when no segment match is offered
        this cycle."""
        c = self.config
        blackout = gnss_pos is None

        self.ukf.Q = self._process_noise(dt, blackout)
        self.ukf.predict(dt=dt, fx=fx, gyro_yaw=gyro_yaw)
        self._symmetrize_p()

        if gnss_pos is not None:
            r_mult = self._gnss_r_multiplier(dt, gnss_available=True)
            if gnss_vel is not None:
                z = np.concatenate([gnss_pos, gnss_vel])
                R = np.diag(
                    [c.r_gnss_pos**2, c.r_gnss_pos**2, c.r_gnss_vel**2, c.r_gnss_vel**2]
                ) * (r_mult**2)
                self.ukf.update(z, R=R, hx=hx_position_velocity)
            else:
                z = gnss_pos
                R = np.diag([c.r_gnss_pos**2, c.r_gnss_pos**2]) * (r_mult**2)
                self.ukf.update(z, R=R, hx=hx_position)
            self._symmetrize_p()
            self._refresh_sigmas()
        else:
            self._gnss_r_multiplier(dt, gnss_available=False)

        # Channel A / Channel B: rotate the scalar forward-speed
        # estimates into (vn, ve) using the filter's *current* heading
        # estimate (Section 5.3 point 2/3), then apply as velocity
        # pseudo-measurements.
        psi_hat = self.ukf.x[PSI]
        r_channel_a = self._channel_a_r(channel_a_speed, channel_b_speed)

        z_a = np.array(
            [channel_a_speed * np.cos(psi_hat), channel_a_speed * np.sin(psi_hat)]
        )
        self.ukf.update(z_a, R=np.eye(2) * r_channel_a**2, hx=hx_velocity)
        self._symmetrize_p()
        self._refresh_sigmas()

        z_b = np.array(
            [channel_b_speed * np.cos(psi_hat), channel_b_speed * np.sin(psi_hat)]
        )
        self.ukf.update(z_b, R=np.eye(2) * c.r_channel_b**2, hx=hx_velocity)
        self._symmetrize_p()

        # Road-signature drift-anchor (Section 5.3 point 4 / Section
        # 4.4 deployment rule): only applied above the confidence
        # threshold, as a soft position correction, never a hard reset.
        if (
            road_signature_pos is not None
            and road_signature_confidence >= c.road_signature_confidence_threshold
        ):
            self._refresh_sigmas()
            self.ukf.update(road_signature_pos, R=np.eye(2) * (c.r_gnss_pos * 2) ** 2, hx=hx_position)
            self._symmetrize_p()

        x = self.ukf.x
        return UkfState(pos=np.array([x[PN], x[PE]]), vel=np.array([x[VN], x[VE]]), heading=float(x[PSI]))
