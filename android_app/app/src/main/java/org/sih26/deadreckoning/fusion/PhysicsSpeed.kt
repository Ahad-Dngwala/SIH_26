package org.sih26.deadreckoning.fusion

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.sqrt

/**
 * Kotlin port of fusion_core/python_prototype/physics_speed.py, Channel P.
 *
 * This is not Channel A. Channel A is a learned TCN regressor that was trained,
 * failed its pre-registered acceptance gate on bias, and was not shipped. Channel P
 * occupies the same slot in the UKF's update order with a classical, untrained
 * estimator, so the filter has some velocity evidence during a blackout instead of
 * none at all. Say that plainly rather than letting anyone infer a model is running.
 *
 * What this channel is, stated honestly: an open-loop single integration of a biased
 * accelerometer, reseeded from GNSS whenever GNSS is available. Its speed error grows
 * roughly linearly with time since the last reseed and the resulting position error
 * roughly quadratically. That is expected and is not a bug to be tuned away. It is
 * also why r_mps must be a measured number rather than an aspirational one.
 *
 * The three front-end pieces here exist specifically for the phone case, where the
 * accelerometer arrives in three axes pointing wherever the holder happens to point,
 * with gravity in them:
 *   1. MountLeveling resolves pitch and roll from gravity.
 *   2. ForwardAxisEstimator resolves the remaining yaw ambiguity by PCA over driving.
 *   3. PhysicsSpeedChannel integrates the longitudinal component.
 *
 * Steps 1 and 2 stand in for the untrained alignment_net and make no claim to be as
 * good as one.
 */

const val GRAVITY_MPS2 = 9.80665

/**
 * Gravity-aligned rotation from the phone's unknown mount frame into a level frame
 * whose z axis points up. Resolves pitch and roll only; which horizontal direction
 * the vehicle faces is not determined by gravity and is [ForwardAxisEstimator]'s job.
 *
 * `rotation` is stored row-major as body to level, matching the Python reference's
 * `np.stack([x_level, y_level, up], axis=0)`, so row 2 is the up vector. Yaw rate is
 * the gyro projected onto that row, never the phone's own z axis.
 */
class MountLeveling(val rotation: Array<DoubleArray>, val gravityMagnitude: Double) {

    val up: DoubleArray get() = rotation[2]

    /** Rotate a raw accelerometer sample into the level frame and return its
     * horizontal (x, y) components. Gravity lands entirely in level-frame z, so it is
     * dropped rather than subtracted from x and y. */
    fun horizontal(accel: DoubleArray): DoubleArray {
        val x = rotation[0][0] * accel[0] + rotation[0][1] * accel[1] + rotation[0][2] * accel[2]
        val y = rotation[1][0] * accel[0] + rotation[1][1] * accel[1] + rotation[1][2] * accel[2]
        return doubleArrayOf(x, y)
    }

    /** Yaw rate about the gravity-aligned vertical axis.
     *
     * Not gyro.z. The mount orientation is arbitrary, so the phone's own z axis is
     * not the vehicle's vertical, and using it is the single easiest way to make a
     * phone-mounted filter turn the wrong way. Validate the sign on a known left turn
     * before trusting any recording: if the phone is face down it flips. */
    fun yawRate(gyro: DoubleArray): Double = LinAlg.dot(gyro, up)

    companion object {
        /**
         * Estimate leveling from a window of accelerometer samples taken while the
         * vehicle is stationary. At rest the accelerometer measures specific force,
         * the reaction to gravity, so its mean points UP in the world frame, not down.
         */
        fun fromStationaryWindow(window: List<DoubleArray>): MountLeveling {
            require(window.size >= 2) { "need at least 2 samples to estimate a gravity vector" }

            val mean = DoubleArray(3)
            for (sample in window) for (i in 0 until 3) mean[i] += sample[i]
            for (i in 0 until 3) mean[i] /= window.size

            val magnitude = LinAlg.norm(mean)
            if (magnitude < 1.0) {
                throw IllegalArgumentException(
                    "stationary-window mean |accel| is $magnitude m/s^2, which is not a " +
                        "plausible gravity magnitude. Either the window was not stationary, " +
                        "or TYPE_LINEAR_ACCELERATION was used instead of TYPE_ACCELEROMETER " +
                        "and gravity has already been removed by a vendor filter."
                )
            }

            val up = DoubleArray(3) { mean[it] / magnitude }

            // Gram-Schmidt an arbitrary body axis against up to get a level-frame x.
            // Which horizontal direction this lands on is arbitrary and is exactly
            // what the forward-axis estimator resolves.
            var seed = doubleArrayOf(1.0, 0.0, 0.0)
            if (abs(LinAlg.dot(seed, up)) > 0.9) seed = doubleArrayOf(0.0, 1.0, 0.0)

            val projection = LinAlg.dot(seed, up)
            val xLevel = DoubleArray(3) { seed[it] - projection * up[it] }
            val xNorm = LinAlg.norm(xLevel)
            for (i in 0 until 3) xLevel[i] /= xNorm

            val yLevel = doubleArrayOf(
                up[1] * xLevel[2] - up[2] * xLevel[1],
                up[2] * xLevel[0] - up[0] * xLevel[2],
                up[0] * xLevel[1] - up[1] * xLevel[0]
            )

            return MountLeveling(arrayOf(xLevel, yLevel, up), magnitude)
        }
    }
}

/**
 * Resolves which horizontal direction is vehicle-forward.
 *
 * Accelerate and brake events dominate horizontal specific force variance and lie
 * along the longitudinal axis, so the first principal component of horizontal accel
 * over a driving window is the longitudinal direction. PCA gives an axis, not a
 * direction, so the sign is fixed by correlating projected acceleration against GNSS
 * speed change over the same window. An unresolved sign makes Channel P integrate
 * backwards, which is worse than not having the channel at all.
 *
 * This accumulates online, unlike the offline Python version which sees the whole
 * session at once. Same estimator, fed incrementally: only samples taken while
 * genuinely moving are admitted, since a stationary stretch carries no longitudinal
 * signal and would only dilute the PCA.
 */
class ForwardAxisEstimator(
    private val minSpeedMps: Double = 2.0,
    private val minSamples: Int = 300,
    /** Stop admitting after this many samples. At 100 Hz this is 60 s of driving,
     * which is far more than the axis needs, and it bounds both memory and the cost
     * of a recompute on a hot path. The offline Python version uses every moving
     * sample in the session; the axis is a fixed property of the mount, so the two
     * agree as soon as there is enough accelerate and brake content to see it. */
    private val maxSamples: Int = 6000
) {
    private val horizontalX = ArrayList<Double>()
    private val horizontalY = ArrayList<Double>()
    private val speedDelta = ArrayList<Double>()

    var axis: DoubleArray? = null
        private set
    var signResolved: Boolean = false
        private set

    val sampleCount: Int get() = horizontalX.size

    /** Admit one sample. `speedChange` is the most recent change in GNSS
     * speed-over-ground, used only to fix the sign. */
    fun add(horizontal: DoubleArray, speedMps: Double, speedChange: Double) {
        if (speedMps <= minSpeedMps) return
        if (horizontalX.size >= maxSamples) return
        horizontalX.add(horizontal[0])
        horizontalY.add(horizontal[1])
        speedDelta.add(speedChange)
    }

    fun isSaturated(): Boolean = horizontalX.size >= maxSamples

    fun hasEnough(): Boolean = horizontalX.size >= minSamples

    /**
     * Compute the axis from everything admitted so far. Returns null if there is not
     * enough driving yet, which the caller must treat as "Channel P is not ready",
     * never as "assume forward is x".
     */
    fun estimate(): DoubleArray? {
        val n = horizontalX.size
        if (n < 3) return null

        var meanX = 0.0
        var meanY = 0.0
        for (i in 0 until n) { meanX += horizontalX[i]; meanY += horizontalY[i] }
        meanX /= n
        meanY /= n

        // Covariance of the centered 2D data. Its principal eigenvector is the first
        // right singular vector the Python reference takes from an SVD; for a 2x2
        // symmetric matrix the closed form below is exact and avoids pulling in a
        // decomposition for two numbers.
        var sxx = 0.0
        var sxy = 0.0
        var syy = 0.0
        for (i in 0 until n) {
            val dx = horizontalX[i] - meanX
            val dy = horizontalY[i] - meanY
            sxx += dx * dx
            sxy += dx * dy
            syy += dy * dy
        }

        val trace = sxx + syy
        val determinant = sxx * syy - sxy * sxy
        val discriminant = max(0.0, trace * trace / 4.0 - determinant)
        val largest = trace / 2.0 + sqrt(discriminant)

        var ax: Double
        var ay: Double
        if (abs(sxy) > 1e-12) {
            ax = largest - syy
            ay = sxy
        } else {
            // Already diagonal: the axis is whichever of the two has more variance.
            if (sxx >= syy) { ax = 1.0; ay = 0.0 } else { ax = 0.0; ay = 1.0 }
        }
        val norm = sqrt(ax * ax + ay * ay)
        ax /= norm
        ay /= norm

        // Sign: positive projected acceleration must correspond to increasing speed.
        var correlation = 0.0
        for (i in 0 until n) {
            val projected = (horizontalX[i] - meanX) * ax + (horizontalY[i] - meanY) * ay
            correlation += projected * speedDelta[i]
        }
        if (correlation < 0.0) { ax = -ax; ay = -ay }

        val resolved = doubleArrayOf(ax, ay)
        axis = resolved
        signResolved = abs(correlation) > 0.0
        return resolved
    }

    fun reset() {
        horizontalX.clear()
        horizontalY.clear()
        speedDelta.clear()
        axis = null
        signResolved = false
    }
}

data class PhysicsSpeedConfig(
    /** The 1-sigma this channel is handed to the UKF as. Override with a measured
     * number from tools/measure_physics_channel_r.py. Note that this channel's error
     * is a slow drift rather than white noise, so the independence assumption behind
     * R is violated; a measured RMSE is a pragmatic proxy, and a defensible one, but
     * it is a proxy. */
    val rMps: Double = 2.0,
    val maxSpeedMps: Double = 70.0,
    /** Optional first-order leak of integrated speed toward the last GNSS-confirmed
     * speed, as a time constant in seconds. Zero disables it. It bounds how far an
     * integrated bias can run away during a long blackout, at the cost of being wrong
     * during genuine sustained acceleration. Off by default; turn it on only with a
     * measured justification. */
    val leakTauS: Double = 0.0
)

/**
 * Integrates longitudinal acceleration into a scalar forward speed, reseeded from
 * GNSS whenever GNSS is available.
 *
 * Returns null until the first reseed, because before that the channel has no
 * absolute reference and must not feed the filter a fabricated one. The caller skips
 * the update on null rather than substituting zero.
 */
class PhysicsSpeedChannel(val config: PhysicsSpeedConfig = PhysicsSpeedConfig()) {

    var speed: Double? = null
        private set
    private var lastGnssSpeed: Double? = null

    fun reset() {
        speed = null
        lastGnssSpeed = null
    }

    /** Snap the integrated speed to a GNSS measurement. This is what keeps the
     * integration from being open-loop over the whole route rather than just over
     * each blackout. */
    fun reseed(gnssSpeed: Double): Double {
        val seeded = max(0.0, gnssSpeed)
        speed = seeded
        lastGnssSpeed = seeded
        return seeded
    }

    /** Force the integrated speed to zero. Called when an external detector says the
     * vehicle is stopped. Kept separate from [update] so the detector's decision and
     * this channel's integration stay independently testable. */
    fun applyZupt(): Double {
        speed = 0.0
        return 0.0
    }

    fun update(dt: Double, longitudinalAccel: Double, gnssSpeed: Double? = null): Double? {
        if (gnssSpeed != null) return reseed(gnssSpeed)

        val current = speed ?: return null

        var next = current + longitudinalAccel * dt

        val tau = config.leakTauS
        val anchor = lastGnssSpeed
        if (tau > 0.0 && anchor != null) {
            val alpha = min(1.0, dt / tau)
            next += alpha * (anchor - next)
        }

        val clamped = min(max(next, 0.0), config.maxSpeedMps)
        speed = clamped
        return clamped
    }
}
