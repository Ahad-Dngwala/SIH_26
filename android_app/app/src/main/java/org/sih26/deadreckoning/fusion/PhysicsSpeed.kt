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
 * Resolves which horizontal direction is vehicle-forward, online.
 *
 * The estimator is the one described in the Python reference's
 * `estimate_forward_axis`: the axis is the direction whose acceleration explains
 * the GNSS speed change, computed as the cross-correlation vector between centred
 * horizontal specific force and speed delta. That resolves the axis and its sign
 * together. An unresolved or flipped sign makes Channel P integrate backwards, which
 * is worse than not having the channel at all.
 *
 * Do not "simplify" this back to a first principal component. PCA was what this used
 * to be, and it fails on exactly the windows a phone runs in. In a turn, lateral
 * specific force is speed times yaw rate, several times larger than the longitudinal
 * content of ordinary speed variation, so a turn-heavy window has its variance
 * dominated by the lateral axis and PCA returns an axis ninety degrees off. Measured
 * against a known mount rotation, PCA's worst case across window sizes was a dot
 * product of 0.364 with the true axis where this estimator's was 0.982. The offline
 * caller never saw it because a whole session dilutes the turns; an online caller
 * passes straight through that regime in the first minute of every drive.
 *
 * Accumulates running moments rather than storing samples, so memory and per-sample
 * cost are constant no matter how long the drive, and the estimate uses every moving
 * sample seen so far rather than an arbitrary recent window.
 */
class ForwardAxisEstimator(
    private val minSpeedMps: Double = 2.0,
    private val minSamples: Int = 300
) {
    private companion object {
        /** Longest gap between fixes whose speed change can still be attributed to
         * the samples inside it. Comfortably above a 1 Hz cadence with a few dropped
         * fixes, well below any deliberate blackout. */
        const val MAX_INTERVAL_S = 3.0
    }

    // Global running moments over every admitted sample.
    private var n: Long = 0
    private var sumX = 0.0
    private var sumY = 0.0
    private var sumXX = 0.0
    private var sumXY = 0.0
    private var sumYY = 0.0
    private var sumD = 0.0
    private var sumXD = 0.0
    private var sumYD = 0.0

    // Accumulators for the samples since the last GNSS fix. GNSS speed change is
    // only known once the fix that ends an interval arrives, so samples are held
    // here and folded in with the delta that actually covers them. Applying each
    // delta to the samples that follow it instead costs a full second of lag
    // between an acceleration and the speed change it caused, which decorrelates
    // the very signal this estimator runs on. Measured: with the lag, the axis
    // landed 7.4 degrees off the known mount rotation and leaked enough cornering
    // force into Channel P to push blackout drift to 38%. Without it, under a
    // degree.
    private var intervalN: Long = 0
    private var intervalSumX = 0.0
    private var intervalSumY = 0.0
    private var intervalSumXX = 0.0
    private var intervalSumXY = 0.0
    private var intervalSumYY = 0.0

    var axis: DoubleArray? = null
        private set
    var signResolved: Boolean = false
        private set

    val sampleCount: Long get() = n

    /** Buffer one sample. Nothing is committed until the interval closes. */
    fun addSample(horizontal: DoubleArray, speedMps: Double) {
        if (speedMps <= minSpeedMps) return
        val x = horizontal[0]
        val y = horizontal[1]
        intervalN++
        intervalSumX += x
        intervalSumY += y
        intervalSumXX += x * x
        intervalSumXY += x * y
        intervalSumYY += y * y
    }

    /**
     * Close the current interval with the GNSS speed change that occurred across it,
     * folding its buffered samples into the running moments.
     *
     * Since the delta is constant across the interval, the per-sample sum of
     * x_i * d reduces to d times the interval's sum of x, so no sample history is
     * needed to get the correlation exactly right.
     *
     * `intervalSeconds` guards against attributing a long gap's total speed change to
     * every sample inside it. A blackout produces exactly that gap: no fix reaches
     * the pipeline for the whole window, and folding thirty seconds of buffered
     * samples in with a thirty-second speed delta swamps every honest interval before
     * it. Measured: that single mistake moved the axis from under a degree off the
     * known mount rotation to thirteen degrees off, and took blackout drift from
     * 1.6% to 30%. Over-long intervals are discarded, not clamped: there is no
     * correct weight for them.
     */
    fun closeInterval(speedChange: Double, intervalSeconds: Double = 1.0) {
        if (intervalN == 0L) return
        if (intervalSeconds > MAX_INTERVAL_S) {
            discardInterval()
            return
        }
        n += intervalN
        sumX += intervalSumX
        sumY += intervalSumY
        sumXX += intervalSumXX
        sumXY += intervalSumXY
        sumYY += intervalSumYY
        sumD += speedChange * intervalN
        sumXD += intervalSumX * speedChange
        sumYD += intervalSumY * speedChange

        discardInterval()
    }

    /** Throw away the buffered samples without committing them. Called when the
     * speed reference that would have explained them is unavailable, which is what a
     * GNSS blackout means. */
    fun discardInterval() {
        intervalN = 0
        intervalSumX = 0.0
        intervalSumY = 0.0
        intervalSumXX = 0.0
        intervalSumXY = 0.0
        intervalSumYY = 0.0
    }

    fun hasEnough(): Boolean = n >= minSamples

    /**
     * Compute the axis from everything committed so far, or null if there is not
     * enough driving yet. The caller must treat null as "Channel P is not ready",
     * never as "assume forward is x".
     */
    fun estimate(): DoubleArray? {
        if (n < 3) return null

        val count = n.toDouble()
        val meanX = sumX / count
        val meanY = sumY / count

        // Cross-correlation of centred horizontal accel with speed change. Expanding
        // sum((x - mx) * d) gives sumXD - mx * sumD, so no sample history is needed.
        var ax = sumXD - meanX * sumD
        var ay = sumYD - meanY * sumD
        var magnitude = sqrt(ax * ax + ay * ay)

        if (magnitude > 1e-9) {
            ax /= magnitude
            ay /= magnitude
            val resolved = doubleArrayOf(ax, ay)
            axis = resolved
            signResolved = true
            return resolved
        }

        // Degenerate: speed was effectively constant across everything seen, so
        // there is nothing to correlate against. Fall back to the principal axis of
        // the centred data, with the sign left unresolved, matching the Python
        // reference's no-reference path.
        val sxx = sumXX - count * meanX * meanX
        val sxy = sumXY - count * meanX * meanY
        val syy = sumYY - count * meanY * meanY

        val trace = sxx + syy
        val determinant = sxx * syy - sxy * sxy
        val discriminant = max(0.0, trace * trace / 4.0 - determinant)
        val largest = trace / 2.0 + sqrt(discriminant)

        if (abs(sxy) > 1e-12) {
            ax = largest - syy
            ay = sxy
        } else {
            if (sxx >= syy) { ax = 1.0; ay = 0.0 } else { ax = 0.0; ay = 1.0 }
        }
        magnitude = sqrt(ax * ax + ay * ay)
        if (magnitude < 1e-12) return null

        val resolved = doubleArrayOf(ax / magnitude, ay / magnitude)
        axis = resolved
        signResolved = false
        return resolved
    }

    fun reset() {
        n = 0
        sumX = 0.0; sumY = 0.0
        sumXX = 0.0; sumXY = 0.0; sumYY = 0.0
        sumD = 0.0; sumXD = 0.0; sumYD = 0.0
        intervalN = 0
        intervalSumX = 0.0; intervalSumY = 0.0
        intervalSumXX = 0.0; intervalSumXY = 0.0; intervalSumYY = 0.0
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
