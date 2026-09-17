package org.sih26.deadreckoning.fusion

/**
 * Kotlin port of fusion_core/python_prototype/zupt.py.
 *
 * When the vehicle is stopped its velocity is exactly zero, and that is a free,
 * near-exact measurement. It stops an integrated-speed channel from wandering while
 * the vehicle sits at a traffic light, which on a city demo route is a large fraction
 * of the blackout.
 *
 * Detection is gated on raw IMU, never on the filter's own speed estimate. Gating on
 * the estimate creates a feedback loop: the filter thinks it is slow, so ZUPT fires,
 * so the filter becomes more confident it is slow, so ZUPT keeps firing, and a
 * vehicle genuinely moving at 3 m/s gets pinned to a standstill by its own belief.
 * The accelerometer and gyro do not care what the filter believes, which is exactly
 * why they are the right input.
 *
 * The hard limit, stated up front: an accelerometer cannot distinguish rest from
 * constant velocity. That is Galilean invariance, not a tuning problem, and no
 * threshold fixes it. Real ZUPT gets away with it because a real stationary vehicle
 * has a different vibration signature from a moving one, engine idle versus
 * road-induced vibration, and that difference lives in the high-frequency content a
 * variance test picks up.
 *
 * Why this matters more on a phone than it did on the benchmark. The synthetic IMU is
 * ground-truth motion plus fixed-variance white noise, so parked and cruising have
 * identical variance: measured 0.0014 accel variance stopped against 0.0025 cruising,
 * with gyro variance actually LOWER while moving. The variance gate therefore fired
 * continuously on synthetic data and drove benchmark drift from 3.91% to 59.70%,
 * which is why it ships disabled. None of that is true on real hardware, where the
 * difference the gate was designed to catch is actually present.
 *
 * So: this is enabled by nobody until a real recording says so. Record the demo
 * vehicle parked with the engine running, measure the real variance floor, set the
 * thresholds from that recording, report the measured separation between parked and
 * moving, and only then ship it on. [statistics] exists to make that measurement a
 * five-minute job on a log you already have.
 */

data class ZuptConfig(
    /** Samples in the rolling window. 100 at 100 Hz is one second. The Python
     * reference uses 10 at its 10 Hz cycle rate; this is the same duration at the
     * phone's rate, not a different decision. */
    val windowN: Int = 100,
    val accelVarThreshold: Double = 0.02,
    val gyroVarThreshold: Double = 0.002,
    /** Mean horizontal specific force must also be below this. On synthetic data this
     * is the only condition doing real work, and note carefully that it does not
     * catch a vehicle cruising in a straight line at constant speed. */
    val accelMagnitudeThreshold: Double = 0.3,
    /** Tight, because a detected stop really is a near-exact measurement. Not zero,
     * because the detector can be wrong. */
    val rMps: Double = 0.05,
    val enabled: Boolean = false
)

/** What the gate actually saw. Log this during a recording; it is the raw material
 * for setting the thresholds from real data rather than from the synthetic defaults. */
data class ZuptStatistics(
    val accelVariance: Double,
    val gyroVariance: Double,
    val accelMean: Double,
    val windowFull: Boolean,
    val detected: Boolean
)

class ZuptDetector(val config: ZuptConfig = ZuptConfig()) {

    private val accelMagnitudes = DoubleArray(config.windowN)
    private val gyroValues = DoubleArray(config.windowN)
    private var writeIndex = 0
    private var filled = 0

    var lastStatistics: ZuptStatistics =
        ZuptStatistics(0.0, 0.0, 0.0, windowFull = false, detected = false)
        private set

    fun reset() {
        writeIndex = 0
        filled = 0
    }

    /**
     * Push one cycle of raw IMU and return whether the vehicle is currently detected
     * as stopped. `accelHorizontal` is the leveled, gravity-free horizontal specific
     * force; `gyroYaw` is the yaw rate about the gravity axis.
     *
     * The ring buffer is pre-allocated. Nothing here allocates on the 100 Hz hot path.
     */
    fun update(accelHorizontal: DoubleArray, gyroYaw: Double): Boolean {
        accelMagnitudes[writeIndex] = LinAlg.norm(accelHorizontal)
        gyroValues[writeIndex] = gyroYaw
        writeIndex = (writeIndex + 1) % config.windowN
        if (filled < config.windowN) filled++

        if (filled < config.windowN) {
            // Not enough history to judge. "Not detected as stopped" is the safe
            // answer: a spurious early ZUPT brakes a moving filter, whereas a missed
            // one merely forgoes a correction.
            lastStatistics = ZuptStatistics(0.0, 0.0, 0.0, windowFull = false, detected = false)
            return false
        }

        var accelMean = 0.0
        var gyroMean = 0.0
        for (i in 0 until config.windowN) {
            accelMean += accelMagnitudes[i]
            gyroMean += gyroValues[i]
        }
        accelMean /= config.windowN
        gyroMean /= config.windowN

        var accelVar = 0.0
        var gyroVar = 0.0
        for (i in 0 until config.windowN) {
            val da = accelMagnitudes[i] - accelMean
            val dg = gyroValues[i] - gyroMean
            accelVar += da * da
            gyroVar += dg * dg
        }
        accelVar /= config.windowN
        gyroVar /= config.windowN

        val detected = accelVar < config.accelVarThreshold &&
            gyroVar < config.gyroVarThreshold &&
            accelMean < config.accelMagnitudeThreshold

        lastStatistics = ZuptStatistics(accelVar, gyroVar, accelMean, windowFull = true, detected = detected)
        return detected
    }
}
