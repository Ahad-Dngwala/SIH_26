package org.sih26.deadreckoning.fusion

import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max

/**
 * The online pipeline: raw phone sensors in, fused state out.
 *
 * Deliberately free of Android imports so the whole thing runs on a desktop JVM
 * against a recorded log. Every correctness question about this file can then be
 * answered without a device, which is the only reason the answers exist before the
 * drive rather than after it.
 *
 * Data flow, one direction, no cleverness:
 *
 *     onImu (100 Hz)  -> leveling -> horizontal accel + yaw rate
 *                     -> ZUPT detector
 *                     -> Channel P integration
 *                     -> UKF step
 *     onGnss (1 Hz)   -> staged for the next IMU cycle, or withheld if blacked out
 *
 * The blackout is a boolean gate on whether a fix reaches the filter. Fixes are
 * always received and always logged. That is the entire experiment: withheld fixes
 * are the ground truth the drift is measured against, and airplane mode would destroy
 * them and cost 10 to 30 seconds of reacquisition afterwards.
 */

/** A snapshot for the UI and for the optional fused stream in the session log. */
data class FusionSnapshot(
    val tSeconds: Double,
    val fusedNorth: Double,
    val fusedEast: Double,
    val coastNorth: Double,
    val coastEast: Double,
    val speedMps: Double,
    val headingDeg: Double,
    val blackout: Boolean,
    val blackoutElapsedS: Double,
    val blackoutDistanceM: Double,
    val driftMeters: Double,
    val driftPercent: Double,
    val channelPSpeed: Double?,
    val zuptActive: Boolean,
    val imuRateHz: Double,
    val stepLatencyMs: Double,
    val phase: PipelinePhase,
    val lastTruthNorth: Double,
    val lastTruthEast: Double
)

enum class PipelinePhase {
    /** Collecting a stationary window so mount leveling has a gravity vector. Start
     * every session parked for a few seconds or this phase never completes cleanly. */
    LEVELING,

    /** Levelled, waiting for the first GNSS fix to set the local frame origin. */
    WAITING_FOR_GNSS,

    /** Filter initialised and running. */
    RUNNING
}

data class PipelineConfig(
    val fusion: FusionConfig = FusionConfig(),
    val physics: PhysicsSpeedConfig = PhysicsSpeedConfig(),
    val zupt: ZuptConfig = ZuptConfig(),
    /** Seconds of stationary samples required before leveling. */
    val levelingWindowS: Double = 3.0,
    /** Speed above which a GNSS bearing is trusted for initial heading. At a
     * standstill the bearing is noise. */
    val minSpeedForBearingMps: Double = 3.0,
    /** Channel P is fed to the filter as Channel A's slot with this override, since
     * its measured R is nothing like Channel A's Section 4.2 target. */
    val useChannelP: Boolean = true,
    /** Largest dt a single cycle may claim. A delivery hiccup or a resumed app must
     * not propagate the filter through a ten-second step as though it were one
     * sample. */
    val maxStepDtS: Double = 0.2
)

private const val NANOS_PER_SECOND = 1_000_000_000.0

/** Recompute the forward axis once a second rather than every cycle. The axis is a
 * property of the mount, so it does not move between recomputes, and the estimate is
 * linear in the accumulated sample count. */
private const val AXIS_RECOMPUTE_CYCLES = 100

class FusionPipeline(val config: PipelineConfig = PipelineConfig()) {

    var phase: PipelinePhase = PipelinePhase.LEVELING
        private set

    var leveling: MountLeveling? = null
        private set
    var localFrame: LocalFrame? = null
        private set

    private val forwardAxis = ForwardAxisEstimator()
    private val channelP = PhysicsSpeedChannel(config.physics)
    private val zuptDetector = ZuptDetector(config.zupt)

    private var ukf: DualChannelUkf? = null

    /** A second filter that receives no GNSS and no velocity channel during a
     * blackout: the honest "what the phone does today" CTCV coast. Having it on
     * screen next to the fused track is the A/B the demo is actually about. */
    private var coastUkf: DualChannelUkf? = null

    private val levelingWindow = ArrayList<DoubleArray>()
    private var levelingStartNs: Long? = null

    private var lastImuNs: Long? = null
    private var sessionStartNs: Long? = null

    private var pendingFix: GnssFix? = null
    private var lastFix: GnssFix? = null
    private var lastAdmittedSpeed: Double? = null
    private var lastSpeedDelta: Double = 0.0
    private var cyclesSinceAxisEstimate: Int = 0

    var blackout: Boolean = false
        private set
    private var blackoutStartNs: Long? = null
    private var blackoutDistanceM: Double = 0.0
    private var blackoutTruthNorth: Double = 0.0
    private var blackoutTruthEast: Double = 0.0

    private var imuCount: Long = 0
    private var imuRateHz: Double = 0.0
    private var rateWindowStartNs: Long? = null
    private var rateWindowCount: Long = 0
    private var lastLatencyMs: Double = 0.0

    var lastSnapshot: FusionSnapshot? = null
        private set

    data class GnssFix(
        val tNs: Long,
        val latDeg: Double,
        val lonDeg: Double,
        val speedMps: Double,
        val bearingDeg: Double,
        val accuracyM: Double
    )

    /** Toggle whether GNSS reaches the filter. Fixes keep arriving and keep being
     * logged either way. */
    fun setBlackout(active: Boolean, tNs: Long) {
        if (active == blackout) return
        blackout = active
        if (active) {
            blackoutStartNs = tNs
            blackoutDistanceM = 0.0
            val fix = lastFix
            val frame = localFrame
            if (fix != null && frame != null) {
                val ne = frame.toNorthEast(fix.latDeg, fix.lonDeg)
                blackoutTruthNorth = ne[0]
                blackoutTruthEast = ne[1]
            }
            // Fork the coast baseline from the fused state at the moment the lights
            // go out, so the two tracks are comparable from a common starting point.
            val current = ukf?.state()
            if (current != null) coastUkf = DualChannelUkf(current, config.fusion)
        } else {
            blackoutStartNs = null
        }
    }

    /**
     * Raw GNSS fix from LocationManager.GPS_PROVIDER. Never the fused provider: it
     * already blends IMU, WiFi and cell into its position, so measuring our dead
     * reckoning against it would be measuring Google's dead reckoning against ours.
     *
     * The fix is staged rather than applied immediately, so all filter work happens
     * on the IMU thread at a single well-defined cadence.
     */
    fun onGnss(fix: GnssFix) {
        lastFix = fix
        if (localFrame == null) localFrame = LocalFrame(fix.latDeg, fix.lonDeg)

        if (blackout) {
            // Ground truth accounting continues through the blackout, using the
            // speed integral rather than summed fix-to-fix distance: independent
            // per-fix position noise inflates a summed path length, and a larger
            // denominator would make our own drift percentage smaller.
            val previous = lastAdmittedTruthNs
            if (previous != null) {
                val dt = (fix.tNs - previous) / NANOS_PER_SECOND
                if (dt in 0.0..5.0) blackoutDistanceM += fix.speedMps * dt
            }
            lastAdmittedTruthNs = fix.tNs
            return
        }

        lastAdmittedTruthNs = fix.tNs
        pendingFix = fix
    }

    private var lastAdmittedTruthNs: Long? = null

    /**
     * Raw IMU sample. `accel` is TYPE_ACCELEROMETER in m/s^2 with gravity included,
     * `gyro` is TYPE_GYROSCOPE in rad/s, both in the phone frame, both raw.
     *
     * `tNs` is SensorEvent.timestamp: nanoseconds on the monotonic elapsedRealtime
     * base, never wall clock. dt is computed per sample from consecutive timestamps,
     * because Android will not hand you a clean 10 ms cadence and pretending it does
     * injects error directly into the integration.
     */
    fun onImu(tNs: Long, accel: DoubleArray, gyro: DoubleArray): FusionSnapshot? {
        val startedNs = System.nanoTime()
        if (sessionStartNs == null) sessionStartNs = tNs
        imuCount++
        updateRate(tNs)

        when (phase) {
            PipelinePhase.LEVELING -> {
                if (levelingStartNs == null) levelingStartNs = tNs
                levelingWindow.add(accel.copyOf())
                val elapsed = (tNs - levelingStartNs!!) / NANOS_PER_SECOND
                if (elapsed >= config.levelingWindowS && levelingWindow.size >= 2) {
                    leveling = MountLeveling.fromStationaryWindow(levelingWindow)
                    levelingWindow.clear()
                    phase = PipelinePhase.WAITING_FOR_GNSS
                }
                lastImuNs = tNs
                return null
            }

            PipelinePhase.WAITING_FOR_GNSS -> {
                val fix = pendingFix ?: lastFix
                val frame = localFrame
                if (fix == null || frame == null) {
                    lastImuNs = tNs
                    return null
                }
                val ne = frame.toNorthEast(fix.latDeg, fix.lonDeg)
                // Heading from GNSS bearing only once actually moving. At a
                // standstill the bearing is noise, and seeding the filter with noise
                // costs the whole session.
                val heading = if (fix.speedMps >= config.minSpeedForBearingMps) {
                    bearingDegToPsi(fix.bearingDeg)
                } else {
                    0.0
                }
                val initial = UkfState(
                    posN = ne[0],
                    posE = ne[1],
                    velN = fix.speedMps * kotlin.math.cos(heading),
                    velE = fix.speedMps * kotlin.math.sin(heading),
                    heading = heading
                )
                ukf = DualChannelUkf(initial, config.fusion)
                channelP.reseed(fix.speedMps)
                pendingFix = null
                phase = PipelinePhase.RUNNING
                lastImuNs = tNs
                return null
            }

            PipelinePhase.RUNNING -> Unit
        }

        val previousNs = lastImuNs
        lastImuNs = tNs
        if (previousNs == null) return lastSnapshot

        var dt = (tNs - previousNs) / NANOS_PER_SECOND
        if (dt <= 0.0) return lastSnapshot
        if (dt > config.maxStepDtS) dt = config.maxStepDtS

        val level = leveling ?: return lastSnapshot
        val filter = ukf ?: return lastSnapshot

        val horizontal = level.horizontal(accel)
        val yawRate = level.yawRate(gyro)

        val fix = pendingFix
        pendingFix = null

        // Forward axis: accumulate at IMU rate while genuinely driving, with the sign
        // resolved against the most recent GNSS speed change. Accumulating only on
        // GNSS fixes would need 300 seconds of driving to reach the sample count the
        // offline version reaches in 3, which would leave Channel P silent through
        // most of a demo.
        val currentSpeed = fix?.speedMps ?: lastFix?.speedMps ?: 0.0
        if (fix != null) {
            val previousSpeed = lastAdmittedSpeed
            lastSpeedDelta = if (previousSpeed == null) 0.0 else fix.speedMps - previousSpeed
            lastAdmittedSpeed = fix.speedMps
        }
        forwardAxis.add(horizontal, currentSpeed, lastSpeedDelta)
        cyclesSinceAxisEstimate++
        if (forwardAxis.hasEnough() && cyclesSinceAxisEstimate >= AXIS_RECOMPUTE_CYCLES) {
            forwardAxis.estimate()
            cyclesSinceAxisEstimate = 0
        }

        val zuptActive = config.zupt.enabled && zuptDetector.update(horizontal, yawRate)
        if (!config.zupt.enabled) zuptDetector.update(horizontal, yawRate)

        // Channel P. Until the forward axis is resolved the channel has nothing
        // trustworthy to integrate, so it reseeds on GNSS and otherwise says nothing
        // rather than integrating along an arbitrary direction.
        val axis = forwardAxis.axis
        var channelSpeed: Double? = null
        if (config.useChannelP) {
            if (zuptActive) {
                channelP.applyZupt()
            }
            channelSpeed = if (fix != null) {
                channelP.update(dt, 0.0, fix.speedMps)
            } else if (axis != null) {
                channelP.update(dt, horizontal[0] * axis[0] + horizontal[1] * axis[1])
            } else {
                null
            }
        }

        val gnssPos: DoubleArray?
        val gnssVel: DoubleArray?
        val frame = localFrame
        if (fix != null && frame != null) {
            gnssPos = frame.toNorthEast(fix.latDeg, fix.lonDeg)
            val psi = bearingDegToPsi(fix.bearingDeg)
            gnssVel = doubleArrayOf(
                fix.speedMps * kotlin.math.cos(psi),
                fix.speedMps * kotlin.math.sin(psi)
            )
        } else {
            gnssPos = null
            gnssVel = null
        }

        val state = filter.step(
            dt = dt,
            gyroYaw = yawRate,
            channelASpeed = channelSpeed,
            channelBSpeed = null,
            gnssPos = gnssPos,
            gnssVel = gnssVel,
            rChannelAOverride = if (channelSpeed != null) config.physics.rMps else null,
            zupt = zuptActive,
            rZupt = config.zupt.rMps
        )

        // Coast baseline: same gyro, no GNSS, no velocity channel.
        val coastState = coastUkf?.step(dt = dt, gyroYaw = yawRate) ?: state

        lastLatencyMs = (System.nanoTime() - startedNs) / 1_000_000.0

        val snapshot = buildSnapshot(tNs, state, coastState, channelSpeed, zuptActive)
        lastSnapshot = snapshot
        return snapshot
    }

    private fun buildSnapshot(
        tNs: Long,
        state: UkfState,
        coast: UkfState,
        channelSpeed: Double?,
        zuptActive: Boolean
    ): FusionSnapshot {
        val startNs = sessionStartNs ?: tNs
        val tSeconds = (tNs - startNs) / NANOS_PER_SECOND

        val frame = localFrame
        val fix = lastFix
        var truthNorth = 0.0
        var truthEast = 0.0
        if (frame != null && fix != null) {
            val ne = frame.toNorthEast(fix.latDeg, fix.lonDeg)
            truthNorth = ne[0]
            truthEast = ne[1]
        }

        // Live drift, against the withheld fixes. This is a display number. The
        // number that goes on a slide is computed offline by
        // tools/phone_replay/run.py from the raw log, so a bug in this arithmetic
        // cannot flatter the reported result.
        val driftMeters = if (blackout) {
            hypot(state.posN - truthNorth, state.posE - truthEast)
        } else {
            0.0
        }
        val driftPercent = if (blackout && blackoutDistanceM > 1.0) {
            driftMeters / blackoutDistanceM * 100.0
        } else {
            0.0
        }

        val blackoutElapsed = blackoutStartNs?.let { (tNs - it) / NANOS_PER_SECOND } ?: 0.0

        return FusionSnapshot(
            tSeconds = tSeconds,
            fusedNorth = state.posN,
            fusedEast = state.posE,
            coastNorth = coast.posN,
            coastEast = coast.posE,
            speedMps = state.speed,
            headingDeg = psiToBearingDeg(state.heading),
            blackout = blackout,
            blackoutElapsedS = blackoutElapsed,
            blackoutDistanceM = blackoutDistanceM,
            driftMeters = driftMeters,
            driftPercent = driftPercent,
            channelPSpeed = channelSpeed,
            zuptActive = zuptActive,
            imuRateHz = imuRateHz,
            stepLatencyMs = lastLatencyMs,
            phase = phase,
            lastTruthNorth = truthNorth,
            lastTruthEast = truthEast
        )
    }

    private fun updateRate(tNs: Long) {
        val windowStart = rateWindowStartNs
        if (windowStart == null) {
            rateWindowStartNs = tNs
            rateWindowCount = 0
            return
        }
        rateWindowCount++
        val elapsed = (tNs - windowStart) / NANOS_PER_SECOND
        if (elapsed >= 1.0) {
            imuRateHz = rateWindowCount / elapsed
            rateWindowStartNs = tNs
            rateWindowCount = 0
        }
    }

    /** Diagnostics for the UI and for a recording's own header. */
    fun describeFrontEnd(): String {
        val level = leveling
        val axis = forwardAxis.axis
        val gravity = level?.gravityMagnitude ?: 0.0
        val axisText = if (axis == null) {
            "not resolved (${forwardAxis.sampleCount} driving samples)"
        } else {
            "[${fmt(axis[0])}, ${fmt(axis[1])}]" +
                if (forwardAxis.signResolved) "" else " SIGN NOT RESOLVED"
        }
        return "gravity ${fmt(gravity)} m/s^2, forward axis $axisText, imu ${fmt(imuRateHz)} Hz"
    }

    private fun fmt(value: Double): String {
        val scaled = kotlin.math.round(value * 1000.0) / 1000.0
        return if (abs(scaled) < 1e-9) "0.0" else scaled.toString()
    }

    /** Total IMU samples seen, for the log header and the sanity check that the
     * foreground service was never throttled. */
    val imuSamples: Long get() = imuCount

    val zuptStatistics: ZuptStatistics get() = zuptDetector.lastStatistics

    val forwardAxisResolved: Boolean get() = forwardAxis.axis != null

    /** The resolved forward axis in the level frame, or null while it is still being
     * estimated. Exposed so the offline harness can compare it against the Python
     * reference's axis component by component rather than trusting a boolean. */
    val forwardAxisValue: DoubleArray? get() = forwardAxis.axis

    val channelPSpeedMps: Double? get() = channelP.speed

    val blackoutDistanceMeters: Double get() = max(0.0, blackoutDistanceM)
}
