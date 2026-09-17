package org.sih26.deadreckoning.harness

import org.sih26.deadreckoning.fusion.FusionPipeline
import org.sih26.deadreckoning.fusion.FusionSnapshot
import org.sih26.deadreckoning.fusion.PipelineConfig
import org.sih26.deadreckoning.fusion.PhysicsSpeedConfig
import java.io.File
import kotlin.math.hypot
import kotlin.system.exitProcess

/**
 * Replay a recorded session through the exact pipeline the phone runs.
 *
 * This is the second correctness gate. tools/parity/ pins the filter math; this pins
 * the front end, which is everything the filter math cannot check: mount leveling,
 * forward-axis resolution, yaw about gravity rather than about the phone's own z,
 * Channel P reseeding, the local frame convention, and the blackout gate.
 *
 * The front end was untested theory until this existed, and it is where the
 * phone-specific bugs live. A sign error in the forward axis makes Channel P
 * integrate backwards; using gyro.z instead of the gravity projection makes the
 * filter turn the wrong way. Neither crashes. Both are caught here, against a
 * session with an arbitrary mount rotation, in seconds, without a car.
 *
 * Input is the CSV from tools/phone_replay/export_session_csv.py. Output is
 * key=value lines so tools/phone_replay/check_kotlin_frontend.py can compare them
 * against the Python reference numerically rather than by eye.
 *
 * Usage:
 *   ReplaySessionKt <session.csv> <blackoutStartS> <blackoutEndS>
 */

private const val NANOS_PER_SECOND = 1_000_000_000.0

fun main(args: Array<String>) {
    if (args.size < 3) {
        System.err.println("usage: ReplaySessionKt <session.csv> <blackoutStartS> <blackoutEndS>")
        exitProcess(2)
    }

    val path = args[0]
    val blackoutStart = args[1].toDouble()
    val blackoutEnd = args[2].toDouble()

    // Channel P's R is a measured number, not an aspiration. This matches the value
    // the benchmark config carries for the physics channel.
    val pipeline = FusionPipeline(
        PipelineConfig(physics = PhysicsSpeedConfig(rMps = 2.0))
    )

    var blackoutApplied = false
    var blackoutEnded = false

    var lastSnapshot: FusionSnapshot? = null
    var snapshotAtBlackoutEnd: FusionSnapshot? = null
    var worstErrorM = 0.0
    var sumSquaredErrorM = 0.0
    var errorSamples = 0
    var imuCount = 0
    var gnssCount = 0
    var withheldCount = 0

    File(path).forEachLine { raw ->
        val line = raw.trim()
        if (line.isEmpty()) return@forEachLine
        val f = line.split(",")
        val t = f[1].toDouble()
        val tNs = (t * NANOS_PER_SECOND).toLong()

        if (!blackoutApplied && t >= blackoutStart) {
            pipeline.setBlackout(true, tNs)
            blackoutApplied = true
        }
        if (blackoutApplied && !blackoutEnded && t >= blackoutEnd) {
            snapshotAtBlackoutEnd = lastSnapshot
            pipeline.setBlackout(false, tNs)
            blackoutEnded = true
        }

        when (f[0]) {
            "imu" -> {
                imuCount++
                val snapshot = pipeline.onImu(
                    tNs,
                    doubleArrayOf(f[2].toDouble(), f[3].toDouble(), f[4].toDouble()),
                    doubleArrayOf(f[5].toDouble(), f[6].toDouble(), f[7].toDouble())
                )
                if (snapshot != null) {
                    lastSnapshot = snapshot
                    if (snapshot.blackout) {
                        val error = snapshot.driftMeters
                        if (error > worstErrorM) worstErrorM = error
                        sumSquaredErrorM += error * error
                        errorSamples++
                    }
                }
            }

            "gnss" -> {
                gnssCount++
                if (blackoutApplied && !blackoutEnded) withheldCount++
                pipeline.onGnss(
                    FusionPipeline.GnssFix(
                        tNs = tNs,
                        latDeg = f[2].toDouble(),
                        lonDeg = f[3].toDouble(),
                        speedMps = f[4].toDouble(),
                        bearingDeg = f[5].toDouble(),
                        accuracyM = f[6].toDouble()
                    )
                )
            }
        }
    }

    val end = snapshotAtBlackoutEnd ?: lastSnapshot
    if (end == null) {
        System.err.println("pipeline never reached RUNNING: no snapshot to report")
        exitProcess(1)
    }

    val leveling = pipeline.leveling
    val gravity = leveling?.gravityMagnitude ?: 0.0
    val axis = pipeline.forwardAxisResolved

    // Reconstructed from the live snapshot. The number that goes on a slide comes
    // from tools/phone_replay/run.py, computed offline from the raw log, so nothing
    // in this harness can flatter the reported result.
    val distance = end.blackoutDistanceM
    val errorAtEnd = end.driftMeters
    val driftPercent = if (distance > 1.0) errorAtEnd / distance * 100.0 else Double.NaN
    val rms = if (errorSamples > 0) kotlin.math.sqrt(sumSquaredErrorM / errorSamples) else 0.0
    val coastError = hypot(end.coastNorth - end.lastTruthNorth, end.coastEast - end.lastTruthEast)

    println("imu_samples=$imuCount")
    println("gnss_fixes=$gnssCount")
    println("withheld_fixes=$withheldCount")
    println("phase=${end.phase}")
    println("gravity_magnitude=$gravity")
    println("forward_axis_resolved=$axis")
    val axisValue = pipeline.forwardAxisValue
    println("forward_axis_x=${axisValue?.get(0) ?: Double.NaN}")
    println("forward_axis_y=${axisValue?.get(1) ?: Double.NaN}")
    println("blackout_distance_m=$distance")
    println("error_at_end_m=$errorAtEnd")
    println("worst_error_m=$worstErrorM")
    println("rms_error_m=$rms")
    println("drift_percent=$driftPercent")
    println("coast_only_error_m=$coastError")
    println("channel_p_speed_mps=${pipeline.channelPSpeedMps}")
    println("mean_step_latency_ms=${end.stepLatencyMs}")
    println("imu_rate_hz=${end.imuRateHz}")
}
