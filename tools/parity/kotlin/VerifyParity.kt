package org.sih26.deadreckoning.parity

import org.sih26.deadreckoning.fusion.DualChannelUkf
import org.sih26.deadreckoning.fusion.FusionConfig
import org.sih26.deadreckoning.fusion.UkfState
import java.io.File
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.system.exitProcess

/**
 * Replay tools/parity/fixture_ukf_reference.json through the Kotlin UKF and report
 * divergence from the Python reference.
 *
 * This is the reason reimplementing the filter in Kotlin was a reasonable decision
 * rather than a reckless one. Three implementations of a numerically delicate filter
 * diverge silently, and a subtly wrong Kalman filter does not crash, it produces
 * plausible trajectories that are quietly wrong, and you find out on stage.
 *
 * Takes the flattened CSV produced by tools/parity/export_fixture_csv.py, so the
 * whole gate runs with kotlinc and a Python interpreter and nothing else.
 *
 * The fixture is the reference by convention, not because it is known to be correct.
 * If this fails, go find out which implementation is wrong. Do not regenerate the
 * fixture to make it pass.
 */

private class Fixture(
    val meta: Map<String, Double>,
    val cycles: List<Cycle>
)

private class Cycle(
    val index: Int,
    val tS: Double,
    val dtS: Double,
    val gyroYawRps: Double,
    val gnssAvailable: Boolean,
    val gnssN: Double,
    val gnssE: Double,
    val expectedPn: Double,
    val expectedPe: Double,
    val expectedVn: Double,
    val expectedVe: Double,
    val expectedPsi: Double
)

/** Python writes NaN as "nan"; java.lang.Double only accepts "NaN". A blackout
 * cycle's absent GNSS position is written as NaN on purpose, so this has to parse
 * rather than throw. */
private fun parseDouble(token: String): Double {
    val trimmed = token.trim()
    return when (trimmed.lowercase()) {
        "nan" -> Double.NaN
        "inf", "infinity" -> Double.POSITIVE_INFINITY
        "-inf", "-infinity" -> Double.NEGATIVE_INFINITY
        else -> trimmed.toDouble()
    }
}

private fun readFixture(path: String): Fixture {
    val meta = mutableMapOf<String, Double>()
    val cycles = mutableListOf<Cycle>()
    var columns: List<String>? = null

    File(path).forEachLine { raw ->
        val line = raw.trim()
        when {
            line.isEmpty() -> Unit
            line.startsWith("#") -> {
                val body = line.removePrefix("#").trim()
                val eq = body.indexOf('=')
                if (eq > 0) {
                    val value = runCatching { parseDouble(body.substring(eq + 1)) }.getOrNull()
                    if (value != null) meta[body.substring(0, eq)] = value
                }
            }
            columns == null -> columns = line.split(",")
            else -> {
                val f = line.split(",")
                cycles.add(
                    Cycle(
                        index = f[0].toInt(),
                        tS = parseDouble(f[1]),
                        dtS = parseDouble(f[2]),
                        gyroYawRps = parseDouble(f[3]),
                        gnssAvailable = f[4].trim().toInt() == 1,
                        gnssN = parseDouble(f[5]),
                        gnssE = parseDouble(f[6]),
                        expectedPn = parseDouble(f[7]),
                        expectedPe = parseDouble(f[8]),
                        expectedVn = parseDouble(f[9]),
                        expectedVe = parseDouble(f[10]),
                        expectedPsi = parseDouble(f[11])
                    )
                )
            }
        }
    }
    return Fixture(meta, cycles)
}

fun main(args: Array<String>) {
    if (args.isEmpty()) {
        System.err.println("usage: VerifyParityKt <flattened-fixture.csv>")
        exitProcess(2)
    }

    val fixture = readFixture(args[0])
    val meta = fixture.meta

    val ukf = DualChannelUkf(
        UkfState(
            posN = meta.getValue("init_pos_n"),
            posE = meta.getValue("init_pos_e"),
            velN = meta.getValue("init_vel_n"),
            velE = meta.getValue("init_vel_e"),
            heading = meta.getValue("init_heading")
        ),
        FusionConfig()
    )

    var worstPosition = 0.0
    var worstVelocity = 0.0
    var worstHeading = 0.0
    val failures = mutableListOf<String>()

    val tolPosition = meta.getValue("tol_position_m")
    val tolVelocity = meta.getValue("tol_velocity_mps")
    val tolHeading = meta.getValue("tol_heading_rad")

    for (cycle in fixture.cycles) {
        val gnssPos = if (cycle.gnssAvailable) doubleArrayOf(cycle.gnssN, cycle.gnssE) else null

        val state = ukf.step(
            dt = cycle.dtS,
            gyroYaw = cycle.gyroYawRps,
            channelASpeed = null,
            channelBSpeed = null,
            gnssPos = gnssPos,
            gnssVel = null
        )

        val dPosition = hypot(state.posN - cycle.expectedPn, state.posE - cycle.expectedPe)
        val dVelocity = hypot(state.velN - cycle.expectedVn, state.velE - cycle.expectedVe)
        val dHeading = abs(state.heading - cycle.expectedPsi)

        if (dPosition > worstPosition) worstPosition = dPosition
        if (dVelocity > worstVelocity) worstVelocity = dVelocity
        if (dHeading > worstHeading) worstHeading = dHeading

        if (dPosition > tolPosition) {
            failures.add("cycle ${cycle.index} (t=${cycle.tS}s): position off by $dPosition m")
        }
        if (dVelocity > tolVelocity) {
            failures.add("cycle ${cycle.index} (t=${cycle.tS}s): velocity off by $dVelocity m/s")
        }
        if (dHeading > tolHeading) {
            failures.add("cycle ${cycle.index} (t=${cycle.tS}s): heading off by $dHeading rad")
        }
    }

    println("Replayed ${fixture.cycles.size} cycles through the Kotlin UKF")
    println("  worst position_m     $worstPosition   (tolerance $tolPosition)")
    println("  worst velocity_mps   $worstVelocity   (tolerance $tolVelocity)")
    println("  worst heading_rad    $worstHeading   (tolerance $tolHeading)")

    if (failures.isNotEmpty()) {
        println()
        println("${failures.size} divergence(s):")
        failures.take(10).forEach { println("  $it") }
        println()
        println("The Kotlin port and the Python reference disagree. Find out which one is")
        println("wrong. Do not regenerate the fixture to make this pass.")
        exitProcess(1)
    }

    println()
    println("PASS - the Kotlin port reproduces the Python reference.")
}
