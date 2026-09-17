package org.sih26.deadreckoning.fusion

import kotlin.math.cos
import kotlin.math.atan2
import kotlin.math.PI

/**
 * Local tangent north/east frame with its origin at the first GNSS fix.
 *
 * Equirectangular about the origin. Over the few kilometres a demo route covers the
 * error is well under a metre, and it keeps this port a direct match to the Python
 * reference, whose state is [pn, pe, vn, ve, psi, ba, bg] with psi measured from
 * north and ve = speed * sin(psi).
 *
 * That convention is the thing to get right. Get it backwards and every trajectory
 * looks mirrored, which is obvious on a map and invisible in a drift number.
 */
class LocalFrame(val lat0Deg: Double, val lon0Deg: Double) {

    private val eastScale = EARTH_RADIUS_M * cos(Math.toRadians(lat0Deg))

    /** Returns [north, east] in metres relative to the origin fix. */
    fun toNorthEast(latDeg: Double, lonDeg: Double): DoubleArray = doubleArrayOf(
        Math.toRadians(latDeg - lat0Deg) * EARTH_RADIUS_M,
        Math.toRadians(lonDeg - lon0Deg) * eastScale
    )

    /** Inverse projection, for drawing a fused track on a map. */
    fun toLatLon(north: Double, east: Double): DoubleArray = doubleArrayOf(
        lat0Deg + Math.toDegrees(north / EARTH_RADIUS_M),
        lon0Deg + Math.toDegrees(east / eastScale)
    )

    companion object {
        const val EARTH_RADIUS_M = 6378137.0
    }
}

/** GNSS bearing in degrees from north, to the filter's psi in radians. */
fun bearingDegToPsi(bearingDeg: Double): Double = Math.toRadians(bearingDeg)

/** Wrap an angle to (-pi, pi]. Used for display and for heading differences, never
 * inside the filter, whose psi is deliberately allowed to run unwrapped so the CTCV
 * propagation stays continuous across a north crossing. */
fun wrapToPi(angleRad: Double): Double {
    var a = angleRad
    while (a > PI) a -= 2.0 * PI
    while (a <= -PI) a += 2.0 * PI
    return a
}

/** Heading in degrees from north, for display. */
fun psiToBearingDeg(psiRad: Double): Double {
    val deg = Math.toDegrees(atan2(kotlin.math.sin(psiRad), cos(psiRad)))
    return if (deg < 0.0) deg + 360.0 else deg
}
