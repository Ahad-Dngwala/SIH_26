package org.sih26.deadreckoning.sensors

import java.io.BufferedWriter
import java.io.File
import java.io.FileWriter
import java.util.concurrent.ArrayBlockingQueue
import java.util.concurrent.TimeUnit
import java.util.concurrent.atomic.AtomicBoolean

/**
 * Writes the JSONL schema that `tools/phone_replay/session.py` reads.
 *
 * This is the single highest-value item in the whole Kotlin build, stated plainly
 * in both handoffs: if the app writes something the loader cannot read, there is no
 * real number, no matter how correct the on-device filter is. So this class is
 * deliberately dumber than everything else here - it does no fusion, no leveling, no
 * interpretation - it only serialises exactly what session.py's module docstring
 * says a record must contain, in the order it says.
 *
 * Format, verbatim from the contract (`t` in seconds since session start, on the
 * monotonic elapsedRealtime base, never wall time):
 *
 *     {"type":"header","schema":1,"device":"...","started_utc":"...","notes":"..."}
 *     {"type":"imu","t":..,"ax":..,"ay":..,"az":..,"gx":..,"gy":..,"gz":..,
 *      "mx":..,"my":..,"mz":..}
 *     {"type":"gnss","t":..,"lat":..,"lon":..,"speed":..,"bearing":..,
 *      "accuracy":..,"withheld":..}
 *
 * `mx,my,mz` are omitted per line when no magnetometer sample is available for that
 * IMU record, matching the optional-field contract in session.py: a log without them
 * loads fine, and `PhoneSession.has_magnetometer` is the presence test the loader
 * uses.
 *
 * Threading: a single writer thread drains a bounded queue so the 100 Hz sensor
 * callback never blocks on file I/O. The queue is bounded rather than unbounded on
 * purpose - an app that falls behind on writing should drop the oldest buffered
 * samples and keep logging live ones, not grow without limit and eventually OOM the
 * whole recording. A drop is recorded as a gap in `t`, which
 * `PhoneSession.sanity_report()` already detects and flags.
 */
class SessionLogger(
    val outputFile: File,
    device: String,
    notes: String,
    private val queueCapacity: Int = 20_000
) {
    private val writer: BufferedWriter = BufferedWriter(FileWriter(outputFile, /* append = */ false))
    private val queue = ArrayBlockingQueue<String>(queueCapacity)
    private val running = AtomicBoolean(true)
    private var droppedSamples = 0L

    private val writerThread = Thread({
        while (running.get() || queue.isNotEmpty()) {
            val line = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
            writer.write(line)
            writer.newLine()
        }
        writer.flush()
        writer.close()
    }, "session-logger-writer").apply {
        isDaemon = true
    }

    init {
        val header = buildString {
            append("{\"type\":\"header\",\"schema\":1")
            append(",\"device\":").append(jsonString(device))
            append(",\"started_utc\":").append(jsonString(java.time.Instant.now().toString()))
            append(",\"notes\":").append(jsonString(notes))
            append("}")
        }
        // Written directly, before the writer thread starts, so the header is
        // guaranteed to be the first line even under queue contention.
        writer.write(header)
        writer.newLine()
        writerThread.start()
    }

    /** Log one IMU sample. `tSeconds` is elapsedRealtime-based, matching [SensorEvent.timestamp]
     * converted to seconds by the caller. Accel is raw TYPE_ACCELEROMETER (gravity
     * included), gyro is raw TYPE_GYROSCOPE, both phone-frame. Magnetometer is
     * optional: pass null for any axis not currently available and the field is
     * omitted entirely rather than written as zero, which would be a plausible-
     * looking but wrong reading. */
    fun logImu(
        tSeconds: Double,
        ax: Double, ay: Double, az: Double,
        gx: Double, gy: Double, gz: Double,
        mx: Double? = null, my: Double? = null, mz: Double? = null
    ) {
        val line = buildString {
            append("{\"type\":\"imu\",\"t\":").append(round6(tSeconds))
            append(",\"ax\":").append(round5(ax))
            append(",\"ay\":").append(round5(ay))
            append(",\"az\":").append(round5(az))
            append(",\"gx\":").append(round6(gx))
            append(",\"gy\":").append(round6(gy))
            append(",\"gz\":").append(round6(gz))
            if (mx != null && my != null && mz != null) {
                append(",\"mx\":").append(round3(mx))
                append(",\"my\":").append(round3(my))
                append(",\"mz\":").append(round3(mz))
            }
            append("}")
        }
        offer(line)
    }

    /** Log one GNSS fix. `withheld` records whether this fix was hidden from the
     * live filter this cycle - it is always logged either way, since withheld fixes
     * are the ground truth the drift is measured against. */
    fun logGnss(
        tSeconds: Double,
        lat: Double, lon: Double,
        speedMps: Double, bearingDeg: Double, accuracyM: Double,
        withheld: Boolean
    ) {
        val line = buildString {
            append("{\"type\":\"gnss\",\"t\":").append(round6(tSeconds))
            append(",\"lat\":").append(lat)
            append(",\"lon\":").append(lon)
            append(",\"speed\":").append(round4(speedMps))
            append(",\"bearing\":").append(round3(bearingDeg))
            append(",\"accuracy\":").append(round2(accuracyM))
            append(",\"withheld\":").append(withheld)
            append("}")
        }
        offer(line)
    }

    /** Log the app's own live fused output. Optional, per session.py's docstring:
     * used only to cross-check that this Kotlin filter and the Python one agree on
     * the same input. The drift number itself is always computed offline from the
     * raw imu/gnss records, so a bug here cannot flatter the reported result. */
    fun logFused(tSeconds: Double, north: Double, east: Double) {
        val line = "{\"type\":\"fused\",\"t\":${round6(tSeconds)},\"pn\":$north,\"pe\":$east}"
        offer(line)
    }

    private fun offer(line: String) {
        if (!queue.offer(line)) {
            // Queue full: the writer thread is behind. Drop the oldest, not the
            // newest, so the log stays live rather than stalling on old data, and
            // the resulting gap in `t` is exactly what sanity_report() already knows
            // how to flag.
            queue.poll()
            queue.offer(line)
            droppedSamples++
        }
    }

    val droppedSampleCount: Long get() = droppedSamples

    /** Flush and close. Blocks briefly for the writer thread to drain. Safe to call
     * from the foreground service's onDestroy. */
    fun close() {
        running.set(false)
        writerThread.join(TimeUnit.SECONDS.toMillis(5))
    }

    private fun jsonString(value: String): String {
        val escaped = value.replace("\\", "\\\\").replace("\"", "\\\"")
        return "\"$escaped\""
    }

    // Locale.ROOT is mandatory here, not stylistic: String.format with the device's
    // default locale renders decimals with a comma on many locales (most of Europe
    // included), which would silently turn every numeric field into invalid JSON on
    // exactly the kind of device this app is meant to run on. This was caught by
    // review, not by a failing test, since the emulator/CI locale is usually
    // US/UK - it would have passed every test here and broken on a real phone.
    private fun round6(v: Double) = String.format(java.util.Locale.ROOT, "%.6f", v)
    private fun round5(v: Double) = String.format(java.util.Locale.ROOT, "%.5f", v)
    private fun round4(v: Double) = String.format(java.util.Locale.ROOT, "%.4f", v)
    private fun round3(v: Double) = String.format(java.util.Locale.ROOT, "%.3f", v)
    private fun round2(v: Double) = String.format(java.util.Locale.ROOT, "%.2f", v)
}
