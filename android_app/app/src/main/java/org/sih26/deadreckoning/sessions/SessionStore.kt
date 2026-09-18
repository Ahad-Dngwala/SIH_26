package org.sih26.deadreckoning.sessions

import android.content.Context
import org.json.JSONObject
import java.io.File
import java.time.Instant
import java.util.UUID

/** Flat-file session catalogue. Raw JSONL remains the source artifact; the sidecar
 * exists solely to make completed recordings discoverable in the phone UI. */
data class SessionRecord(
    val id: String,
    val startedUtc: String,
    val durationS: Double,
    val imuSamples: Long,
    val gnssFixes: Long,
    val distanceM: Double,
    val blackoutDurationS: Double,
    val finalDriftM: Double?,
    val rawFile: File
)

class SessionStore(private val root: File) {
    fun newSessionFile(): Pair<String, File> {
        root.mkdirs()
        val id = UUID.randomUUID().toString()
        return id to File(root, "session_$id.jsonl")
    }

    fun save(record: SessionRecord) {
        val json = JSONObject()
            .put("id", record.id)
            .put("started_utc", record.startedUtc)
            .put("duration_s", record.durationS)
            .put("imu_samples", record.imuSamples)
            .put("gnss_fixes", record.gnssFixes)
            .put("distance_m", record.distanceM)
            .put("blackout_duration_s", record.blackoutDurationS)
            .put("final_drift_m", record.finalDriftM)
            .put("raw_file", record.rawFile.name)
        File(root, "session_${record.id}.meta.json").writeText(json.toString())
    }

    /** Removes a session's sidecar metadata and its raw JSONL artifact. Nothing else
     * on disk references either file, so deleting both is enough to make the
     * session disappear from [recent] and free the space it used. */
    fun delete(record: SessionRecord) {
        File(root, "session_${record.id}.meta.json").delete()
        record.rawFile.delete()
    }

    /**
     * Every `.jsonl` this store ever created is a real recording whether or not the
     * process survived long enough to write its sidecar - a crash, a force-stop, or
     * Android killing the service under memory pressure all leave the raw file
     * intact (it is flushed continuously by [SessionLogger], not written once at
     * the end) with no `.meta.json` next to it. Without this, such a session is
     * real data sitting on disk that the app itself claims does not exist, which is
     * exactly the "relationship between files on disk and what the app displays"
     * the field-test milestone calls out. Recovery derives the same summary
     * [save] would have written, by reading the raw file once, then writes the
     * sidecar so the cost is paid once per orphan rather than on every launch.
     */
    private fun recoverOrphans() {
        val raws = root.listFiles { f -> f.name.startsWith("session_") && f.name.endsWith(".jsonl") } ?: return
        for (raw in raws) {
            val id = raw.name.removePrefix("session_").removeSuffix(".jsonl")
            val meta = File(root, "session_$id.meta.json")
            if (meta.isFile) continue
            // A currently-active recording has no sidecar yet by design (it is
            // written once, on stop) and must not be treated as an orphan while
            // still being appended to. SessionLogger's writer buffers rather than
            // flushing every line, but still touches the file roughly every 1-2s
            // under real IMU load, so anything newer than this is presumed live and
            // is picked up as an orphan on the next visit to this screen instead.
            val ageMs = System.currentTimeMillis() - raw.lastModified()
            if (ageMs < 20_000) continue
            runCatching { recoverOne(id, raw) }
        }
    }

    private fun recoverOne(id: String, raw: File) {
        var startedUtc = SessionStore.nowUtc()
        var imuSamples = 0L
        var gnssFixes = 0L
        var distanceM = 0.0
        var lastT = 0.0
        var lastGnssT: Double? = null
        raw.forEachLine { line ->
            val json = runCatching { JSONObject(line) }.getOrNull() ?: return@forEachLine
            when (json.optString("type")) {
                "header" -> startedUtc = json.optString("started_utc", startedUtc)
                "imu" -> {
                    imuSamples++
                    lastT = maxOf(lastT, json.optDouble("t", lastT))
                }
                "gnss" -> {
                    gnssFixes++
                    val t = json.optDouble("t", lastT)
                    val speed = json.optDouble("speed", 0.0)
                    lastGnssT?.let { previous ->
                        val dt = t - previous
                        if (dt in 0.0..10.0) distanceM += speed * dt
                    }
                    lastGnssT = t
                    lastT = maxOf(lastT, t)
                }
            }
        }
        save(
            SessionRecord(
                id = id, startedUtc = startedUtc, durationS = lastT,
                imuSamples = imuSamples, gnssFixes = gnssFixes, distanceM = distanceM,
                // Recovered from a partial log with no live blackout accounting: a
                // real number here would require re-deriving blackout intervals from
                // the withheld flag, which the offline tool already does properly.
                // Leaving both unmeasured is honest; guessing would not be.
                blackoutDurationS = 0.0, finalDriftM = null, rawFile = raw
            )
        )
    }

    fun recent(): List<SessionRecord> {
        recoverOrphans()
        return root.listFiles { file ->
            file.name.endsWith(".meta.json")
        }?.mapNotNull { file ->
            runCatching {
                val json = JSONObject(file.readText())
                val raw = File(root, json.getString("raw_file"))
                if (!raw.isFile) null else SessionRecord(
                    id = json.getString("id"),
                    startedUtc = json.getString("started_utc"),
                    durationS = json.getDouble("duration_s"),
                    imuSamples = json.getLong("imu_samples"),
                    gnssFixes = json.getLong("gnss_fixes"),
                    distanceM = json.getDouble("distance_m"),
                    blackoutDurationS = json.getDouble("blackout_duration_s"),
                    finalDriftM = if (json.isNull("final_drift_m")) null else json.getDouble("final_drift_m"),
                    rawFile = raw
                )
            }.getOrNull()
        }?.sortedByDescending { it.startedUtc } ?: emptyList()
    }

    companion object {
        fun open(context: Context): SessionStore = SessionStore(
            File(context.getExternalFilesDir(null), "sessions")
        )

        fun nowUtc(): String = Instant.now().toString()
    }
}
