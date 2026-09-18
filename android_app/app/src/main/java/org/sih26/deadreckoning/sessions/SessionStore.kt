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

    fun recent(): List<SessionRecord> = root.listFiles { file ->
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

    companion object {
        fun open(context: Context): SessionStore = SessionStore(
            File(context.getExternalFilesDir(null), "sessions")
        )

        fun nowUtc(): String = Instant.now().toString()
    }
}
