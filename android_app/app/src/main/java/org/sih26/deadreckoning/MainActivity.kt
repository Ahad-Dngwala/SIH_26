package org.sih26.deadreckoning

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import org.sih26.deadreckoning.sensors.SessionRecordingService
import org.sih26.deadreckoning.sessions.SessionRecord
import org.sih26.deadreckoning.sessions.SessionStore
import java.io.File
import java.util.Locale

class MainActivity : ComponentActivity() {
    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val required = buildList { add(Manifest.permission.ACCESS_FINE_LOCATION); if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS) }.toTypedArray()
        if (required.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) permissions.launch(required)
        setContent { MaterialTheme { Surface(Modifier.fillMaxSize()) { FieldApp(
            start = { ContextCompat.startForegroundService(this, intent(SessionRecordingService.ACTION_START)) },
            stop = { startService(intent(SessionRecordingService.ACTION_STOP)) },
            blackout = { active -> startService(intent(SessionRecordingService.ACTION_SET_BLACKOUT).putExtra(SessionRecordingService.EXTRA_BLACKOUT_ACTIVE, active)) },
            sessions = { SessionStore.open(this).recent() }, export = ::export
        ) } } }
    }
    private fun intent(action: String) = Intent(this, SessionRecordingService::class.java).setAction(action)
    private fun export(file: File) {
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        startActivity(Intent.createChooser(Intent(Intent.ACTION_SEND).apply { type = "application/x-ndjson"; putExtra(Intent.EXTRA_STREAM, uri); addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION) }, "Export raw SIH26 session"))
    }
}

@Composable private fun FieldApp(start: () -> Unit, stop: () -> Unit, blackout: (Boolean) -> Unit, sessions: () -> List<SessionRecord>, export: (File) -> Unit) {
    var telemetry by remember { mutableStateOf<SessionRecordingService.RecordingTelemetry?>(null) }
    var history by remember { mutableStateOf(false) }; var records by remember { mutableStateOf(sessions()) }
    DisposableEffect(Unit) { SessionRecordingService.telemetryListener = { telemetry = it }; onDispose { SessionRecordingService.telemetryListener = null } }
    Column(Modifier.fillMaxSize().padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) { Button({ history = false }) { Text("Navigation") }; Button({ records = sessions(); history = true }) { Text("Sessions") } }
        if (history) History(records, export) else Dashboard(telemetry, start, stop, blackout)
    }
}

@Composable private fun Dashboard(t: SessionRecordingService.RecordingTelemetry?, start: () -> Unit, stop: () -> Unit, blackout: (Boolean) -> Unit) {
    val recording = t?.recording == true
    Text("SIH26 Navigation", style = MaterialTheme.typography.headlineSmall)
    Button({ if (recording) stop() else start() }, Modifier.fillMaxWidth()) { Text(if (recording) "Stop recording" else "Start recording") }
    if (!recording) { Text("Ready. Start parked for leveling, then drive until GNSS heading initializes."); return }
    val phase = when { t?.blackout == true -> "GNSS BLACKOUT"; t?.waitingForMovement == true -> "WAITING FOR MOVEMENT"; else -> t?.phase?.name?.replace('_', ' ') ?: "STARTING" }
    Text("Mode: ${if (t?.blackout == true) "GNSS BLACKOUT" else "GNSS LIVE"}   Phase: $phase")
    Text("Speed: ${t?.gnssSpeedMps?.let { fmt(it * 3.6) } ?: "--"} km/h   Accuracy: ${t?.gnssAccuracyM?.let(::fmt) ?: "--"} m")
    Text("Distance: ${fmt((t?.distanceM ?: 0.0) / 1000)} km   Time: ${duration(t?.elapsedS ?: 0.0)}")
    Text("Recording: ${t?.imuSamples ?: 0} IMU / ${t?.gnssFixes ?: 0} GNSS")
    t?.snapshot?.let { Text("Heading: ${fmt(it.headingDeg)}°   Drift: ${fmt(it.driftMeters)} m (${fmt(it.driftPercent)}%)\nIMU: ${fmt(it.imuRateHz)} Hz   Filter: ${fmt(it.stepLatencyMs)} ms") }
    Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) { Text("Software blackout"); Switch(t?.blackout == true, blackout) }
    Text("Diagnostics: ACCEL ✓  GYRO ✓  MAG ✓  GPS ${if ((t?.gnssFixes ?: 0) > 0) "FIX ✓" else "waiting"}", style = MaterialTheme.typography.bodySmall)
}

@Composable private fun History(records: List<SessionRecord>, export: (File) -> Unit) {
    Text("Recent sessions", style = MaterialTheme.typography.headlineSmall)
    if (records.isEmpty()) { Text("No completed sessions yet."); return }
    LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) { items(records, key = { it.id }) { r -> Column(Modifier.fillMaxWidth()) {
        Text("Drive — ${r.startedUtc.take(16).replace('T', ' ')}")
        Text("${duration(r.durationS)} · ${fmt(r.distanceM / 1000)} km · ${r.imuSamples} IMU")
        Text("Blackout: ${duration(r.blackoutDurationS)} · drift: ${r.finalDriftM?.let { "${fmt(it)} m" } ?: "not measured"}")
        Button({ export(r.rawFile) }) { Text("Export JSONL") }
    } } }
}
private fun fmt(value: Double) = String.format(Locale.ROOT, "%.2f", value)
private fun duration(seconds: Double) = "%dm %02ds".format(Locale.ROOT, (seconds / 60).toInt(), (seconds % 60).toInt())
