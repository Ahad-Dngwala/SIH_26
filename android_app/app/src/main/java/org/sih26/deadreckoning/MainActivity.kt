package org.sih26.deadreckoning

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.List
import androidx.compose.material.icons.filled.Navigation
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import org.sih26.deadreckoning.sensors.SessionRecordingService
import org.sih26.deadreckoning.sessions.SessionRecord
import org.sih26.deadreckoning.sessions.SessionStore
import org.sih26.deadreckoning.ui.TrackKind
import org.sih26.deadreckoning.ui.TrackPoint
import org.sih26.deadreckoning.ui.TrajectoryCanvas
import java.io.File
import java.util.Locale

/**
 * Three destinations, matching how a field test actually happens: watch the live
 * pipeline while driving, review what got recorded afterwards, and dig into raw
 * diagnostics only when something looks wrong. Keeping diagnostics off the main
 * screen was an explicit call: the person operating the phone during a demo needs
 * four or five numbers, not a debug console.
 */
private enum class Tab(val label: String) { LIVE("Live"), SESSIONS("Sessions"), DIAGNOSTICS("Diagnostics") }

class MainActivity : ComponentActivity() {
    private val permissions = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val required = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
        }.toTypedArray()
        if (required.any { ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED }) {
            permissions.launch(required)
        }
        setContent {
            MaterialTheme {
                Surface(Modifier.fillMaxSize()) {
                    App(
                        start = { ContextCompat.startForegroundService(this, intent(SessionRecordingService.ACTION_START)) },
                        stop = { startService(intent(SessionRecordingService.ACTION_STOP)) },
                        blackout = { active -> startService(intent(SessionRecordingService.ACTION_SET_BLACKOUT).putExtra(SessionRecordingService.EXTRA_BLACKOUT_ACTIVE, active)) },
                        listSessions = { SessionStore.open(this).recent() },
                        deleteSession = { SessionStore.open(this).delete(it) },
                        export = ::export
                    )
                }
            }
        }
    }

    private fun intent(action: String) = Intent(this, SessionRecordingService::class.java).setAction(action)

    private fun export(file: File) {
        val uri = FileProvider.getUriForFile(this, "$packageName.fileprovider", file)
        startActivity(
            Intent.createChooser(
                Intent(Intent.ACTION_SEND).apply {
                    type = "application/x-ndjson"
                    putExtra(Intent.EXTRA_STREAM, uri)
                    addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                },
                "Export raw SIH26 session"
            )
        )
    }
}

@Composable
private fun App(
    start: () -> Unit,
    stop: () -> Unit,
    blackout: (Boolean) -> Unit,
    listSessions: () -> List<SessionRecord>,
    deleteSession: (SessionRecord) -> Unit,
    export: (File) -> Unit
) {
    var tab by remember { mutableStateOf(Tab.LIVE) }
    var telemetry by remember { mutableStateOf<SessionRecordingService.RecordingTelemetry?>(null) }
    var frontEndStatus by remember { mutableStateOf("") }
    val trail = remember { mutableStateListOf<TrackPoint>() }
    var records by remember { mutableStateOf(listSessions()) }

    DisposableEffect(Unit) {
        SessionRecordingService.telemetryListener = { t ->
            telemetry = t
            val snap = t.snapshot
            if (t.recording && snap != null) {
                trail.add(TrackPoint(snap.lastTruthNorth, snap.lastTruthEast, TrackKind.TRUTH))
                trail.add(TrackPoint(snap.fusedNorth, snap.fusedEast, TrackKind.FUSED))
                trail.add(TrackPoint(snap.coastNorth, snap.coastEast, TrackKind.COAST))
                // Cap the trail so a long drive does not grow the canvas's per-frame
                // work without bound; 5 Hz telemetry * 3 points/sample means ~10
                // minutes of history before the oldest points start dropping.
                while (trail.size > 9000) repeat(3) { trail.removeAt(0) }
            }
        }
        SessionRecordingService.statusListener = { frontEndStatus = it }
        onDispose {
            SessionRecordingService.telemetryListener = null
            SessionRecordingService.statusListener = null
        }
    }

    Scaffold(bottomBar = {
        NavigationBar {
            NavigationBarItem(tab == Tab.LIVE, { tab = Tab.LIVE }, { Icon(Icons.Default.Navigation, null) }, label = { Text(Tab.LIVE.label) })
            NavigationBarItem(tab == Tab.SESSIONS, { tab = Tab.SESSIONS; records = listSessions() }, { Icon(Icons.Default.List, null) }, label = { Text(Tab.SESSIONS.label) })
            NavigationBarItem(tab == Tab.DIAGNOSTICS, { tab = Tab.DIAGNOSTICS }, { Icon(Icons.Default.Info, null) }, label = { Text(Tab.DIAGNOSTICS.label) })
        }
    }) { padding ->
        Box(Modifier.padding(padding)) {
            when (tab) {
                Tab.LIVE -> LiveScreen(
                    t = telemetry, trail = trail,
                    start = { trail.clear(); start() }, stop = stop, blackout = blackout
                )
                Tab.SESSIONS -> SessionsScreen(records, export) { record ->
                    deleteSession(record)
                    records = listSessions()
                }
                Tab.DIAGNOSTICS -> DiagnosticsScreen(telemetry, frontEndStatus)
            }
        }
    }
}

@Composable
private fun LiveScreen(
    t: SessionRecordingService.RecordingTelemetry?,
    trail: List<TrackPoint>,
    start: () -> Unit,
    stop: () -> Unit,
    blackout: (Boolean) -> Unit
) {
    val recording = t?.recording == true
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("SIH26 Field Test", style = MaterialTheme.typography.headlineSmall)

        Button(onClick = { if (recording) stop() else start() }, modifier = Modifier.fillMaxWidth()) {
            Text(if (recording) "Stop recording" else "Start recording")
        }

        if (!recording) {
            Text(
                "Ready. Start parked so leveling can find gravity, then drive until " +
                    "GNSS heading initialises the filter.",
                style = MaterialTheme.typography.bodyMedium
            )
            return@Column
        }

        val phase = when {
            t?.blackout == true -> "GNSS BLACKOUT"
            t?.waitingForMovement == true -> "WAITING FOR MOVEMENT"
            else -> t?.phase?.name?.replace('_', ' ') ?: "STARTING"
        }
        PhaseCard(phase = phase, blackout = t?.blackout == true)

        TrajectoryCanvas(trail, Modifier.fillMaxWidth())

        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(4.dp)) {
                MetricRow("Speed", "${t?.gnssSpeedMps?.let { fmt(it * 3.6) } ?: "--"} km/h", "Accuracy", "${t?.gnssAccuracyM?.let(::fmt) ?: "--"} m")
                MetricRow("Distance", "${fmt((t?.distanceM ?: 0.0) / 1000)} km", "Elapsed", duration(t?.elapsedS ?: 0.0))
                MetricRow("IMU samples", "${t?.imuSamples ?: 0}", "GNSS fixes", "${t?.gnssFixes ?: 0}")
                t?.snapshot?.let { snap ->
                    MetricRow("Heading", "${fmt(snap.headingDeg)} deg", "Drift", if (t.blackout) "${fmt(snap.driftMeters)} m (${fmt(snap.driftPercent)}%)" else "n/a - GNSS live")
                }
            }
        }

        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween, verticalAlignment = Alignment.CenterVertically) {
            Column {
                Text("Software GNSS blackout", style = MaterialTheme.typography.bodyLarge)
                Text("Withholds fixes from the filter only; raw GNSS keeps logging as truth.", style = MaterialTheme.typography.bodySmall)
            }
            Switch(checked = t?.blackout == true, onCheckedChange = blackout)
        }
    }
}

@Composable
private fun PhaseCard(phase: String, blackout: Boolean) {
    val container = if (blackout) MaterialTheme.colorScheme.errorContainer else MaterialTheme.colorScheme.secondaryContainer
    val content = if (blackout) MaterialTheme.colorScheme.onErrorContainer else MaterialTheme.colorScheme.onSecondaryContainer
    Card(Modifier.fillMaxWidth(), colors = CardDefaults.cardColors(containerColor = container)) {
        Text(phase, Modifier.padding(12.dp), style = MaterialTheme.typography.titleMedium, color = content)
    }
}

@Composable
private fun MetricRow(label1: String, value1: String, label2: String, value2: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Column(Modifier.weight(1f)) { Text(label1, style = MaterialTheme.typography.labelSmall); Text(value1, style = MaterialTheme.typography.bodyLarge) }
        Column(Modifier.weight(1f)) { Text(label2, style = MaterialTheme.typography.labelSmall); Text(value2, style = MaterialTheme.typography.bodyLarge) }
    }
}

@Composable
private fun SessionsScreen(records: List<SessionRecord>, export: (File) -> Unit, delete: (SessionRecord) -> Unit) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
        Text("Recorded sessions", style = MaterialTheme.typography.headlineSmall)
        if (records.isEmpty()) {
            Text("No completed sessions yet. Recordings appear here after you stop them.")
            return@Column
        }
        LazyColumn(verticalArrangement = Arrangement.spacedBy(10.dp)) {
            items(records, key = { it.id }) { r -> SessionCard(r, export, delete) }
        }
    }
}

@Composable
private fun SessionCard(r: SessionRecord, export: (File) -> Unit, delete: (SessionRecord) -> Unit) {
    var expanded by remember { mutableStateOf(false) }
    var confirmingDelete by remember { mutableStateOf(false) }
    Card(Modifier.fillMaxWidth()) {
        Column(Modifier.padding(12.dp).clickable { expanded = !expanded }, verticalArrangement = Arrangement.spacedBy(4.dp)) {
            Text("Drive - ${r.startedUtc.take(16).replace('T', ' ')} UTC", style = MaterialTheme.typography.titleMedium)
            Text("${duration(r.durationS)} - ${fmt(r.distanceM / 1000)} km - ${r.imuSamples} IMU samples - ${r.gnssFixes} GNSS fixes")
            if (expanded) {
                Text("Blackout duration: ${duration(r.blackoutDurationS)}")
                Text("Final blackout drift: ${r.finalDriftM?.let { "${fmt(it)} m" } ?: "not measured (no blackout this session)"}")
                Text("Raw file: ${r.rawFile.name}", style = MaterialTheme.typography.bodySmall)
                Text("File on disk: ${r.rawFile.length() / 1024} KB", style = MaterialTheme.typography.bodySmall)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    Button({ export(r.rawFile) }) { Text("Export JSONL") }
                    OutlinedButton({ confirmingDelete = true }) { Text("Delete") }
                }
            }
        }
    }
    if (confirmingDelete) {
        AlertDialog(
            onDismissRequest = { confirmingDelete = false },
            title = { Text("Delete this session?") },
            text = { Text("This removes the raw JSONL and its metadata from the phone permanently. Export first if you need it.") },
            confirmButton = { TextButton({ confirmingDelete = false; delete(r) }) { Text("Delete") } },
            dismissButton = { TextButton({ confirmingDelete = false }) { Text("Cancel") } }
        )
    }
}

@Composable
private fun DiagnosticsScreen(t: SessionRecordingService.RecordingTelemetry?, frontEndStatus: String) {
    Column(Modifier.fillMaxSize().padding(16.dp), verticalArrangement = Arrangement.spacedBy(10.dp)) {
        Text("Diagnostics", style = MaterialTheme.typography.headlineSmall)
        Text(
            "Raw pipeline internals. Nothing here is smoothed for presentation; it is " +
                "exactly what the fusion front end reports.",
            style = MaterialTheme.typography.bodySmall
        )
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                DiagRow("Recording", if (t?.recording == true) "yes" else "no")
                DiagRow("Pipeline phase", t?.phase?.name ?: "n/a")
                DiagRow("Waiting for moving fix", if (t?.waitingForMovement == true) "yes" else "no")
                DiagRow("Software blackout", if (t?.blackout == true) "ACTIVE" else "off")
                t?.snapshot?.let { snap ->
                    DiagRow("IMU sample rate", "${fmt(snap.imuRateHz)} Hz")
                    DiagRow("Filter step latency", "${fmt(snap.stepLatencyMs)} ms")
                    DiagRow("ZUPT active this cycle", if (snap.zuptActive) "yes" else "no")
                    DiagRow("Channel P speed", snap.channelPSpeed?.let { "${fmt(it)} m/s" } ?: "not resolved")
                    DiagRow("Fused position (N,E)", "${fmt(snap.fusedNorth)}, ${fmt(snap.fusedEast)} m")
                    DiagRow("Coast position (N,E)", "${fmt(snap.coastNorth)}, ${fmt(snap.coastEast)} m")
                    DiagRow("Truth position (N,E)", "${fmt(snap.lastTruthNorth)}, ${fmt(snap.lastTruthEast)} m")
                }
            }
        }
        Card(Modifier.fillMaxWidth()) {
            Column(Modifier.padding(12.dp)) {
                Text("Front-end status", style = MaterialTheme.typography.titleSmall)
                Text(frontEndStatus.ifBlank { "No GNSS-triggered update yet this session." }, style = MaterialTheme.typography.bodySmall)
            }
        }
        Text(
            "ACCEL / GYRO / MAG are raw android.hardware sensors, never fused/virtual " +
                "ones. GPS is LocationManager.GPS_PROVIDER, never FusedLocationProviderClient.",
            style = MaterialTheme.typography.labelSmall
        )
    }
}

@Composable
private fun DiagRow(label: String, value: String) {
    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
        Text(label, style = MaterialTheme.typography.bodyMedium)
        Text(value, style = MaterialTheme.typography.bodyMedium)
    }
}

private fun fmt(value: Double) = String.format(Locale.ROOT, "%.2f", value)
private fun duration(seconds: Double) = "%dm %02ds".format(Locale.ROOT, (seconds / 60).toInt(), (seconds % 60).toInt())
