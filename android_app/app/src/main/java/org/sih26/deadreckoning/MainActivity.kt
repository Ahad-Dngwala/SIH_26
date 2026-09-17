package org.sih26.deadreckoning

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import org.sih26.deadreckoning.fusion.FusionSnapshot
import org.sih26.deadreckoning.sensors.SessionRecordingService
import java.util.Locale

/**
 * Tier 1 item 7 from the handoff: minimal UI, not polished UI. What is shown is
 * exactly what the recording protocol (docs/HANDOFF_kotlin_demo_app*.md section 7)
 * says has to stay visible for the demo to be credible: mode, elapsed blackout time,
 * distance travelled during blackout, drift metres and percent, IMU rate, and step
 * latency. A plain Compose column is fine on video and costs hours less than the
 * MapLibre canvas this Tier 1 does not build - a live trajectory canvas is Tier 2/3
 * polish, not part of the measurement itself.
 */
class MainActivity : ComponentActivity() {

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { /* result observed by the user retrying "Start recording" */ }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val required = buildList {
            add(Manifest.permission.ACCESS_FINE_LOCATION)
            if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
        }.toTypedArray()

        if (required.any {
                ContextCompat.checkSelfPermission(this, it) != PackageManager.PERMISSION_GRANTED
            }
        ) {
            permissionLauncher.launch(required)
        }

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    RecordingScreen(
                        onStart = {
                            ContextCompat.startForegroundService(
                                this,
                                serviceIntent(SessionRecordingService.ACTION_START)
                            )
                        },
                        onStop = { startService(serviceIntent(SessionRecordingService.ACTION_STOP)) },
                        onBlackoutToggle = { active ->
                            val intent = serviceIntent(SessionRecordingService.ACTION_SET_BLACKOUT)
                            intent.putExtra(SessionRecordingService.EXTRA_BLACKOUT_ACTIVE, active)
                            startService(intent)
                        }
                    )
                }
            }
        }
    }

    private fun serviceIntent(action: String) =
        Intent(this, SessionRecordingService::class.java).setAction(action)
}

@Composable
private fun RecordingScreen(
    onStart: () -> Unit,
    onStop: () -> Unit,
    onBlackoutToggle: (Boolean) -> Unit
) {
    var recording by remember { mutableStateOf(false) }
    var blackoutOn by remember { mutableStateOf(false) }
    var snapshot by remember { mutableStateOf<FusionSnapshot?>(null) }
    var status by remember { mutableStateOf("") }

    // The service posts both callbacks to the main Looper. Installing and removing
    // them as a Compose effect prevents a stale Activity from being retained after
    // rotation or navigation.
    DisposableEffect(Unit) {
        SessionRecordingService.snapshotListener = { snapshot = it }
        SessionRecordingService.statusListener = { status = it }
        onDispose {
            SessionRecordingService.snapshotListener = null
            SessionRecordingService.statusListener = null
        }
    }

    Column(
        modifier = Modifier.fillMaxSize().padding(24.dp),
        verticalArrangement = Arrangement.spacedBy(12.dp)
    ) {
        Text("SIH Dead Reckoning", style = MaterialTheme.typography.headlineSmall)

        Button(onClick = {
            if (!recording) onStart() else onStop()
            recording = !recording
        }) {
            Text(if (recording) "Stop recording" else "Start recording")
        }

        if (recording) {
            Column(verticalArrangement = Arrangement.spacedBy(4.dp)) {
                Text("Blackout")
                Switch(checked = blackoutOn, onCheckedChange = {
                    blackoutOn = it
                    onBlackoutToggle(it)
                })
            }

            val s = snapshot
            if (s != null) {
                Text("Mode: ${if (s.blackout) "GNSS BLACKOUT" else "GNSS live"}")
                Text("Speed: ${fmt(s.speedMps)} m/s   Heading: ${fmt(s.headingDeg)} deg")
                if (s.blackout) {
                    Text("Blackout elapsed: ${fmt(s.blackoutElapsedS)} s")
                    Text("Distance in blackout: ${fmt(s.blackoutDistanceM)} m")
                    Text("Drift: ${fmt(s.driftMeters)} m  (${fmt(s.driftPercent)}%)")
                }
                Text("IMU rate: ${fmt(s.imuRateHz)} Hz   step latency: ${fmt(s.stepLatencyMs)} ms")
                Text("Phase: ${s.phase}")
            } else {
                Text("Waiting for the first fix and a stationary window...")
            }

            if (status.isNotEmpty()) Text(status, style = MaterialTheme.typography.bodySmall)
        }
    }
}

private fun fmt(v: Double): String = String.format(Locale.ROOT, "%.2f", v)
