package io.github.ayushpatnaikgit.tellwell

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.IBinder
import android.os.PowerManager

/**
 * Runs the one-time model download as a foreground task with a progress notification,
 * so it keeps going when the screen locks or the user switches apps.
 */
class DownloadService : Service() {
    override fun onBind(intent: Intent?): IBinder? = null

    private fun note(text: String, pct: Int, ongoing: Boolean): Notification =
        Notification.Builder(this, CHANNEL)
            .setSmallIcon(R.drawable.ic_launcher_mono)
            .setContentTitle("TellWell")
            .setContentText(text)
            .setOnlyAlertOnce(true)
            .setOngoing(ongoing)
            .apply { if (ongoing) setProgress(100, pct, false) }
            .build()

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        if (intent?.action == ACTION_PAUSE) { ModelDownloader.cancel(); return START_NOT_STICKY }
        if (ModelDownloader.running) return START_NOT_STICKY
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Model download", NotificationManager.IMPORTANCE_LOW))
        startForeground(1, note("Downloading models…", 0, true), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        val wake = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "TellWell:download")
        wake.acquire(60 * 60 * 1000L)
        var lastPct = -1
        ModelDownloader.start(this, { done, total ->
            ModelDownloader.listener?.invoke(done, total, null, false)
            val pct = (done * 100 / total).toInt()
            if (pct != lastPct) { lastPct = pct; nm.notify(1, note("Downloading models… $pct%", pct, true)) }
        }) { error ->
            fun finish(text: String) {
                if (wake.isHeld) wake.release()
                stopForeground(STOP_FOREGROUND_REMOVE)
                nm.notify(2, note(text, 0, false))
                stopSelf()
            }
            if (error != null || !Npu.candidate(this) || Npu.compiled(this)) {
                ModelDownloader.listener?.invoke(0, 0, error, true)
                finish(error ?: "Models ready. TellWell no longer needs the internet.")
            } else {
                // Compile the NPU model for this phone now, while this task keeps the app running (about a minute, once).
                nm.notify(1, Notification.Builder(this, CHANNEL).setSmallIcon(R.drawable.ic_launcher_mono)
                    .setContentTitle("TellWell").setContentText("Getting ready for this phone's NPU: keep TellWell open for about a minute…")
                    .setOngoing(true).setProgress(0, 0, true).build())
                ModelDownloader.listener?.invoke(0, 0, null, true)
                val ready = { _: Any? ->
                    ModelDownloader.listener?.invoke(0, 0, null, true)
                    finish(if (Engine.asrOnNpu) "Ready. Speech runs on this phone's NPU." else "Models ready. TellWell no longer needs the internet.")
                }
                if (Engine.ready) Engine.upgradeToNpu(this) { ready(null) } else Engine.ensureLoaded(this) { ready(it) }
            }
        }
        return START_NOT_STICKY
    }

    companion object {
        const val CHANNEL = "download"
        const val ACTION_PAUSE = "pause"
    }
}
