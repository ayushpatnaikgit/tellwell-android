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
 * Downloads an optional language's models as a foreground task with a progress notification, so it keeps going when
 * the screen locks or the user switches apps. Then loads them, so the keyboard can switch to the language.
 */
class LanguageDownloadService : Service() {
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
        if (intent?.action == ACTION_PAUSE) { Languages.cancel(); return START_NOT_STICKY }
        val lang = Lang.entries.firstOrNull { it.code == intent?.getStringExtra(EXTRA_LANG) }
        if (lang == null || lang == Lang.EN || Languages.downloading != null) return START_NOT_STICKY
        val nm = getSystemService(NotificationManager::class.java)
        nm.createNotificationChannel(NotificationChannel(CHANNEL, "Language download", NotificationManager.IMPORTANCE_LOW))
        val what = "Downloading ${lang.englishName}"
        startForeground(NOTE_ID, note("$what…", 0, true), ServiceInfo.FOREGROUND_SERVICE_TYPE_DATA_SYNC)
        val wake = getSystemService(PowerManager::class.java).newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "TellWell:language")
        wake.acquire(60 * 60 * 1000L)
        var lastPct = -1
        Languages.download(this, lang, { done, total ->
            Languages.listener?.invoke(lang, done, total, null, false)
            val pct = (done * 100 / total).toInt()
            if (pct != lastPct) { lastPct = pct; nm.notify(NOTE_ID, note("$what… $pct%", pct, true)) }
        }) { error ->
            fun finish(text: String) {
                if (wake.isHeld) wake.release()
                stopForeground(STOP_FOREGROUND_REMOVE)
                nm.notify(NOTE_ID + 1, note(text, 0, false))
                stopSelf()
            }
            if (error != null) {
                Languages.listener?.invoke(lang, 0, 0, error, true)
                finish(error)
            } else {
                // Load the new models now, so the keyboard offers the language straight away.
                Engine.loadLanguage(this, lang) { ok ->
                    Languages.listener?.invoke(lang, 0, 0, if (ok) null else "${lang.englishName} couldn't load on this phone.", true)
                    finish(if (ok) "${lang.englishName} is ready. Switch to it on TellWell's keyboard." else "${lang.englishName} couldn't load on this phone.")
                }
            }
        }
        return START_NOT_STICKY
    }

    companion object {
        const val CHANNEL = "language-download"
        const val ACTION_PAUSE = "pause"
        const val EXTRA_LANG = "lang"
        private const val NOTE_ID = 10
    }
}
