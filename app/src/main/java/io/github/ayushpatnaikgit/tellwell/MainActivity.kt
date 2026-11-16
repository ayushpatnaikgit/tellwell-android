package io.github.ayushpatnaikgit.tellwell

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Typeface
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.Settings
import android.text.SpannableStringBuilder
import android.text.Spanned
import android.text.style.ForegroundColorSpan
import android.text.style.StrikethroughSpan
import android.text.style.StyleSpan
import android.util.Log
import android.view.Gravity
import android.view.View
import android.view.WindowInsets
import android.view.WindowInsetsController
import android.view.inputmethod.InputMethodManager
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import java.io.File
import java.io.RandomAccessFile
import java.nio.ByteBuffer
import java.nio.ByteOrder

/** Setup: microphone permission, enabling the keyboard, a place to try it, model status. */
class MainActivity : Activity() {
    private val main = Handler(Looper.getMainLooper())
    private lateinit var p: Palette
    private lateinit var micStep: Step
    private lateinit var kbdStep: Step
    private lateinit var status: TextView
    private lateinit var mark: MarkView
    private lateinit var modelStep: Step

    /** A setup row: teal number (a check when done), title and note, and a teal text action. */
    private inner class Step(val n: Int, title: String, body: String, action: String, onAction: () -> Unit) {
        val view = LinearLayout(this@MainActivity).apply {
            orientation = LinearLayout.HORIZONTAL
            gravity = Gravity.CENTER_VERTICAL
            setPadding(0, dpi(16), 0, dpi(16))
        }
        private val badge = TextView(this@MainActivity).apply {
            gravity = Gravity.CENTER; textSize = 15f; typeface = Fonts.inter(this@MainActivity, 700)
        }
        val bodyView = TextView(this@MainActivity).apply {
            textSize = 14f; typeface = Fonts.inter(this@MainActivity, 400); setTextColor(p.muted)
        }
        val button = TextView(this@MainActivity).apply {
            text = action; textSize = 15f; typeface = Fonts.inter(this@MainActivity, 600)
            setTextColor(p.accent); setPadding(dpi(12), dpi(8), 0, dpi(8))
            setOnClickListener { onAction() }
        }
        init {
            view.addView(badge, LinearLayout.LayoutParams(dpi(30), dpi(30)))
            val text = LinearLayout(this@MainActivity).apply { orientation = LinearLayout.VERTICAL; setPadding(dpi(14), 0, 0, 0) }
            text.addView(TextView(this@MainActivity).apply {
                this.text = title; textSize = 16.5f; typeface = Fonts.inter(this@MainActivity, 600); setTextColor(p.ink)
            })
            text.addView(bodyView.apply { this.text = body })
            view.addView(text, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
            view.addView(button)
            setDone(false)
        }
        fun setDone(done: Boolean) {
            badge.text = if (done) "✓" else "$n"
            badge.setTextColor(if (done) p.accentInk else p.accent)
            badge.background = rounded(if (done) p.accent else p.tint, dp(15f))
            button.visibility = if (done) View.GONE else View.VISIBLE
        }
    }

    private fun text(t: CharSequence, size: Float, weight: Int, color: Int, center: Boolean = false) = TextView(this).apply {
        text = t; textSize = size; typeface = Fonts.inter(this@MainActivity, weight); setTextColor(color)
        if (size >= 24f) letterSpacing = -0.03f
        if (center) gravity = Gravity.CENTER_HORIZONTAL
    }

    private fun card() = LinearLayout(this).apply {
        orientation = LinearLayout.VERTICAL
        background = rounded(p.card, dp(22f), p.line, dpi(1))
        elevation = dp(if (p.dark) 0f else 10f)
        setPadding(dpi(20), dpi(18), dpi(20), dpi(20))
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        p = Palette.of(this)
        actionBar?.hide()
        window.decorView.setBackgroundColor(p.bg)
        val light = if (p.dark) 0 else WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS
        window.insetsController?.setSystemBarsAppearance(light,
            WindowInsetsController.APPEARANCE_LIGHT_STATUS_BARS or WindowInsetsController.APPEARANCE_LIGHT_NAVIGATION_BARS)

        val col = LinearLayout(this).apply { orientation = LinearLayout.VERTICAL; clipToPadding = false }
        val scroll = ScrollView(this).apply {
            setBackgroundColor(p.bg)
            isFillViewport = true
            clipToPadding = false
            addView(col)
            setOnApplyWindowInsetsListener { _, insets ->
                val b = insets.getInsets(WindowInsets.Type.systemBars() or WindowInsets.Type.ime())
                col.setPadding(dpi(24), b.top + dpi(36), dpi(24), b.bottom + dpi(32))
                insets
            }
        }
        fun gap(h: Int) = col.addView(View(this), LinearLayout.LayoutParams(1, dpi(h)))
        val wide = LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT)

        // The film's close: the mark dances, then settles into the logo, with the wordmark beneath.
        mark = MarkView(this, p).apply { demoLevel = 0.85f }
        col.addView(mark, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpi(118)))
        gap(10)
        col.addView(text("TellWell", 36f, 700, p.ink, center = true), wide)
        gap(4)
        col.addView(text("Free, private dictation for Android", 16f, 500, p.muted, center = true), wide)
        gap(40)

        // Scene C: the three words, the last in the accent.
        col.addView(text("Blazing fast.", 34f, 700, p.ink))
        col.addView(text("Private.", 34f, 700, p.ink))
        col.addView(text("Free.", 34f, 700, p.accent))
        gap(14)
        col.addView(text("Keep Gboard. Switch to TellWell, talk, and tap. It types what you meant and hands you straight back.",
            16.5f, 400, p.muted).apply { setLineSpacing(0f, 1.3f) })
        gap(26)

        // What you say → what gets typed, like the film's message card.
        val demo = card()
        demo.addView(TextView(this).apply {
            val parts = listOf("Um, so " to true, "I think we should " to false, "uh, " to true, "meet on " to false,
                "Tuesday. No wait, " to true, "Wednesday at three." to false)
            val sb = SpannableStringBuilder()
            for ((t, drop) in parts) {
                val st = sb.length; sb.append(t)
                if (drop) {
                    sb.setSpan(StrikethroughSpan(), st, sb.length - 1, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                    sb.setSpan(ForegroundColorSpan(p.red), st, sb.length, Spanned.SPAN_EXCLUSIVE_EXCLUSIVE)
                }
            }
            text = sb; textSize = 16f; typeface = Fonts.inter(this@MainActivity, 400); setTextColor(p.muted); setLineSpacing(0f, 1.25f)
        })
        demo.addView(View(this).apply { setBackgroundColor(p.line) }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpi(1)).apply {
            topMargin = dpi(14); bottomMargin = dpi(14)
        })
        demo.addView(text("I think we should meet on Wednesday at three.", 18f, 600, p.ink).apply { setLineSpacing(0f, 1.2f) })
        col.addView(demo)
        gap(40)

        col.addView(text("Set up", 24f, 700, p.ink))
        gap(4)
        modelStep = Step(1, "Download the models", "About 1 GB, once. Use Wi-Fi.", "Download") { startDownload() }
        col.addView(modelStep.view)
        fun divider() = col.addView(View(this).apply { setBackgroundColor(p.line) }, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpi(1)))
        divider()
        micStep = Step(2, "Allow the microphone", "TellWell only listens while it's on screen.", "Allow") {
            requestPermissions(arrayOf(Manifest.permission.RECORD_AUDIO), 1)
        }
        col.addView(micStep.view)
        divider()
        kbdStep = Step(3, "Turn on the keyboard", "Gboard stays your main keyboard.", "Open") {
            startActivity(Intent(Settings.ACTION_INPUT_METHOD_SETTINGS))
        }
        col.addView(kbdStep.view)
        divider()
        col.addView(Step(4, "Try it", "Tap the box, then switch to TellWell.", "Switch") {
            getSystemService(InputMethodManager::class.java).showInputMethodPicker()
        }.view)
        gap(6)
        col.addView(EditText(this).apply {
            hint = "Say something here"
            textSize = 16f; typeface = Fonts.inter(this@MainActivity, 400); setTextColor(p.ink); setHintTextColor(p.faint)
            background = rounded(p.card, dp(18f), p.line, dpi(1))
            elevation = dp(if (p.dark) 0f else 6f)
            setPadding(dpi(18), dpi(16), dpi(18), dpi(16)); minLines = 3; gravity = Gravity.TOP
        })
        gap(24)


        status = text("", 13.5f, 500, p.muted, center = true)
        col.addView(status, wide)
        gap(6)
        col.addView(text("What you say never leaves this phone. The internet is only used once, to download the models from GitHub.", 13.5f, 400, p.faint, center = true), wide)
        setContentView(scroll)
        watchDownload()

        if (BuildConfig.DEBUG) {
            if (intent.hasExtra("draft")) {
                Engine.draftK = intent.getIntExtra("draft", 0)
                Engine.draftOverridden = true
            }
            intent.getStringExtra("devices")?.let { Engine.devices = it.split(",") }
            intent.getStringExtra("selftest")?.let { selfTest(it) }
        }
        if (BuildConfig.DEBUG && intent.getBooleanExtra("download", false)) startDownload()
    }

    /** Plays the film's close once: dance, then settle into the logo. */
    private fun playIntro() {
        mark.mode = MarkView.Mode.LISTENING
        main.postDelayed({ mark.mode = MarkView.Mode.WRITING }, 1500)
        main.postDelayed({ mark.mode = MarkView.Mode.IDLE }, 2400)
    }

    override fun onResume() {
        super.onResume()
        playIntro()
        refresh()
        Engine.ensureLoaded(this) { main.post { refresh() } }
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<out String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        refresh()
    }

    private fun refresh() {
        val mic = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED
        val enabled = getSystemService(InputMethodManager::class.java).enabledInputMethodList.any { it.packageName == packageName }
        micStep.setDone(mic)
        kbdStep.setDone(enabled)
        val have = Engine.modelsPresent(this)
        val missingMb = ModelDownloader.missing(this).sumOf { it.size } / 1_000_000
        // A phone already dictating on the CPU can download the faster NPU encoder.
        val upgrade = have && missingMb > 0 && Npu.candidate(this)
        modelStep.setDone(have && !upgrade)
        if (!ModelDownloader.running) modelStep.bodyView.text = when {
            !have -> downloadNote ?: "About ${(missingMb + 50) / 100 * 100} MB, once. Use Wi-Fi."
            upgrade -> downloadNote ?: "Faster speech for this phone's NPU: about $missingMb MB more, once."
            else -> "Downloaded and verified."
        }
        status.text = if (have) Engine.status else "Models not downloaded yet"
        status.setTextColor(if (Engine.ready) p.accent else p.muted)
        // While loading (or compiling for the NPU, about a minute), keep the status line current, and keep the screen on:
        // the compile needs a lot of memory, and Android stops a visible app last.
        if (have && !Engine.ready) window.addFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        else window.clearFlags(android.view.WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        if (have && !Engine.ready && !polling) {
            polling = true
            main.postDelayed({ polling = false; if (!isDestroyed) refresh() }, 2000)
        }
    }
    private var polling = false

    private var downloadNote: String? = null

    private fun startDownload() {
        if (ModelDownloader.running) {
            startService(Intent(this, DownloadService::class.java).setAction(DownloadService.ACTION_PAUSE))
            return
        }
        // The progress notification needs this on Android 13+; the download works without it.
        if (checkSelfPermission(Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.POST_NOTIFICATIONS), 2)
        }
        modelStep.button.text = "Pause"
        modelStep.bodyView.text = "Starting…"
        startForegroundService(Intent(this, DownloadService::class.java))
    }

    private fun watchDownload() {
        ModelDownloader.listener = { done, total, error, finished ->
            main.post {
                if (!finished) {
                    modelStep.button.text = "Pause"
                    modelStep.bodyView.text = "Downloading… ${done / 1_000_000} of ${total / 1_000_000} MB"
                } else {
                    modelStep.button.text = if (error == null) "Download" else "Resume"
                    downloadNote = error
                    refresh()
                    if (error == null) Engine.ensureLoaded(this) { main.post { refresh() } }
                }
            }
        }
    }

    override fun onDestroy() {
        ModelDownloader.listener = null
        super.onDestroy()
    }


    /**
     * Debug builds only: transcribe and clean up a synthetic test WAV (16 kHz mono 16-bit) from the app's files dir.
     * Logs timings and the result so the pipeline can be checked without a microphone.
     */
    private fun selfTest(names: String) {
        Engine.ensureLoaded(this) { s -> Log.i("TellWellSelfTest", "load: $s") }
        for (name in names.split(",")) {
            val samples = readWav(File(filesDir, "selftest/$name"))
            val t0 = System.nanoTime()
            Engine.process(samples) { r ->
                val total = (System.nanoTime() - t0) / 1_000_000
                Log.i("TellWellSelfTest", "draft=${Engine.draftK} audio=${samples.size / 16}ms asr=${r.asrMs}ms cleanup=${r.cleanupMs}ms total=${total}ms")
                Log.i("TellWellSelfTest", "raw: ${r.raw}")
                Log.i("TellWellSelfTest", "out: ${r.text}")
            }
        }
    }

    private fun readWav(file: File): FloatArray {
        RandomAccessFile(file, "r").use { f ->
            val bytes = ByteArray(f.length().toInt())
            f.readFully(bytes)
            val buf = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
            var pos = 12
            while (pos + 8 <= bytes.size) {
                val id = String(bytes, pos, 4)
                val size = buf.getInt(pos + 4)
                if (id == "data") {
                    val n = minOf(size, bytes.size - pos - 8) / 2
                    return FloatArray(n) { buf.getShort(pos + 8 + it * 2) / 32768f }
                }
                pos += 8 + size + (size and 1)
            }
        }
        return FloatArray(0)
    }
}
