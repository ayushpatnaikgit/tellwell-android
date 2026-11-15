package io.github.ayushpatnaikgit.tellwell

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.inputmethodservice.InputMethodService
import android.os.Handler
import android.os.Looper
import android.text.InputType
import android.view.Gravity
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import android.view.WindowInsets
import android.view.inputmethod.EditorInfo
import android.widget.LinearLayout
import android.widget.TextView

/**
 * A voice-only keyboard. Switching to it (Gboard's globe / keyboard key) starts listening straight away.
 * Tap the mark to finish: TellWell types the cleaned-up text and hands the field back to the previous
 * keyboard. Holding works as push-to-talk. The keyboard key cancels.
 *
 * Privacy: the microphone is open only while TellWell is on screen and listening (at most two minutes), audio stays in memory,
 * nothing is stored or logged, and the app has no network permission. Password fields are refused.
 */
class TellWellIME : InputMethodService() {
    private val main = Handler(Looper.getMainLooper())
    private var recorder: Recorder? = null
    private var busy = false
    private lateinit var p: Palette
    private lateinit var label: TextView
    private lateinit var mark: MarkView
    private var refused = false
    private var shown = false
    private var downAt = 0L
    private var stopOnUp = false
    private val maxListen = Runnable { if (recorder != null) finish() }

    @SuppressLint("ClickableViewAccessibility")
    override fun onCreateInputView(): View {
        p = Palette.of(this)
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setBackgroundColor(p.bg)
            setOnApplyWindowInsetsListener { v, insets ->
                val nav = insets.getInsets(WindowInsets.Type.navigationBars()).bottom
                v.setPadding(dpi(8), dpi(14), dpi(8), dpi(4) + nav)
                insets
            }
        }
        label = TextView(this).apply {
            textSize = 14f
            typeface = Fonts.inter(this@TellWellIME, 500)
            setTextColor(p.muted)
            gravity = Gravity.CENTER
        }
        root.addView(label)

        mark = MarkView(this, p).apply {
            contentDescription = "Tap to finish"
            setOnTouchListener { v, e ->
                if (!v.isEnabled) return@setOnTouchListener true
                when (e.action) {
                    MotionEvent.ACTION_DOWN -> { v.performHapticFeedback(HapticFeedbackConstants.VIRTUAL_KEY); onDown() }
                    MotionEvent.ACTION_UP -> { onUp(); v.performClick() }
                    MotionEvent.ACTION_CANCEL -> onUp()
                }
                true
            }
        }
        root.addView(mark, LinearLayout.LayoutParams(LinearLayout.LayoutParams.MATCH_PARENT, dpi(170)))

        val row = LinearLayout(this).apply { orientation = LinearLayout.HORIZONTAL; gravity = Gravity.CENTER_VERTICAL }
        val key = dpi(52)
        row.addView(IconKey(this, p, R.drawable.ic_keyboard, "Cancel and go back to keyboard").apply { onTap = { cancel(); switchBack() } },
            LinearLayout.LayoutParams(key, key))
        row.addView(View(this), LinearLayout.LayoutParams(0, 1, 1f))
        row.addView(IconKey(this, p, R.drawable.ic_backspace, "Delete").apply {
            repeatWhileHeld = true
            onTap = { currentInputConnection?.deleteSurroundingText(1, 0) }
        }, LinearLayout.LayoutParams(key, key))
        row.addView(IconKey(this, p, R.drawable.ic_return, "Enter").apply {
            onTap = { if (!sendDefaultEditorAction(true)) currentInputConnection?.commitText("\n", 1) }
        }, LinearLayout.LayoutParams(key, key))
        root.addView(row)
        return root
    }

    override fun onStartInputView(info: EditorInfo, restarting: Boolean) {
        super.onStartInputView(info, restarting)
        shown = true
        refused = isPassword(info)
        when {
            refused -> setLabel("Not in password fields")
            !hasMic() -> setLabel("Open TellWell to allow the mic")
            !Engine.modelsPresent(this) -> setLabel("Models missing")
            else -> {
                if (!Engine.ready) Engine.ensureLoaded(this)
                setLabel("Tap to talk")
            }
        }
        mark.isEnabled = !refused
        mark.alpha = if (refused) 0.4f else 1f
        // Opening TellWell means "I want to talk": start listening at once.
        if (!refused && !restarting && !busy && recorder == null) startListening()
    }

    override fun onFinishInputView(finishingInput: Boolean) {
        // Never keep the microphone open after the keyboard goes away.
        shown = false
        cancel()
        super.onFinishInputView(finishingInput)
    }

    private fun isPassword(info: EditorInfo): Boolean {
        val cls = info.inputType and InputType.TYPE_MASK_CLASS
        val variation = info.inputType and InputType.TYPE_MASK_VARIATION
        return (cls == InputType.TYPE_CLASS_TEXT && variation in setOf(
            InputType.TYPE_TEXT_VARIATION_PASSWORD,
            InputType.TYPE_TEXT_VARIATION_VISIBLE_PASSWORD,
            InputType.TYPE_TEXT_VARIATION_WEB_PASSWORD,
        )) || (cls == InputType.TYPE_CLASS_NUMBER && variation == InputType.TYPE_NUMBER_VARIATION_PASSWORD)
    }

    private fun hasMic() = checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun setLabel(s: String) { label.text = s }

    private fun onDown() {
        downAt = System.currentTimeMillis()
        stopOnUp = recorder != null       // a tap while listening finishes
        if (recorder == null && !busy) startListening()
    }

    private fun onUp() {
        if (recorder == null) return
        val held = System.currentTimeMillis() - downAt
        // Tap while listening, or press-and-hold (push-to-talk), finishes. A short tap that started it keeps listening.
        if (stopOnUp || held > 600) finish() else setLabel("Listening… tap to finish")
    }

    private fun startListening() {
        if (refused || busy || recorder != null) return
        if (!hasMic()) { setLabel("Open TellWell to allow the mic"); return }
        if (!Engine.modelsPresent(this)) { setLabel("Models missing"); return }
        // Record straight away even if the models are still loading. The engine's single worker
        // finishes loading first, then processes this clip.
        if (!Engine.ready) Engine.ensureLoaded(this)
        val r = Recorder { level -> main.post { mark.setLevel(level) } }
        if (!r.start()) { setLabel("Couldn't open the mic"); return }
        recorder = r
        mark.mode = MarkView.Mode.LISTENING
        setLabel("Listening… tap to finish")
        main.postDelayed(maxListen, 120_000)
    }

    /** Stops listening and throws the audio away. */
    private fun cancel() {
        main.removeCallbacks(maxListen)
        recorder?.stop()
        recorder = null
        if (::mark.isInitialized) {
            if (!busy) mark.mode = MarkView.Mode.IDLE
        }
    }

    private fun finish() {
        val r = recorder ?: return
        recorder = null
        main.removeCallbacks(maxListen)
        val audio = r.stop()
        busy = true
        mark.isEnabled = false
        mark.mode = MarkView.Mode.WRITING
        mark.invalidate()
        setLabel(if (Engine.ready) "Writing…" else "Loading, then writing…")
        Engine.process(audio) { res ->
            main.post {
                busy = false
                mark.isEnabled = true
                mark.mode = MarkView.Mode.IDLE
                mark.invalidate()
                if (!shown) return@post   // the keyboard was closed meanwhile: type nothing
                if (res.text.isEmpty()) { setLabel("Didn't catch that. Tap to try again"); return@post }
                insert(res.text)
                setLabel("Tap to talk")
                switchBack()
            }
        }
    }

    private fun insert(text: String) {
        val ic = currentInputConnection ?: return
        val before = ic.getTextBeforeCursor(1, 0)?.toString().orEmpty()
        val spaced = if (before.isNotEmpty() && !before[0].isWhitespace()) " $text" else text
        ic.commitText(spaced, 1)
    }

    private fun switchBack() {
        if (!switchToPreviousInputMethod()) requestHideSelf(0)
    }
}
