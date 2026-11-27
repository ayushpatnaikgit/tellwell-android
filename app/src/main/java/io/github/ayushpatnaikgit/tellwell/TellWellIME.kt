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
    /** English / हिंदी: shown when more than one language is ready. */
    private lateinit var langSwitch: LinearLayout
    private val langButtons = HashMap<Lang, TextView>()
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
        // Between the keyboard key and delete: the language switch, when there is a choice.
        val middle = LinearLayout(this).apply { gravity = Gravity.CENTER }
        langSwitch = LinearLayout(this).apply {
            orientation = LinearLayout.HORIZONTAL
            setPadding(dpi(3), dpi(3), dpi(3), dpi(3))
            background = rounded(p.press, dp(23f))
            contentDescription = "Dictation language"
        }
        for (l in Lang.entries) {
            val b = TextView(this).apply {
                text = l.label
                textSize = 15f
                typeface = Fonts.inter(this@TellWellIME, 600)
                gravity = Gravity.CENTER
                minWidth = dpi(84)
                setPadding(dpi(16), 0, dpi(16), 0)
                contentDescription = "Dictate in ${l.englishName}"
                setOnClickListener { pickLanguage(l) }
            }
            langButtons[l] = b
            langSwitch.addView(b, LinearLayout.LayoutParams(LinearLayout.LayoutParams.WRAP_CONTENT, dpi(40)))
        }
        middle.addView(langSwitch)
        row.addView(middle, LinearLayout.LayoutParams(0, LinearLayout.LayoutParams.WRAP_CONTENT, 1f))
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
                Engine.refreshNames(this)
                if (Engine.preparing(this)) showPreparing() else setLabel("Tap to talk")
            }
        }
        mark.isEnabled = !refused
        mark.alpha = if (refused) 0.4f else 1f
        showLanguages()
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

    /** The switch shows only when more than one language is ready; the chosen one is filled in. */
    private fun showLanguages() {
        val ready = Languages.ready(this)
        langSwitch.visibility = if (ready.size > 1) View.VISIBLE else View.GONE
        val active = Languages.active(this)
        for ((l, b) in langButtons) {
            val on = l == active
            b.visibility = if (l in ready) View.VISIBLE else View.GONE
            b.background = if (on) rounded(p.accent, dp(20f)) else null
            b.setTextColor(if (on) p.accentInk else p.muted)
            b.isSelected = on
        }
    }

    /** Takes effect for the dictation in progress too: recognition runs when it ends. */
    private fun pickLanguage(l: Lang) {
        Languages.setActive(this, l)
        showLanguages()
        // Load the chosen language's models now, so they are likely ready when the dictation ends.
        Engine.prepare(this, l)
        if (recorder != null) setLabel(listeningLabel())
    }

    /** "Listening in Hindi…" when there is a choice, so a wrong one is noticed before finishing. */
    private fun listeningLabel(): String =
        if (Languages.ready(this).size > 1) "Listening in ${Languages.active(this).englishName}… tap to finish"
        else "Listening… tap to finish"

    /** Shown while the speech model compiles; checks again every 2 s and offers "Tap to talk" when done. */
    private val preparingCheck = object : Runnable {
        override fun run() {
            if (!shown) return
            if (Engine.preparing(this@TellWellIME)) { main.postDelayed(this, 2000); return }
            setLabel("Ready. Tap to talk")
        }
    }

    private fun showPreparing() {
        setLabel("Getting ready for this phone, about a minute (once)")
        main.removeCallbacks(preparingCheck)
        main.postDelayed(preparingCheck, 2000)
    }

    private fun onDown() {
        downAt = System.currentTimeMillis()
        stopOnUp = recorder != null       // a tap while listening finishes
        if (recorder == null && !busy) startListening()
    }

    private fun onUp() {
        if (recorder == null) return
        val held = System.currentTimeMillis() - downAt
        // Tap while listening, or press-and-hold (push-to-talk), finishes. A short tap that started it keeps listening.
        if (stopOnUp || held > 600) finish() else setLabel(listeningLabel())
    }

    private fun startListening() {
        if (refused || busy || recorder != null) return
        if (!hasMic()) { setLabel("Open TellWell to allow the mic"); return }
        if (!Engine.modelsPresent(this)) { setLabel("Models missing"); return }
        // The first time on an NPU phone, the speech model compiles for about a minute. Don't record a clip that would
        // only wait: say what's happening, and listen once it's done.
        if (Engine.preparing(this)) { showPreparing(); return }
        // Record straight away even if the models are still loading. The engine's single worker
        // finishes loading first, then processes this clip.
        if (!Engine.ready) Engine.ensureLoaded(this)
        val r = Recorder { level -> main.post { mark.setLevel(level) } }
        if (!r.start()) { setLabel("Couldn't open the mic"); return }
        recorder = r
        mark.mode = MarkView.Mode.LISTENING
        setLabel(listeningLabel())
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
        // Names in the field being typed into help spell what was said (memory only, never stored).
        val context = Names.fieldText(currentInputConnection, currentInputEditorInfo)
        busy = true
        mark.isEnabled = false
        mark.mode = MarkView.Mode.WRITING
        mark.invalidate()
        setLabel(if (Engine.ready) "Writing…" else "Loading, then writing…")
        Engine.process(audio, context, Languages.active(this)) { res ->
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
