package io.github.ayushpatnaikgit.tellwell

import android.content.Context
import android.content.res.Configuration
import android.graphics.Camera
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.graphics.drawable.GradientDrawable
import android.os.SystemClock
import android.util.TypedValue
import android.view.HapticFeedbackConstants
import android.view.MotionEvent
import android.view.View
import kotlin.math.min
import kotlin.math.sin

/** Colours from the TellWell film (light) and the website's dark mode. */
class Palette(val dark: Boolean) {
    val bg = c(if (dark) "#0e1013" else "#fbfbfd")
    val card = c(if (dark) "#16191d" else "#ffffff")
    val ink = c(if (dark) "#eef0f2" else "#1d1d1f")
    val muted = c(if (dark) "#a3abb5" else "#6e6e73")
    val faint = c(if (dark) "#737c87" else "#a1a1a6")
    val accent = c(if (dark) "#2dd4bf" else "#0b8a87")
    val accentInk = c(if (dark) "#03211f" else "#ffffff")
    val red = c(if (dark) "#ff7b6b" else "#d0453a")
    val line = if (dark) Color.argb(23, 255, 255, 255) else Color.argb(18, 0, 0, 0)
    val tint = if (dark) Color.argb(51, 45, 212, 191) else Color.argb(31, 11, 138, 135)
    val press = if (dark) Color.argb(26, 255, 255, 255) else Color.argb(15, 0, 0, 0)

    companion object {
        private fun c(hex: String) = Color.parseColor(hex)
        fun of(context: Context) = Palette(
            (context.resources.configuration.uiMode and Configuration.UI_MODE_NIGHT_MASK) == Configuration.UI_MODE_NIGHT_YES,
        )
    }
}

/** Inter, the film's typeface, at a given weight. */
object Fonts {
    private val cache = HashMap<Int, Typeface>()
    fun inter(context: Context, weight: Int): Typeface = cache.getOrPut(weight) {
        try {
            Typeface.Builder(context.assets, "fonts/Inter.ttf").setFontVariationSettings("'wght' $weight").build()
        } catch (_: Throwable) {
            if (weight >= 600) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
        }
    }
}

fun Context.dp(v: Float) = TypedValue.applyDimension(TypedValue.COMPLEX_UNIT_DIP, v, resources.displayMetrics)
fun Context.dpi(v: Int) = dp(v.toFloat()).toInt()

fun rounded(color: Int, radius: Float, stroke: Int = 0, strokeWidth: Int = 0) = GradientDrawable().apply {
    setColor(color)
    cornerRadius = radius
    if (strokeWidth > 0) setStroke(strokeWidth, stroke)
}

/**
 * The TellWell mark from the film's closing shot: seven rounded bars, the middle one the accent "cursor".
 *
 * LISTENING: the bars dance like a waveform (the film's motion), driven by the voice, turned slightly in 3D.
 * WRITING (after the tap): they turn flat and settle into the logo, and stay there.
 * IDLE: the settled mark.
 */
class MarkView(context: Context, private val p: Palette) : View(context) {
    enum class Mode { IDLE, LISTENING, WRITING }

    // The film's mark in a 264 x 280 box: left, top, height (width 26).
    private val mx = floatArrayOf(0f, 40f, 80f, 119f, 158f, 198f, 238f)
    private val mt = floatArrayOf(76f, 64f, 52f, 40f, 52f, 64f, 76f)
    private val mh = floatArrayOf(36f, 66f, 96f, 246f, 96f, 66f, 36f)
    private val curT = mt.copyOf(); private val curH = mh.copyOf()
    private val fromT = mt.copyOf(); private val fromH = mh.copyOf()
    private var fromAngle = 0f; private var curAngle = 0f
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private val r = RectF()
    private val camera = Camera()
    private val matrix = Matrix()
    private var modeStart = SystemClock.uptimeMillis()
    @Volatile private var level = 0f
    private var smooth = 0f
    /** Fixed loudness for demos (setup screen); negative means follow the microphone. */
    var demoLevel = -1f

    var mode = Mode.IDLE
        set(v) {
            if (field == v) return
            curT.copyInto(fromT); curH.copyInto(fromH); fromAngle = curAngle
            field = v; modeStart = SystemClock.uptimeMillis(); invalidate()
        }

    fun setLevel(rms: Float) { level = ((rms - 0.004f) * 10f).coerceIn(0f, 1f) }

    private fun noise(k: Int, t: Float) =
        0.5f + 0.5f * sin(t * 9.1f + k * 1.7f) * sin(t * 4.3f + k * 0.6f + sin(t * 2f + k))
    private fun ease(x: Float) = if (x < 0.5f) 4 * x * x * x else 1 - (-2 * x + 2).let { it * it * it } / 2
    private fun outC(x: Float) = 1 - (1 - x).let { it * it * it }
    private fun lerp(a: Float, b: Float, k: Float) = a + (b - a) * k

    override fun onDraw(canvas: Canvas) {
        val now = SystemClock.uptimeMillis()
        val t = (now - modeStart) / 1000f
        val g = (now % 100_000L) / 1000f
        val target = if (demoLevel >= 0f) demoLevel else level
        smooth += (target - smooth) * (if (target > smooth) 0.35f else 0.08f)
        when (mode) {
            Mode.IDLE -> { mt.copyInto(curT); mh.copyInto(curH); curAngle = 0f }
            Mode.LISTENING -> {
                val e = outC((t / 0.45f).coerceIn(0f, 1f))
                val amp = 0.2f + 0.8f * smooth
                for (k in 0 until 7) {
                    val dance = 40f + 170f * noise(k + 3, g) * amp
                    curH[k] = lerp(fromH[k], dance, e)
                    curT[k] = lerp(fromT[k], 150f - dance / 2f, e)
                }
                curAngle = lerp(fromAngle, 25f, e)
            }
            Mode.WRITING -> {
                val st = ease((t / 0.7f).coerceIn(0f, 1f))
                for (k in 0 until 7) {
                    curH[k] = lerp(fromH[k], mh[k], st)
                    curT[k] = lerp(fromT[k], mt[k], st)
                }
                curAngle = lerp(fromAngle, 0f, st)
            }
        }

        val s = min(width / 264f, height / 280f)
        val ox = (width - 264f * s) / 2f
        val oy = (height - 280f * s) / 2f
        canvas.save()
        if (curAngle != 0f) {
            camera.save(); camera.rotateY(curAngle); camera.getMatrix(matrix); camera.restore()
            matrix.preTranslate(-width / 2f, -height / 2f)
            matrix.postTranslate(width / 2f, height / 2f)
            canvas.concat(matrix)
        }
        for (k in 0 until 7) {
            paint.color = if (k == 3) p.accent else p.ink
            r.set(ox + mx[k] * s, oy + curT[k] * s, ox + (mx[k] + 26f) * s, oy + (curT[k] + curH[k]) * s)
            canvas.drawRoundRect(r, 13f * s, 13f * s, paint)
        }
        canvas.restore()
        if (mode != Mode.IDLE) postInvalidateOnAnimation()
    }
}

/** A flat icon key: just the glyph, with a soft circle while pressed. */
class IconKey(context: Context, private val p: Palette, iconRes: Int, desc: String) : View(context) {
    private val icon = context.getDrawable(iconRes)!!.mutate().apply { setTint(p.muted) }
    private val paint = Paint(Paint.ANTI_ALIAS_FLAG)
    private var down = false
    var onTap: () -> Unit = {}
    var repeatWhileHeld = false
    private val repeater = object : Runnable {
        override fun run() { onTap(); postDelayed(this, 60) }
    }

    init { contentDescription = desc; isClickable = true }

    override fun onTouchEvent(e: MotionEvent): Boolean {
        when (e.action) {
            MotionEvent.ACTION_DOWN -> {
                down = true; invalidate()
                performHapticFeedback(HapticFeedbackConstants.KEYBOARD_TAP)
                onTap()
                if (repeatWhileHeld) postDelayed(repeater, 400)
            }
            MotionEvent.ACTION_UP, MotionEvent.ACTION_CANCEL -> {
                down = false; invalidate(); removeCallbacks(repeater)
                if (e.action == MotionEvent.ACTION_UP) performClick()
            }
        }
        return true
    }

    override fun performClick(): Boolean = super.performClick()

    override fun onDraw(canvas: Canvas) {
        if (down) {
            paint.color = p.press
            canvas.drawCircle(width / 2f, height / 2f, min(width, height) / 2f, paint)
        }
        val sz = context.dpi(24)
        val cx = width / 2; val cy = height / 2
        icon.setBounds(cx - sz / 2, cy - sz / 2, cx + sz / 2, cy + sz / 2)
        icon.draw(canvas)
    }
}
