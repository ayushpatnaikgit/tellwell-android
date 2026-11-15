package io.github.ayushpatnaikgit.tellwell

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.concurrent.thread
import kotlin.math.sqrt

/** Records 16 kHz mono float audio into memory while running. Nothing is written to disk. */
class Recorder(private val onLevel: (Float) -> Unit) {
    private var record: AudioRecord? = null
    private val chunks = ArrayList<FloatArray>()
    @Volatile private var running = false
    private var reader: Thread? = null

    @SuppressLint("MissingPermission") // checked by the caller
    fun start(): Boolean {
        val rate = 16000
        val min = AudioRecord.getMinBufferSize(rate, AudioFormat.CHANNEL_IN_MONO, AudioFormat.ENCODING_PCM_FLOAT)
        val rec = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION, rate, AudioFormat.CHANNEL_IN_MONO,
            AudioFormat.ENCODING_PCM_FLOAT, maxOf(min, rate / 5 * 4),
        )
        if (rec.state != AudioRecord.STATE_INITIALIZED) {
            rec.release()
            return false
        }
        synchronized(chunks) { chunks.clear() }
        record = rec
        running = true
        rec.startRecording()
        reader = thread(name = "tellwell-mic") {
            val buf = FloatArray(1600)
            while (running) {
                val n = rec.read(buf, 0, buf.size, AudioRecord.READ_BLOCKING)
                if (n > 0) {
                    synchronized(chunks) { chunks.add(buf.copyOf(n)) }
                    var sum = 0f
                    for (i in 0 until n) sum += buf[i] * buf[i]
                    onLevel(sqrt(sum / n))
                }
            }
        }
        return true
    }

    /** Stops and returns everything recorded since [start]. */
    fun stop(): FloatArray {
        running = false
        reader?.join(500)
        record?.run { stop(); release() }
        record = null
        val all = synchronized(chunks) { chunks.toList().also { chunks.clear() } }
        val out = FloatArray(all.sumOf { it.size })
        var pos = 0
        for (c in all) { c.copyInto(out, pos); pos += c.size }
        return out
    }
}
