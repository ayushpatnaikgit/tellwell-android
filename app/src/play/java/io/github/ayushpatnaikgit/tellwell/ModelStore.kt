package io.github.ayushpatnaikgit.tellwell

import android.app.Activity
import android.content.Context
import com.google.android.play.core.assetpacks.AssetPackManager
import com.google.android.play.core.assetpacks.AssetPackManagerFactory
import com.google.android.play.core.assetpacks.AssetPackState
import com.google.android.play.core.assetpacks.AssetPackStateUpdateListener
import com.google.android.play.core.assetpacks.model.AssetPackStatus
import java.io.File
import java.lang.ref.WeakReference

/**
 * Where the model files are and how this build gets them. Play build: Google Play delivers them as asset packs, so
 * TellWell has no internet permission at all. Packs (see the models_* modules):
 *  - models_common (fast-follow, arrives right after install): decoder, joiner, tokens and the cleanup model;
 *  - models_npu (on demand): the speech encoder for Qualcomm NPUs, on Snapdragon phones;
 *  - models_cpu (on demand): the speech encoder for the CPU, on other phones or if the NPU fails.
 * Play unpacks them into folders TellWell can read but must not change; it removes them through the pack manager.
 */
object ModelStore {
    private const val COMMON = "models_common"
    private const val CPU = "models_cpu"
    private const val NPU = "models_npu"
    private val SIZES = mapOf(COMMON to 390_571_913L, CPU to 652_282_309L, NPU to 592_035_372L)

    private fun manager(c: Context): AssetPackManager = AssetPackManagerFactory.getInstance(c.applicationContext)
    private fun location(c: Context, pack: String): File? =
        manager(c).getPackLocation(pack)?.assetsPath()?.let { File(it) }
    /** A file in a pack; a path that doesn't exist if the pack isn't on the phone yet. */
    private fun file(c: Context, pack: String, name: String) =
        File(location(c, pack) ?: File(c.filesDir, "pack-not-installed/$pack"), name)

    fun decoder(c: Context) = file(c, COMMON, "decoder.int8.onnx")
    fun joiner(c: Context) = file(c, COMMON, "joiner.int8.onnx")
    fun tokens(c: Context) = file(c, COMMON, "tokens.txt")
    fun llm(c: Context) = file(c, COMMON, "s1-mini-freeflow-cleanup-q4_0.gguf")
    fun cpuEncoder(c: Context) = file(c, CPU, "encoder.int8.onnx")
    /** The NPU encoder's graph; its weights file (`.data`) sits next to it in the same pack. */
    fun npuModel(c: Context) = file(c, NPU, "enc-w8a16-1000.onnx")

    fun removeCpuEncoder(c: Context) { manager(c).removePack(CPU) }
    fun removeNpuModel(c: Context) { manager(c).removePack(NPU) }

    /** Text for the setup screen's privacy line. */
    const val SOURCE_NOTE = "What you say never leaves this phone. TellWell has no internet access: Google Play delivers its models."

    /** The packs this phone still needs: the NPU encoder on a Qualcomm phone (until compiled), else the CPU one. */
    private fun needed(c: Context): List<String> = listOfNotNull(
        COMMON.takeIf { location(c, it) == null },
        if (Npu.candidate(c)) NPU.takeIf { !Npu.compiled(c) && location(c, it) == null }
        else CPU.takeIf { location(c, it) == null },
    )

    @Volatile var busy = false
        private set
    fun missingBytes(c: Context): Long = needed(c).sumOf { SIZES.getValue(it) }

    /** Progress for the setup screen: (bytes done, total, error, finished). */
    @Volatile var listener: ((Long, Long, String?, Boolean) -> Unit)? = null

    private var requested: List<String> = emptyList()
    private val states = HashMap<String, AssetPackState>()
    private var activity: WeakReference<Activity>? = null
    private var registered = false
    /** Play's mobile-data question is shown once per request, not on every progress update. */
    private var confirmShown = false

    private val update = AssetPackStateUpdateListener { s ->
        // Pack name, status and error code only (no user content), to diagnose delivery.
        android.util.Log.i("TellWellPacks", "${s.name()} status=${s.status()} error=${s.errorCode()} ${s.bytesDownloaded()}/${s.totalBytesToDownload()}")
        if (s.name() !in requested) return@AssetPackStateUpdateListener
        states[s.name()] = s
        when (s.status()) {
            // Large packs can wait for Wi-Fi or for the user's OK to use mobile data; Play shows that question.
            AssetPackStatus.WAITING_FOR_WIFI, AssetPackStatus.REQUIRES_USER_CONFIRMATION ->
                if (!confirmShown) activity?.get()?.let { a ->
                    if (!a.isFinishing) { confirmShown = true; manager(a).showConfirmationDialog(a) }
                }
            AssetPackStatus.FAILED -> finish("Google Play couldn't deliver the models (error ${s.errorCode()}). Try again.")
            AssetPackStatus.CANCELED -> finish("The download was cancelled. Try again.")
        }
        if (requested.all { states[it]?.status() == AssetPackStatus.COMPLETED }) { finish(null); return@AssetPackStateUpdateListener }
        if (busy) listener?.invoke(states.values.sumOf { it.bytesDownloaded() },
            requested.sumOf { states[it]?.totalBytesToDownload()?.takeIf { t -> t > 0 } ?: SIZES.getValue(it) }, null, false)
    }

    private fun finish(error: String?) {
        if (!busy) return
        busy = false
        listener?.invoke(0, 0, error, true)
    }

    /** Asks Google Play for the packs this phone still needs. */
    fun start(activity: Activity) {
        if (busy) return
        val m0 = manager(activity)
        android.util.Log.i("TellWellPacks", "locations: " + listOf(COMMON, CPU, NPU).joinToString { "$it=${m0.getPackLocation(it) != null}" })
        val packs = needed(activity)
        if (packs.isEmpty()) { listener?.invoke(0, 0, null, true); return }
        this.activity = WeakReference(activity)
        val m = manager(activity)
        if (!registered) { m.registerListener(update); registered = true }
        requested = packs
        states.clear()
        confirmShown = false
        busy = true
        // Packs already delivered (fast-follow often is) may send no further update: use the states the fetch returns.
        m.fetch(packs).addOnSuccessListener { st -> st.packStates().values.forEach { update.onStateUpdate(it) } }
            .addOnFailureListener { e ->
                val code = (e as? com.google.android.play.core.assetpacks.AssetPackException)?.errorCode
                android.util.Log.w("TellWellPacks", "fetch $packs failed: ${e.javaClass.simpleName} code=$code")
                finish("Couldn't reach Google Play (error ${code ?: e.javaClass.simpleName}). Try again.")
            }
    }

    const val CAN_PAUSE = false
    /**
     * Setup asks Play for the missing packs as soon as it opens, so there is nothing to tap. Only the shared pack comes
     * with the install itself (fast-follow): shipping both speech encoders at install would cost every phone about
     * 600 MB it never uses, so the one this phone needs is requested here.
     */
    const val AUTO_FETCH = true
    const val STEP_TITLE = "Get the models"
    fun missingNote(mb: Long) = "Google Play delivers them: about ${(mb + 50) / 100 * 100} MB, once."
    fun upgradeNote(mb: Long) = "Faster speech for this phone's NPU: Google Play delivers about $mb MB, once."
}
