import io.github.ayushpatnaikgit.tellwell.HindiTwoPass
import java.io.File

// Host test for HindiTwoPass (no Android needed): its rules must give exactly the lab code's outputs
// (tellwell-lab/hindi2pass, stage 1: no model scoring) on tests/hindi_two_pass_golden.tsv (synthetic cases).
// Run: scripts/test-hindi-two-pass.sh [golden.tsv]
fun main(args: Array<String>) {
    val path = args.firstOrNull() ?: "tests/hindi_two_pass_golden.tsv"
    fun un(s: String) = s.replace("\\n", "\n").replace("\\t", "\t").replace("\\\\", "\\")
    var n = 0; var bad = 0
    for (line in File(path).readLines()) {
        if (line.isBlank()) continue
        val f = line.split("\t")
        val (kind, heard, cleaned, want) = listOf(f[0], un(f[1]), un(f[2]), un(f[3]))
        val got = when (kind) {
            "finish" -> HindiTwoPass.finish(heard, cleaned, null)
            "pieces" -> HindiTwoPass.pieces(heard).joinToString("|")
            else -> error("unknown row $kind")
        }
        n++
        if (got != want) {
            bad++
            if (bad <= 10) println("MISMATCH $kind\n  heard: $heard\n  want:  $want\n  got:   $got")
        }
    }
    println("$n rows, $bad mismatches")
    if (bad > 0) kotlin.system.exitProcess(1)
}
