package io.github.ayushpatnaikgit.tellwell

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.provider.ContactsContract
import android.view.inputmethod.EditorInfo
import android.view.inputmethod.InputConnection

/**
 * Names that help speech recognition spell what was said: the text in the field being typed into and, only if the
 * user turns it on, contact names. Both stay on the phone (TellWell has no internet permission), are held in memory
 * only, and are never logged. The recogniser tries a contact name only if it sounds like something just heard.
 */
object Names {
    private const val PREFS = "names"
    private const val KEY_CONTACTS = "contacts"
    /** Contacts are read again after this long, so new ones are picked up. */
    private const val REFRESH_MS = 10 * 60 * 1000L
    private const val MAX_NAMES = 5000

    @Volatile private var readAt = 0L

    fun contactsAllowed(context: Context) =
        context.checkSelfPermission(Manifest.permission.READ_CONTACTS) == PackageManager.PERMISSION_GRANTED

    /** Turned on in TellWell's settings, and the permission is still granted. */
    fun contactsOn(context: Context) =
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).getBoolean(KEY_CONTACTS, false) && contactsAllowed(context)

    fun setContactsOn(context: Context, on: Boolean) {
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE).edit().putBoolean(KEY_CONTACTS, on).apply()
        readAt = 0L
    }

    /** Whether the contact names should be read again (turned on and not read recently, or just turned off). */
    fun stale(context: Context): Boolean = readAt == 0L || (contactsOn(context) && System.currentTimeMillis() - readAt > REFRESH_MS)

    /** The words of the contacts' names (first names, surnames), or nothing if contacts are off. */
    fun contactWords(context: Context): Array<String> {
        readAt = System.currentTimeMillis()
        if (!contactsOn(context)) return emptyArray()
        val words = LinkedHashSet<String>()
        try {
            context.contentResolver.query(
                ContactsContract.Contacts.CONTENT_URI,
                arrayOf(ContactsContract.Contacts.DISPLAY_NAME_PRIMARY), null, null, null,
            )?.use { c ->
                while (c.moveToNext() && words.size < MAX_NAMES) {
                    val name = c.getString(0) ?: continue
                    for (w in name.split(Regex("[^\\p{L}\\p{N}']+"))) {
                        // Names only: capitalised, at least 3 letters (initials and lowercase labels are left out).
                        if (w.length >= 3 && w[0].isUpperCase() && w.all { it.isLetter() || it == '\'' }) words += w
                    }
                }
            }
        } catch (_: SecurityException) {
            return emptyArray()
        }
        return words.toTypedArray()
    }

    /**
     * Text around the cursor, for the names in it. Nothing from fields that ask keyboards not to learn from them
     * (incognito tabs, many messaging apps' private modes). Password fields never reach this (dictation is refused).
     */
    fun fieldText(ic: InputConnection?, info: EditorInfo?): String {
        if (ic == null || info == null) return ""
        if (info.imeOptions and EditorInfo.IME_FLAG_NO_PERSONALIZED_LEARNING != 0) return ""
        val before = ic.getTextBeforeCursor(2000, 0)?.toString().orEmpty()
        val after = ic.getTextAfterCursor(500, 0)?.toString().orEmpty()
        return "$before\n$after"
    }
}
