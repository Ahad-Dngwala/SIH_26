package org.sih26.deadreckoning

import android.app.Application

/** No custom initialisation needed yet. Exists as a named class (rather than the
 * default android.app.Application) because the manifest already references it and
 * swapping in shared state later - a crash reporter, a settings store - should not
 * require touching the manifest again. */
class DeadReckoningApp : Application()
