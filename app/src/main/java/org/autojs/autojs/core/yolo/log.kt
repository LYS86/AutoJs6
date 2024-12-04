package org.autojs.autojs.core.yolo

import android.util.Log
import org.autojs.autojs.AutoJs

class AutoLog {
    private var tag = "JsLog"
    private val log = AutoJs.instance.globalConsole

    fun tag(tag: String): AutoLog {
        this.tag = tag
        return this
    }

    fun d(msg: String) {
        Log.d(tag, msg)
        log.verbose("${tag}:\n%s", msg)
    }

    fun i(msg: String) {
        Log.i(tag, msg)
        log.info("${tag}:\n%s", msg)
    }

    fun w(msg: String) {
        Log.w(tag, msg)
        log.warn("${tag}:\n%s", msg)
    }

    fun e(msg: String) {
        Log.e(tag, msg)
        log.error("${tag}:\n%s", msg)
    }
}