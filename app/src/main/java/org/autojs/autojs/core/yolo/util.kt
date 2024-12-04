package org.autojs.autojs.core.yolo  // 确保这个包名与其他地方一致

import android.util.Log

inline fun <T> time(TAG: String, str: String = "", block: () -> T): T {
    val start = System.currentTimeMillis()
    val result = block()  // 执行传入的代码块
    val end = System.currentTimeMillis()
    Log.d(TAG, "${str}执行时间: ${end - start}ms")
    return result  // 返回代码块的结果
}
