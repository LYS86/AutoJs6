package org.autojs.autojs.core.yolo
import android.util.Log

object Config {
    var conf = 0.3F // 默认置信度阈值
    var iou = 0.5F // 默认IOU阈值
    var numDetections = 0 // 预测框数量（如 8400）
    var numClasses = 0 // 类别数量（如 84 - 4 = 80）

    fun setShape(shape: IntArray) {
        Log.d("Config", "setShape: ${shape.contentToString()}")
        when {
            shape[1] == 300 && shape[2] == 6 -> { // YOLOv10
                this.numDetections = shape[1]
                // this.numClasses = 6
            }
            shape[2] in arrayOf(2100, 8400) -> { // YOLOv8, YOLOv9, YOLOv11
                this.numClasses = shape[1] - 4
                this.numDetections = shape[2]
            }
            else -> throw IllegalArgumentException("未适配输出形状: ${shape.contentToString()}")
        }
    }
}
