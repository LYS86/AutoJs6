package org.autojs.autojs.core.yolo

import android.graphics.Bitmap
import org.autojs.autojs.AutoJs
import org.autojs.autojs.core.image.ImageWrapper
import org.mozilla.javascript.NativeObject
import org.tensorflow.lite.DataType
import org.tensorflow.lite.InterpreterApi
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.MappedByteBuffer
import java.util.concurrent.locks.ReentrantLock

class Detection : AutoCloseable {
    private val TAG = "TFLiteModel"
    private val log = AutoLog().tag(TAG)
    private val jsLog = AutoJs.instance.globalConsole

    // 线程保护
    private val threadLock = ReentrantLock()

    // TFLite组件
    private lateinit var interpreter: InterpreterApi
    private lateinit var metadataExtractor: MetadataExtractor
    private var gpuDelegate: GpuDelegate? = null
    private lateinit var outputBuffer: TensorBuffer

    // 图像处理
    private var inputWidth = 640
    private var inputHeight = 640
    private lateinit var imageProcessor: ImageProcessor
    private lateinit var tensorImage: TensorImage

    // 模型数据
    private var labels: List<String> = emptyList()
    private lateinit var options: InterpreterApi.Options

    init {
        initGpu()
    }

    fun init(options: NativeObject) {
        threadLock.lock()
        try {
            val model = options.get("model") as? String
            val labels = options.get("labels") as? String
            val isGPU = options.get("gpu") as? Boolean ?: false
            log.i("init model: $model, labels: $labels, gpu: $isGPU")
            if (model.isNullOrBlank()) {
                throw IllegalArgumentException("模型路径不能为空")
            }
            loadModel(model, isGPU)
            loadLabels(labels)
        } finally {
            threadLock.unlock()
        }
    }

    /**
     * 在检测结果上绘制边界框
     * @param bitmap 输入图像
     * @param results 检测结果
     * @return 绘制了边界框的图像
     */
    fun drawBoxes(bitmap: Any, results: Array<Result>): Any {
        return when (bitmap) {
            is Bitmap -> FileUtil.drawBoxes(bitmap, results)
            is ImageWrapper -> ImageWrapper(FileUtil.drawBoxes(bitmap.bitmap, results))
            else -> throw IllegalArgumentException("不支持的图像类型: ${bitmap.javaClass}")
        }
    }

    /**
     * 设置检测阈值
     * @param conf 置信度阈值 (0-1)
     * @param iou IoU阈值 (0-1)
     */
    fun setThresholds(conf: Float? = null, iou: Float? = null) {
        conf?.let {
            require(it in 0.0F..1.0F) { "置信度阈值必须在0到1之间" }
            Config.conf = it
        }
        iou?.let {
            require(it in 0.0F..1.0F) { "IoU阈值必须在0到1之间" }
            Config.iou = it
        }
    }

    /**
     * 重置检测阈值为默认值
     */
    fun resetThresholds() {
        Config.conf = 0.3F
        Config.iou = 0.7F
    }

    /**
     * 运行目标检测
     * @param input 输入图像 (Bitmap或ImageWrapper)
     * @return 检测结果数组
     */
    fun run(input: Any?): Array<Result> {
        threadLock.lock()
        try {
            val bitmap = when (input) {
                is Bitmap -> input
                is ImageWrapper -> input.bitmap
                else -> throw IllegalArgumentException("不支持的图像类型: ${input?.javaClass}")
            }

            val (tensorImage, resizeInfo) = preprocessImage(bitmap)
            val outputBuffer = runInference(tensorImage)
            val results = Output.parseOutput(outputBuffer.floatArray, labels)

            val scale = resizeInfo[0]
            val offsetX = resizeInfo[1]
            val offsetY = resizeInfo[2]

            // 修正坐标
            results.forEach { result ->
                result.rect.apply {
                    left = ((left * inputWidth - offsetX) / scale).coerceIn(0f, bitmap.width.toFloat())
                    top = ((top * inputHeight - offsetY) / scale).coerceIn(0f, bitmap.height.toFloat())
                    right = ((right * inputWidth - offsetX) / scale).coerceIn(0f, bitmap.width.toFloat())
                    bottom = ((bottom * inputHeight - offsetY) / scale).coerceIn(0f, bitmap.height.toFloat())
                }
            }

            return results
        } finally {
            threadLock.unlock()
        }
    }

    /**
     * 获取最后一次推理的运行时间（毫秒）
     */
    fun runtime(): Long {
        return interpreter.lastNativeInferenceDurationNanoseconds / 1_000_000
    }

    /**
     * 加载模型
     * @param path 模型文件路径
     */
    fun loadModel(path: String) {
        loadModel(path, true)
    }

    /**
     * 加载模型
     * @param path 模型文件路径
     * @param isGPU 是否使用GPU
     */
    fun loadModel(path: String, isGPU: Boolean) {
        options = getOptions(isGPU)
        loadModel(path, options)
    }

    /**
     * 加载模型（使用指定选项）
     */
    fun loadModel(path: String, options: InterpreterApi.Options) {
        threadLock.lock()
        try {
            val modelBuffer: MappedByteBuffer = FileUtil.loadModel(path)
            metadataExtractor = MetadataExtractor(modelBuffer)
            interpreter = InterpreterApi.create(modelBuffer, options)
            initProcessors()
            loadLabels(null)
        } catch (e: Exception) {
            throw RuntimeException("${e.message}")
        } finally {
            threadLock.unlock()
        }
    }

    /**
     * 获取模型元数据
     */
    fun getMetadata(): String {
        if (!hasMetadata()) return ""
        return metadataExtractor.associatedFileNames.firstOrNull()?.let { getAssociatedFile(it) } ?: ""
    }

    /**
     * 加载标签
     */
    fun loadLabels(path: String?) {
        if (!path.isNullOrBlank()) {
            labels = FileUtil.fromFile(path)
            return
        }
        val info = getMetadata()
        if (info.isBlank()) return
        labels = FileUtil.fromJson(info)

    }

    /**
     * 获取标签列表
     */
    fun getLabels(): Array<String> {
        return labels.toTypedArray()
    }

    override fun close() {
        threadLock.lock()
        try {
            if (::interpreter.isInitialized) {
                interpreter.close()
            }
            gpuDelegate?.close()
            gpuDelegate = null
        } finally {
            threadLock.unlock()
        }
    }

    /**
     * 缩放图像并保持宽高比，同时填充以匹配目标尺寸。
     *
     * @param bitmap 原始图像
     * @param targetWidth 模型期望的宽度
     * @param targetHeight 模型期望的高度
     * @return 经过缩放和填充后的图像，尺寸严格匹配 targetWidth 和 targetHeight
     */
    private fun resizeAndPadBitmap(bitmap: Bitmap, targetWidth: Int, targetHeight: Int): Pair<Bitmap, FloatArray> {
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height
        val scale = minOf(targetWidth.toFloat() / originalWidth, targetHeight.toFloat() / originalHeight)
        val scaledWidth = (originalWidth * scale).toInt()
        val scaledHeight = (originalHeight * scale).toInt()
        val offsetX = (targetWidth - scaledWidth) / 2f
        val offsetY = (targetHeight - scaledHeight) / 2f
        val outputBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
        android.graphics.Canvas(outputBitmap).apply {
            drawBitmap(bitmap, null, android.graphics.RectF(offsetX, offsetY, offsetX + scaledWidth, offsetY + scaledHeight), null)
        }
        return Pair(outputBitmap, floatArrayOf(scale, offsetX, offsetY))
    }


    private fun initGpu() {
        CompatibilityList().use { compatibilityList ->
            if (compatibilityList.isDelegateSupportedOnThisDevice) {
                // val delegateOptions = compatibilityList.bestOptionsForThisDevice
                // gpuDelegate = GpuDelegate(delegateOptions)
                // gpuDelegate = GpuDelegateFactory()
                gpuDelegate = GpuDelegate()
            }
        }
    }

    private fun getOptions(isGPU: Boolean): InterpreterApi.Options {
        val opts = InterpreterApi.Options().apply {
            useNNAPI = true
        }
        if (!isGPU) return opts
        if (gpuDelegate == null) {
            jsLog.error("GPU不支持")
            return opts
        }
        return opts.apply { addDelegate(gpuDelegate) }
    }

    private fun initProcessors() {
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()
        Config.setShape(outputShape)
        inputWidth = inputShape[1]
        inputHeight = inputShape[2]
        outputBuffer = TensorBuffer.createFixedSize(outputShape, DataType.FLOAT32)
        imageProcessor = ImageProcessor.Builder().add(NormalizeOp(0f, 255f)).build()
        tensorImage = TensorImage(DataType.FLOAT32)
    }

    private fun preprocessImage(bitmap: Bitmap): Pair<TensorImage, FloatArray> {
        val (resizedBitmap, resizeInfo) = resizeAndPadBitmap(bitmap, inputWidth, inputHeight)
        tensorImage.load(resizedBitmap)
        return Pair(imageProcessor.process(tensorImage), resizeInfo)
    }

    private fun runInference(tensorImage: TensorImage): TensorBuffer {
        interpreter.run(tensorImage.buffer, outputBuffer.buffer)
        return outputBuffer
    }

    private fun hasMetadata(): Boolean = metadataExtractor.hasMetadata()

    private fun getAssociatedFile(fileName: String): String {
        return metadataExtractor.getAssociatedFile(fileName).use { stream ->
            BufferedReader(InputStreamReader(stream)).use { it.readText() }
        }
    }
}
