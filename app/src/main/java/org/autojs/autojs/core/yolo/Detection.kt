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
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.metadata.MetadataExtractor
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.BufferedReader
import java.io.InputStreamReader
import java.nio.MappedByteBuffer

/**
 * 此类是线程安全的，但必须在同一线程中使用
 */
class Detection : AutoCloseable {
    private val TAG = "TFLiteModel"
    private val log = AutoLog().tag(TAG)
    private val jsLog = AutoJs.instance.globalConsole

    // 线程保护
    @Volatile
    private var ownerThread: Thread? = null
    private val threadLock = Object()

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
    private var options: InterpreterApi.Options = InterpreterApi.Options()

    init {
        initGpu()
    }

    fun init(options: NativeObject) {
        checkThread()
        val model = options.get("model") as? String
        val labels = options.get("labels") as? String
        val isGPU = options.get("gpu") as? Boolean ?: false
        log.i("init model: $model, labels: $labels, gpu: $isGPU")

        if (model.isNullOrBlank()) {
            throw IllegalArgumentException("模型路径不能为空")
        }
        loadModel(model, isGPU)
        loadLabels(labels)


    }


    /**
     * 在检测结果上绘制边界框
     * @param bitmap 输入图像
     * @param results 检测结果
     * @return 绘制了边界框的图像
     */
    @Synchronized
    fun drawBoxes(bitmap: Any, results: Array<Result>): Any {
        checkThread()
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
    @Synchronized
    fun setThresholds(conf: Float? = null, iou: Float? = null) {
        checkThread()
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
    @Synchronized
    fun resetThresholds() {
        checkThread()
        Config.conf = 0.3F
        Config.iou = 0.7F
    }

    /**
     * 运行目标检测
     * @param input 输入图像 (Bitmap或ImageWrapper)
     * @return 检测结果数组
     */
    @Synchronized
    fun run(input: Any?): Array<Result> {
        checkThread()

        val bitmap = when (input) {
            is Bitmap -> input
            is ImageWrapper -> input.bitmap
            else -> throw IllegalArgumentException("不支持的图像类型: ${input?.javaClass}")
        }

        val tensorImage = preprocessImage(bitmap)
        val outputBuffer = runInference(tensorImage)
        val results = Output.parseOutput(outputBuffer.floatArray, labels)

        val width = bitmap.width
        val height = bitmap.height
        results.forEach { result ->
            result.rect.apply {
                left *= width
                top *= height
                right *= width
                bottom *= height
            }
        }

        return results
    }

    /**
     * 获取最后一次推理的运行时间（毫秒）
     */
    @Synchronized
    fun runtime(): Long {
        checkThread()
        return interpreter.lastNativeInferenceDurationNanoseconds / 1_000_000
    }

    /**
     * 加载模型
     * @param path 模型文件路径
     */
    @Synchronized
    fun loadModel(path: String) {
        loadModel(path, true)
    }

    /**
     * 加载模型
     * @param path 模型文件路径
     * @param isGPU 是否使用GPU
     */
    @Synchronized
    fun loadModel(path: String, isGPU: Boolean) {
        options = getOptions(isGPU)
        loadModel(path, options)
    }

    /**
     * 加载模型（使用指定选项）
     */
    @Synchronized
    fun loadModel(path: String, options: InterpreterApi.Options) {
        checkThread()
        try {
            val modelBuffer: MappedByteBuffer = FileUtil.loadModel(path)
            metadataExtractor = MetadataExtractor(modelBuffer)
            interpreter = InterpreterApi.create(modelBuffer, options)
            initProcessors()
            loadLabels(null)
        } catch (e: Exception) {
            throw RuntimeException("${e.message}")
        }
    }

    /**
     * 获取模型元数据
     */
    @Synchronized
    fun getMetadata(): String {
        checkThread()
        if (!hasMetadata()) return ""
        return metadataExtractor.associatedFileNames.firstOrNull()?.let { getAssociatedFile(it) } ?: ""
    }


    /**
     * 加载标签
     */
    @Synchronized
    fun loadLabels(path: String?) {
        checkThread()
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
    @Synchronized
    fun getLabels(): Array<String> {
        checkThread()
        return labels.toTypedArray()
    }

    @Synchronized
    override fun close() {
        checkThread()
        if (::interpreter.isInitialized) {
            interpreter.close()
        }
        gpuDelegate?.close()
        gpuDelegate = null
        synchronized(threadLock) {
            ownerThread = null
        }
    }

    /**
     * 确保在同一线程中使用
     * @throws IllegalStateException 如果在不同线程中使用
     */
    private fun checkThread() {
        synchronized(threadLock) {
            val currentThread = Thread.currentThread()
            if (ownerThread == null) {
                ownerThread = currentThread
            } else if (ownerThread != currentThread) {
                throw IllegalStateException("不允许多线程调用")
            }
        }
    }


    private fun initGpu() {
        CompatibilityList().use { compatibilityList ->
            if (compatibilityList.isDelegateSupportedOnThisDevice) {
                val delegateOptions = compatibilityList.bestOptionsForThisDevice
                gpuDelegate = GpuDelegate(delegateOptions)
            }
        }

    }

    private fun getOptions(isGPU: Boolean): InterpreterApi.Options {
        val opts = InterpreterApi.Options()
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

        imageProcessor = ImageProcessor.Builder().add(ResizeOp(inputHeight, inputWidth, ResizeOp.ResizeMethod.BILINEAR)).add(NormalizeOp(0f, 255f)).build()

        tensorImage = TensorImage(DataType.FLOAT32)
    }

    private fun preprocessImage(bitmap: Bitmap): TensorImage {
        tensorImage.load(bitmap)
        return imageProcessor.process(tensorImage)
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
