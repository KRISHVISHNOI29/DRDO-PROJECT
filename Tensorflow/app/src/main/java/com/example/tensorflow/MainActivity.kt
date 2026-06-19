package com.example.tensorflow

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.speech.tts.TextToSpeech
import android.text.format.Formatter
import android.util.Log
import android.view.WindowManager
import android.widget.Button
import android.widget.EditText
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.BufferedInputStream
import java.io.ByteArrayOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel
import java.net.HttpURLConnection
import java.net.Inet4Address
import java.net.URL
import java.util.Locale
import android.net.wifi.WifiManager
import java.net.DatagramPacket
import java.net.DatagramSocket
import java.net.InetAddress

class MainActivity : AppCompatActivity() {

    private lateinit var ipEditText: EditText
    private lateinit var startButton: Button
    private lateinit var frameImageView: ImageView
    private lateinit var outputText: TextView
    private val mainHandler = Handler(Looper.getMainLooper())

    @Volatile
    private var streaming = false

    // UDP Discovery
    private val UDP_PORT = 4210
    private var udpSocket: DatagramSocket? = null
    @Volatile
    private var listeningUDP = true

    // Model & Stream dimensions
    private val STREAM_WIDTH = 320
    private val STREAM_HEIGHT = 240
    private val MODEL_WIDTH = 320
    private val MODEL_HEIGHT = 320

    // YOLOv8n default (adjust if you have v8s/m/l!)
    private val NUM_CHANNELS = 84
    private val NUM_ELEMENTS = 2100

    // For robust outputs, use higher threshold
    private val CONFIDENCE_THRESHOLD = 0.5f
    private val IOU_THRESHOLD = 0.4f

    private val INFERENCE_EVERY_N_FRAMES = 3
    private var frameCounter = 0

    // TFLite interpreter & buffers
    private lateinit var tfLiteModel: Interpreter
    private lateinit var labels: List<String>
    private lateinit var inputBuffer: ByteBuffer
    private lateinit var outputBuffer: ByteBuffer

    private val INPUT_SIZE_BYTES = MODEL_WIDTH * MODEL_HEIGHT * 3 * 4
    private val OUTPUT_SIZE_BYTES = NUM_CHANNELS * NUM_ELEMENTS * 4

    // FPS
    private var displayedFrameCount = 0
    private var lastFPSCalcTime = System.currentTimeMillis()
    private var currentFPS = 0.0

    // Inference & TTS tracking
    private var lastInferenceTime = 0L
    private var lastDetection: Detection? = null

    private lateinit var tts: TextToSpeech
    private var lastSpokenTime = 0L
    private var ttsDelayMs = 2000L
    private var lastAnnouncedClassId: Int? = null
    private var lastAnnouncedConfidence: Float = -1f

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        // Keep the screen on while the app is running
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        ipEditText = findViewById(R.id.ipEditText)
        startButton = findViewById(R.id.startButton)
        frameImageView = findViewById(R.id.frameImageView)
        outputText = findViewById(R.id.outputText)

        loadModelAndLabels()
        initBuffers()

        tts = TextToSpeech(this) { status ->
            if (status == TextToSpeech.SUCCESS) {
                tts.language = Locale.UK
            }
        }

        startUdpDiscovery()

        startButton.setOnClickListener {
            if (!streaming) {
                val ip = ipEditText.text.toString().trim()
                if (ip.isEmpty()) {
                    Toast.makeText(this, "Enter ESP32 IP", Toast.LENGTH_SHORT).show()
                    return@setOnClickListener
                }
                streaming = true
                startButton.text = "Stop Stream"
                startStream(ip)
            } else {
                streaming = false
                startButton.text = "Start Stream Inference"
            }
        }
    }

    // Listen for ESP32 UDP broadcast: "ESP32CAM_IP:192.168.x.x"
    private fun startUdpDiscovery() {
        listeningUDP = true
        Thread {
            try {
                // Use a socket that can receive broadcasts
                udpSocket = DatagramSocket(UDP_PORT).apply {
                    broadcast = true
                    reuseAddress = true
                }
                val buffer = ByteArray(1024)
                Log.d("UDP", "Listening on port $UDP_PORT...")
                
                while (listeningUDP) {
                    val packet = DatagramPacket(buffer, buffer.size)
                    udpSocket?.receive(packet)
                    val message = String(packet.data, 0, packet.length)
                    Log.d("UDP", "Received: $message")
                    
                    if (message.startsWith("ESP32CAM_IP:")) {
                        val ip = message.substringAfter("ESP32CAM_IP:").trim()
                        mainHandler.post {
                            // Only update if empty or already contains a discovered IP (keep port :81)
                            val currentText = ipEditText.text.toString()
                            if (currentText.isEmpty() || currentText.contains(":81")) {
                                ipEditText.setText("$ip:81")
                                if (!currentText.contains(ip)) {
                                    Toast.makeText(this, "ESP32 Discovered: $ip", Toast.LENGTH_SHORT).show()
                                }
                            }
                        }
                    }
                }
            } catch (e: Exception) {
                if (listeningUDP) {
                    Log.e("UDP", "Socket error: ${e.message}")
                }
            } finally {
                udpSocket?.close()
                udpSocket = null
            }
        }.start()
    }

    // Load your YOLOv8 TFLite model and labels
    private fun loadModelAndLabels() {
        try {
            val afd = assets.openFd("best_320_tushar_float32.tflite")
            val fis = afd.createInputStream()
            val fc = fis.channel
            val mapped = fc.map(FileChannel.MapMode.READ_ONLY, afd.startOffset, afd.declaredLength)
            tfLiteModel = Interpreter(mapped, Interpreter.Options().apply { setNumThreads(4) })
            labels = assets.open("labels.txt").bufferedReader().useLines { it.toList() }
        } catch (e: Exception) {
            showError("Model load failed: ${e.message}")
        }
    }

    // Allocate model input/output buffer
    private fun initBuffers() {
        inputBuffer =
            ByteBuffer.allocateDirect(INPUT_SIZE_BYTES).apply { order(ByteOrder.nativeOrder()) }
        outputBuffer =
            ByteBuffer.allocateDirect(OUTPUT_SIZE_BYTES).apply { order(ByteOrder.nativeOrder()) }
    }

    // Preprocess input image
    private fun preprocess(bitmap: Bitmap) =
        Bitmap.createScaledBitmap(bitmap, MODEL_WIDTH, MODEL_HEIGHT, true)

    // Convert bitmap to float32 RGB input
    private fun convertToInput(buffered: Bitmap) {
        inputBuffer.rewind()
        val pixels = IntArray(MODEL_WIDTH * MODEL_HEIGHT)
        buffered.getPixels(pixels, 0, MODEL_WIDTH, 0, 0, MODEL_WIDTH, MODEL_HEIGHT)
        for (p in pixels) {
            inputBuffer.putFloat((p shr 16 and 0xFF) / 255.0f) // R
            inputBuffer.putFloat((p shr 8 and 0xFF) / 255.0f)  // G
            inputBuffer.putFloat((p and 0xFF) / 255.0f)        // B
        }
        inputBuffer.rewind()
    }

    // Run model, parse full YOLO output with NMS (Detector.kt style)
    private fun runInference(bitmap: Bitmap): Detection? {
        val start = System.currentTimeMillis()
        try {
            val resized = preprocess(bitmap)
            convertToInput(resized)
            outputBuffer.rewind()
            tfLiteModel.run(inputBuffer, outputBuffer)
            outputBuffer.rewind()
            lastInferenceTime = System.currentTimeMillis() - start
            return yoloParseDetection()
        } catch (e: Exception) {
            lastInferenceTime = System.currentTimeMillis() - start
            Log.e("TFLite", "Inference error: ${e.message}")
            return null
        }
    }

    // Parse the output tensor: scan all anchors, classes, do NMS
    private fun yoloParseDetection(): Detection? {
        return try {
            val fb = outputBuffer.asFloatBuffer()
            val array = FloatArray(NUM_CHANNELS * NUM_ELEMENTS)
            fb.get(array)
            val detections = bestBoxes(array)
            detections.maxByOrNull { it.confidence }
        } catch (e: Exception) {
            Log.e("TFLite", "parseDetection error: ${e.message}")
            null
        }
    }

    // Parse all boxes/classes and perform NMS (Detector.kt style)
    private fun bestBoxes(array: FloatArray): List<Detection> {
        val boxes = mutableListOf<Detection>()
        for (c in 0 until NUM_ELEMENTS) {
            var maxConf = CONFIDENCE_THRESHOLD
            var maxIdx = -1
            var j = 4
            var arrayIdx = c + NUM_ELEMENTS * j
            while (j < NUM_CHANNELS) {
                if (array[arrayIdx] > maxConf) {
                    maxConf = array[arrayIdx]
                    maxIdx = j - 4
                }
                j++
                arrayIdx += NUM_ELEMENTS
            }
            if (maxConf > CONFIDENCE_THRESHOLD && maxIdx >= 0) {
                // box coords are normalized [0,1]
                val cx = array[c]
                val cy = array[c + NUM_ELEMENTS]
                val w = array[c + NUM_ELEMENTS * 2]
                val h = array[c + NUM_ELEMENTS * 3]
                boxes.add(
                    Detection(
                        classId = maxIdx,
                        confidence = maxConf,
                        cx = cx, cy = cy, w = w, h = h
                    )
                )
            }
        }
        return applyNMS(boxes)
    }

    // Standard NMS: greedily pick highest, suppress overlap
    private fun applyNMS(boxes: List<Detection>): MutableList<Detection> {
        val sorted = boxes.sortedByDescending { it.confidence }.toMutableList()
        val selected = mutableListOf<Detection>()
        while (sorted.isNotEmpty()) {
            val first = sorted.first()
            selected.add(first)
            sorted.removeAt(0)
            val it = sorted.iterator()
            while (it.hasNext()) {
                val box = it.next()
                if (calculateIoU(first, box) >= IOU_THRESHOLD) {
                    it.remove()
                }
            }
        }
        return selected
    }

    private fun calculateIoU(b1: Detection, b2: Detection): Float {
        val x1 = maxOf(b1.cx - b1.w / 2, b2.cx - b2.w / 2)
        val y1 = maxOf(b1.cy - b1.h / 2, b2.cy - b2.h / 2)
        val x2 = minOf(b1.cx + b1.w / 2, b2.cx + b2.w / 2)
        val y2 = minOf(b1.cy + b1.h / 2, b2.cy + b2.h / 2)
        val intersection = maxOf(0f, x2 - x1) * maxOf(0f, y2 - y1)
        val area1 = b1.w * b1.h
        val area2 = b2.w * b2.h
        return intersection / (area1 + area2 - intersection)
    }

    // Draw unmodified (no bounding box overlay)
    private fun drawFrame(orig: Bitmap, det: Detection?): Bitmap {
        return orig.copy(Bitmap.Config.ARGB_8888, true)
    }

    private fun updateFPS() {
        val now = System.currentTimeMillis()
        displayedFrameCount++
        val elapsed = now - lastFPSCalcTime
        if (elapsed >= 2000) {
            currentFPS = displayedFrameCount * 1000.0 / elapsed
            displayedFrameCount = 0
            lastFPSCalcTime = now
        }
    }

    private fun formatOutput(det: Detection?): String {
        return buildString {
            append("Display FPS: %.1f\n".format(currentFPS))
            append("Inference: ${lastInferenceTime}ms\n")
            append("Resolution: ${STREAM_WIDTH}x${STREAM_HEIGHT}\n")
            if (det == null) append("Detected: None\n")
            else {
                val name = labels.getOrNull(det.classId) ?: "Unknown"
                append("Detected: $name\n")
                append("Confidence: %.2f\n".format(det.confidence))
            }
        }
    }

    // Announce top detection (unchanged TTS logic with added no-detection speech)
    private fun announceDetection(det: Detection?) {
        val now = System.currentTimeMillis()
        if (det != null) {
            if (now - lastSpokenTime >= ttsDelayMs &&
                (det.classId != lastAnnouncedClassId ||
                        kotlin.math.abs(det.confidence - lastAnnouncedConfidence) > 0.05f)
            ) {
                val name = labels.getOrNull(det.classId) ?: "unknown"
                val msg = "object detected: $name"
                tts.speak(msg, TextToSpeech.QUEUE_FLUSH, null, "det_id")
                lastSpokenTime = now
                ttsDelayMs = (2000L..3000L).random()
                lastAnnouncedClassId = det.classId
                lastAnnouncedConfidence = det.confidence
            }
        } else {
            // Speak warning with a 2 second delay
            if (now - lastSpokenTime >= ttsDelayMs) {
                val handler = Handler(Looper.getMainLooper())
                handler.postDelayed({
                    tts.speak("NO OBJECT DETECTED BUT BE CAREFUL", TextToSpeech.QUEUE_FLUSH, null, "no_det_id")
                    lastSpokenTime = System.currentTimeMillis()
                    lastAnnouncedClassId = null
                    lastAnnouncedConfidence = -1f
                }, 2000L)
            }
        }
    }

    // MJPEG streaming from ESP32, frame decoding/processing
    private fun startStream(ipWithPort: String) {
        Thread {
            var conn: HttpURLConnection? = null
            try {
                val url = URL("http://$ipWithPort/stream")
                conn = url.openConnection() as HttpURLConnection
                conn.connectTimeout = 8000
                conn.readTimeout = 15000
                conn.requestMethod = "GET"
                conn.doInput = true
                conn.connect()

                val input = BufferedInputStream(conn.inputStream, 4096)
                val boundary = getBoundary(conn.getHeaderField("Content-Type"))
                if (boundary == null) {
                    showError("No boundary")
                    return@Thread
                }

                val delim = "--$boundary".toByteArray()
                val buf = ByteArray(4096)
                val baos = ByteArrayOutputStream()
                displayedFrameCount = 0
                lastFPSCalcTime = System.currentTimeMillis()

                while (streaming) {
                    val r = input.read(buf)
                    if (r == -1) break
                    baos.write(buf, 0, r)
                    val data = baos.toByteArray()
                    val start = indexOf(data, delim, 0)
                    if (start >= 0) {
                        val end = indexOf(data, delim, start + delim.size)
                        if (end > start) {
                            val part = data.copyOfRange(start + delim.size, end)
                            val js = part.indexOfFirst { it == 0xFF.toByte() }
                            val je = part.indexOfLast { it == 0xD9.toByte() }
                            if (js != -1 && je > js) {
                                val jpeg = part.copyOfRange(js, je + 1)
                                val orig = BitmapFactory.decodeByteArray(jpeg, 0, jpeg.size)
                                if (orig != null) {
                                    frameCounter++
                                    if (frameCounter % INFERENCE_EVERY_N_FRAMES == 0) {
                                        lastDetection = runInference(orig)
                                    }
                                    val processed = drawFrame(orig, lastDetection)
                                    updateFPS()
                                    val text = formatOutput(lastDetection)
                                    mainHandler.post {
                                        frameImageView.setImageBitmap(processed)
                                        outputText.text = text
                                        announceDetection(lastDetection)
                                    }
                                }
                            }
                            baos.reset()
                            if (end < data.size) {
                                baos.write(data, end, data.size - end)
                            }
                        }
                    }
                }
                input.close()
            } catch (e: Exception) {
                showError("Stream error: ${e.message}")
            } finally {
                conn?.disconnect()
                streaming = false
                mainHandler.post { startButton.text = "Start Stream Inference" }
            }
        }.start()
    }

    private fun getBoundary(contentType: String?): String? {
        contentType?.split(";")?.forEach {
            val t = it.trim()
            if (t.startsWith("boundary=")) return t.substringAfter("boundary=").trim('"')
        }
        return null
    }

    private fun indexOf(data: ByteArray, pattern: ByteArray, start: Int): Int {
        for (i in start..data.size - pattern.size) {
            var ok = true
            for (j in pattern.indices) {
                if (data[i + j] != pattern[j]) {
                    ok = false
                    break
                }
            }
            if (ok) return i
        }
        return -1
    }

    private fun showError(msg: String) {
        mainHandler.post {
            Toast.makeText(this, msg, Toast.LENGTH_LONG).show()
            outputText.text = msg
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        streaming = false
        listeningUDP = false
        udpSocket?.close()
        if (::tts.isInitialized) {
            tts.stop()
            tts.shutdown()
        }
    }

    // Detection class, compatible with Detector.kt/NMS logic
    data class Detection(
        val classId: Int,
        val confidence: Float,
        val cx: Float,
        val cy: Float,
        val w: Float,
        val h: Float
    )
}