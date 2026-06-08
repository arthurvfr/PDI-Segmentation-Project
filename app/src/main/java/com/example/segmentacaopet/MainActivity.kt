package com.example.segmentacaopet

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import java.io.FileInputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var btnSelecionar: Button
    private lateinit var btnSegmentar: Button
    private lateinit var ivOriginal: ImageView
    private lateinit var ivMascara: ImageView

    private var selectedBitmap: Bitmap? = null
    private var tfliteInterpreter: Interpreter? = null

    private val IMAGE_SIZE = 128

    private val galleryLauncher = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK && result.data != null) {
            val imageUri: Uri? = result.data?.data
            imageUri?.let {
                val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, it)
                selectedBitmap = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true)

                ivOriginal.setImageBitmap(selectedBitmap)
                btnSegmentar.isEnabled = true
                ivMascara.setImageDrawable(null)
            }
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        btnSelecionar = findViewById(R.id.btnSelecionar)
        btnSegmentar = findViewById(R.id.btnSegmentar)
        ivOriginal = findViewById(R.id.ivOriginal)
        ivMascara = findViewById(R.id.ivMascara)

        inicializarModelo()

        btnSelecionar.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            galleryLauncher.launch(intent)
        }

        btnSegmentar.setOnClickListener {
            selectedBitmap?.let { bitmap ->
                realizarSegmentacao(bitmap)
            }
        }
    }

    private fun inicializarModelo() {
        try {
            val fileDescriptor = assets.openFd("unet_segmentacao_fp32.tflite")
            val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
            val fileChannel = inputStream.channel
            val startOffset = fileDescriptor.startOffset
            val declaredLength = fileDescriptor.declaredLength
            val modelBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)

            tfliteInterpreter = Interpreter(modelBuffer)
        } catch (e: Exception) {
            e.printStackTrace()
        }
    }

    private fun realizarSegmentacao(bitmap: Bitmap) {
        val inputBuffer = ByteBuffer.allocateDirect(1 * 3 * IMAGE_SIZE * IMAGE_SIZE * 4)
        inputBuffer.order(ByteOrder.nativeOrder())

        val intValues = IntArray(IMAGE_SIZE * IMAGE_SIZE)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        var pixel = 0
        for (c in 0 until 3) {
            for (y in 0 until IMAGE_SIZE) {
                for (x in 0 until IMAGE_SIZE) {
                    val valor = intValues[y * IMAGE_SIZE + x]
                    val canalFloat = when (c) {
                        0 -> ((valor shr 16) and 0xFF) / 255.0f
                        1 -> ((valor shr 8) and 0xFF) / 255.0f
                        else -> (valor and 0xFF) / 255.0f
                    }
                    inputBuffer.putFloat(canalFloat)
                }
            }
        }

        val outputArray = Array(1) { Array(3) { Array(IMAGE_SIZE) { FloatArray(IMAGE_SIZE) } } }
        tfliteInterpreter?.run(inputBuffer, outputArray)

        val mascaraBitmap = Bitmap.createBitmap(IMAGE_SIZE, IMAGE_SIZE, Bitmap.Config.ARGB_8888)

        for (y in 0 until IMAGE_SIZE) {
            for (x in 0 until IMAGE_SIZE) {

                val probPet = outputArray[0][0][y][x]
                val probFundo = outputArray[0][1][y][x]
                val probBorda = outputArray[0][2][y][x]

                var maxClass = 0
                var maxProb = probFundo
                if (probPet > maxProb) { maxProb = probPet; maxClass = 1 }
                if (probBorda > maxProb) { maxProb = probBorda; maxClass = 2 }

                val color = when (maxClass) {
                    1 -> Color.argb(150, 255, 0, 0)
                    2 -> Color.argb(150, 0, 0, 255)
                    else -> Color.TRANSPARENT
                }
                mascaraBitmap.setPixel(x, y, color)
            }
        }

        ivMascara.setImageBitmap(mascaraBitmap)
    }
}