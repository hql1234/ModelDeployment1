package com.example.myapplication;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;
import android.widget.Button;
import android.widget.Toast;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.graphics.Insets;
import androidx.core.view.ViewCompat;
import androidx.core.view.WindowInsetsCompat;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.LiteModuleLoader;
import org.pytorch.torchvision.TensorImageUtils;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class MainActivity extends AppCompatActivity {

    Button btnRunModel;
    ExecutorService executor = Executors.newSingleThreadExecutor();

    private Bitmap mBitmap = null;
    private Module mModule = null;

    public static String assetFilePath(Context context, String assetName) throws IOException {
        File file = new File(context.getFilesDir(), assetName);
        if (file.exists() && file.length() > 0) {
            return file.getAbsolutePath();
        }

        try (InputStream is = context.getAssets().open(assetName)) {
            try (OutputStream os = new FileOutputStream(file)) {
                byte[] buffer = new byte[4 * 1024];
                int read;
                while ((read = is.read(buffer)) != -1) {
                    os.write(buffer, 0, read);
                }
                os.flush();
            }
            return file.getAbsolutePath();
        }
    }



    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        EdgeToEdge.enable(this);
        setContentView(R.layout.activity_main);

        ViewCompat.setOnApplyWindowInsetsListener(findViewById(R.id.main), (v, insets) -> {
            Insets systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars());
            v.setPadding(systemBars.left, systemBars.top, systemBars.right, systemBars.bottom);
            return insets;
        });

        btnRunModel = findViewById(R.id.btnRunModel);
        btnRunModel.setOnClickListener(v -> runModelAsync());

        try {
            mBitmap = BitmapFactory.decodeStream(getAssets().open("deeplab/deeplab.jpg"));
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }

        try {
            mModule = LiteModuleLoader.load(MainActivity.assetFilePath(getApplicationContext(), "deeplabv3_scripted_optimized.ptl"));
        } catch (IOException e) {
            Log.e("ImageSegmentation", "Error reading assets", e);
            finish();
        }
    }

    private void runModelAsync() {
        btnRunModel.setEnabled(false); // prevent double clicks

        executor.execute(() -> {
            // 🚀 Heavy model work here
            double result = runModelInference();

            runOnUiThread(() -> {
                btnRunModel.setEnabled(true);
                Toast.makeText(this, "InferenceTime = " + result + " ms", Toast.LENGTH_LONG ).show();
            });
        });
    }

    private double runModelInference()  {
        final long startTime = SystemClock.elapsedRealtime();
        final Tensor inputTensor = TensorImageUtils.bitmapToFloat32Tensor(mBitmap,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB);
        Map<String, IValue> outTensors = mModule.forward(IValue.from(inputTensor)).toDictStringKey();
        final long inferenceTime = SystemClock.elapsedRealtime() - startTime;
        return inferenceTime;
    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        executor.shutdownNow(); // prevent leaks
    }
}