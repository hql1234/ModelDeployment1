package com.example.myapplication;

import android.content.Context;

import com.example.myapplication.jnpy.Npy;

public class TensorF32 {
    public final Npy npy;
    public final int[] shape;
    public final float[] data;

    public TensorF32(Npy npy){
        this.npy = npy;
        this.shape = npy.shape;
        this.data = npy.floatElements();
    }

    public TensorF32(Context ctx, String assetPath){
        this(NpyLoadHelper.loadNumpy(ctx, assetPath));
    }

    public float[] getWindow(int windowIndex){
        int blockSize = shape[1] * shape[2];
        float[] out = new float[blockSize];
        int base = windowIndex * blockSize;
        System.arraycopy(data, base, out, 0, blockSize);
        return out;
    }

    public float[] getData(int firstDim){
        int blockSize = shape[1] * shape[2];
        float[] out = new float[blockSize];
        int base = firstDim * blockSize;
        System.arraycopy(data, base, out, 0, blockSize);
        return out;
    }

    public float[] getData(int firstDim, int secondDim){
        int A = shape[0];
        int B = shape[1];
        int C = shape[2];
        float[] out = new float[C];
        // int index = firstDim * B * C + secondDim * C;
        int base = (firstDim * B + secondDim) * C;
        System.arraycopy(data, base, out, 0, C);
        return out;
    }

    public float getData(int firstDim, int secondDim, int thirdDim){
        int A = shape[0];
        int B = shape[1];
        int C = shape[2];
        // int index = firstDim * B * C + secondDim * C + thirdDim;
        int index = (firstDim * B + secondDim) * C + thirdDim;
        return data[index];
    }
}
