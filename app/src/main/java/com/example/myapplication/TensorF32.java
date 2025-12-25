package com.example.myapplication;

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

    public float[] getWindow(int windowIndex){
        int blockSize = shape[1] * shape[2];
        float[] out = new float[blockSize];
        int base = windowIndex * blockSize;
        System.arraycopy(data, base, out, 0, blockSize);
        return out;
    }


}
