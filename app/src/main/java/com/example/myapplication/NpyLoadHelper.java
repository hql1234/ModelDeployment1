package com.example.myapplication;

import android.content.Context;

import com.example.myapplication.jnpy.Npy;

import java.io.InputStream;

public final class NpyLoadHelper {
    private void NpyLoaderHelper() {}
    public static Npy loadNumpy(Context ctx, String assetPath){
        try (InputStream is = ctx.getAssets().open(assetPath)){
            return new Npy(is);
        } catch (Exception e) {
            throw new RuntimeException("Failed to load NPY: " + assetPath, e);
        }
    }
}
