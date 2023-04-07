package com.example.opencv_ocr;

import org.opencv.core.MatOfPoint;
import org.opencv.imgproc.Imgproc;

import java.util.Comparator;

public class ListComparetor implements Comparator<MatOfPoint> {
    @Override
    public int compare(MatOfPoint matOfPoint, MatOfPoint t1) {
        if (Imgproc.contourArea(matOfPoint) > Imgproc.contourArea(t1)){
            return -1;
        }else if (Imgproc.contourArea(matOfPoint) == Imgproc.contourArea(t1)){
            return 0;
        }else {
            return +1;
        }
    }
}
