package com.example.opencv_ocr;

import static java.nio.file.Files.write;

import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.ContentUris;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.content.res.AssetManager;
import android.database.Cursor;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.os.Environment;
import android.provider.DocumentsContract;
import android.provider.MediaStore;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import com.googlecode.tesseract.android.TessBaseAPI;

import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class MainActivity extends AppCompatActivity {

    private TessBaseAPI baseAPI;
    private InputStream m_instream;
    private FileOutputStream opentxt_outputstream;
    private String datapath, datapath_result;

    private Button btn_load, btn_bin, btn_rec;
    private ImageView img;
    private TextView text;

    private Bitmap bitmap, bit, orig;
    private String imagePath;
    private List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
    private List<Point> ScreenCnt;
    private Mat warp;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        datapath = Environment.getExternalStorageDirectory().toString() + "/Download/tesseract/";
        datapath_result = Environment.getExternalStorageDirectory().toString() + "/Download/tesseract/";
        bitmap = BitmapFactory.decodeResource(this.getResources(), R.drawable.img);

        btn_load = findViewById(R.id.IDC_BTN_LOAD);
        btn_bin = findViewById(R.id.IDC_BTN_BIN);
        btn_rec = findViewById(R.id.IDC_BTN_REC);

        img = findViewById(R.id.IDC_IMGVIEW);
        text = findViewById(R.id.IDC_TEXT);


        img.setImageBitmap(bitmap);
        btn_load.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                initTessBaseAPI();
                openSysAlbum();
            }
        });
        btn_bin.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Bin_Image();
            }
        });
        btn_rec.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                Rec_Image();
            }
        });

    }

    @Override
    protected void onResume() {
        super.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.i("CV", "未找到opencv库，使用openmanager");
        } else {
            Log.i("CV", "找到opencv库，使用opencv");
        }
    }

    private void initTessBaseAPI() {
        AssetManager assetManager = getAssets();
        try {
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                int writePermission = checkSelfPermission(Manifest.permission.WRITE_EXTERNAL_STORAGE);
                if (writePermission != PackageManager.PERMISSION_GRANTED) {
                    requestPermissions(new String[]{Manifest.permission.WRITE_EXTERNAL_STORAGE}, 101);
                    return;
                }
            }
            File dir = new File(datapath + "tessdata");
            if (!dir.exists()) {
                if (!dir.mkdirs()) {
                    Toast.makeText(this, "创建文件夹失败！", Toast.LENGTH_SHORT).show();
                } else {
                    Toast.makeText(this, "创建文件夹成功！", Toast.LENGTH_SHORT).show();
                }
            }
            try {
                m_instream = assetManager.open("tessdata/chi_sim.traineddata");
            } catch (IOException e) {
                e.printStackTrace();
            }
            File file = new File(dir, "chi_sim.traineddata");
            FileOutputStream output = new FileOutputStream(file);
            byte[] buff = new byte[1024];
            int len = 0;
            while ((len = m_instream.read(buff)) != -1) {
                output.write(buff, 0, len);
            }
            m_instream.close();
            output.close();
            Toast.makeText(this, "加载字库成功！", Toast.LENGTH_SHORT).show();

        } catch (IOException ioe) {
            ioe.printStackTrace();
            Toast.makeText(this, "加载字库失败！", Toast.LENGTH_SHORT).show();
        }
    }

    //重载onActivityResult方法，获取相应数据
    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        handleImageOnKitKat(data);
    }

    public static int ALBUM_RESULT_CODE = 0x999;


    //    打开系统相册
//    定义Intent跳转到特定图库的Uri下挑选，然后将挑选结果返回给Activity
    private void openSysAlbum() {
        Intent albumIntent = new Intent(Intent.ACTION_PICK);
        albumIntent.setDataAndType(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, "image/*");
        startActivityForResult(albumIntent, ALBUM_RESULT_CODE);
    }

    //    @TargetApi(value = 19)
    private void handleImageOnKitKat(Intent data) {
//        String imagePath = null;
        Uri uri = data.getData();
        if (DocumentsContract.isDocumentUri(this, uri)) {
            // 如果是document类型的Uri，则通过document id处理
            String docId = DocumentsContract.getDocumentId(uri);
            if ("com.android.providers.media.documents".equals(uri.getAuthority())) {
                String id = docId.split(":")[1];
                // 解析出数字格式的id
                String selection = MediaStore.Images.Media._ID + "=" + id;
                imagePath = getImagePath(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, selection);
            } else if ("com.android.providers.downloads.documents".equals(uri.getAuthority())) {
                Uri contentUri = ContentUris.withAppendedId(Uri.parse("content: //downloads/public_downloads"), Long.parseLong(docId));
                imagePath = getImagePath(contentUri, null);
            }
        } else if ("content".equalsIgnoreCase(uri.getScheme())) {
            // 如果是content类型的Uri，则使用普通方式处理
            imagePath = getImagePath(uri, null);
        } else if ("file".equalsIgnoreCase(uri.getScheme())) {
            // 如果是file类型的Uri，直接获取图片路径即可
            imagePath = uri.getPath();
        }
        // 根据图片路径显示图片
        displayImage(imagePath);
        System.out.println(imagePath);
    }

    @SuppressLint("Range")
    private String getImagePath(Uri uri, String selection) {
        String path = null;
        Cursor cursor = getContentResolver().query(uri, null, selection, null, null);
        if (cursor != null) {

            if (cursor.moveToFirst()) {
                path = cursor.getString(cursor.getColumnIndex(MediaStore.Images.Media.DATA));
            }
            cursor.close();
        }
        return path;
    }

    //展示图片
    private void displayImage(String imagePath) {
        Bitmap bitmap = BitmapFactory.decodeFile(imagePath);
        img.setImageBitmap(bitmap);
    }


//    private void

    public void Bin_Image() {
        if (imagePath == null){
            Toast.makeText(this, "请先添加图片~", Toast.LENGTH_SHORT).show();
        }
        else {
            bitmap = BitmapFactory.decodeFile(imagePath);
            Mat image = new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType. CV_8UC3);
            Utils.bitmapToMat(bitmap, image);

            Mat orig = image.clone();

            float ratio = bitmap.getHeight() / 500.0f;
//        float width = bitmap.getWidth() / ratio;

            Imgproc.resize(orig, image, new Size(bitmap.getWidth() / ratio, 500.0f));
//        int h = image.height();
//        int w = image.width();

            Mat gray = new Mat(image.height(), image.width(), CvType. CV_8UC1);
            Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);     // 灰度化
//        Imgproc.threshold(gray, gray, 150, 255.0, Imgproc.THRESH_BINARY_INV);
            Imgproc.medianBlur(gray,gray, 5);
//        Imgproc.GaussianBlur(gray, gray, new Size(3,3),0,0);
            Mat edge = new Mat(image.height(), image.width(), CvType. CV_8UC1);
            Imgproc.Canny(gray, edge, 75, 200);   // 找边缘
//        bit = Bitmap.createBitmap(edge.width(), edge.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
//        Utils.matToBitmap(edge, bit);
//        img.setImageBitmap(bit);

            Mat edged = edge.clone();
            Mat hierarchy = new Mat();
//        List<MatOfPoint> contours = new ArrayList<MatOfPoint>();
            Imgproc.findContours(edged, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE);   // 找轮廓

//        Imgproc.drawContours(image, contours, -1, new Scalar(255, 255, 255), 1);

            // 对轮廓按照面积排序
            ListComparetor listComparetor = new ListComparetor();
            contours.sort(listComparetor);

//        double[] areas = new double[contours.size()];
            Iterator<MatOfPoint> iterator = contours.iterator();
//        int j = 0;
            double peri;
            while (iterator.hasNext()) {
                MatOfPoint contour = iterator.next();
//            areas[j] = Imgproc.contourArea(contour);
                peri = Imgproc.arcLength(new MatOfPoint2f(contour.toArray()), true); // 计算轮廓近似多边形
                MatOfPoint2f approx = new MatOfPoint2f();
                Imgproc.approxPolyDP(new MatOfPoint2f(contour.toArray()), approx, 0.02 * peri, true);
                if (approx.toList().size() >= 4) {
                    ScreenCnt = approx.toList().subList(0, 4);
                    break;
                }
//            j++;
            }
            float[][] points = new float[4][2];
            for (int k=0;k<4;k++){
                points[k][0] = (float) ScreenCnt.get(k).x * ratio;
                points[k][1] = (float) ScreenCnt.get(k).y * ratio;

            }

            warp = four_point_transform(orig, points);

            bit = Bitmap.createBitmap(warp.width(), warp.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
            Utils.matToBitmap(warp, bit);
            img.setImageBitmap(bit);



//        warped =



////        List<MatOfPoint> topFive = new ArrayList<>();
//
////        int size = contours.size();
//        for (int i=0; i < 5 ;i++){
////            topFive.add(contours.get(i));
//            peri = Imgproc.arcLength(new MatOfPoint2f(contours.get(i).toArray()),true); // 计算轮廓近似
//            MatOfPoint2f approx = new MatOfPoint2f();
//            Imgproc.approxPolyDP(new MatOfPoint2f(contours.get(i).toArray()), approx, 0.02 * peri,true);
//
////            int b = approx.toList().size();
//            if (approx.toList().size() == 4){
//                List<Point> ScreenCnt = approx.toList();
//                break;
////                Imgproc.drawContours(image, approx, -1, new Scalar(255, 0, 0), 2);
////                Imgproc.drawContours(image, contours, i, new Scalar(255, 0, 0), 1, 8);
////                bit = Bitmap.createBitmap(image.width(), image.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
////
////                Utils.matToBitmap(image, bit);
////                img.setImageBitmap(bit);
////                break;
//            }
//        }


//        double[] areas = new double[contours.size()];
//        Iterator<MatOfPoint> iterator = contours.iterator();
//        int i = 0;
//        while (iterator.hasNext()) {
//            MatOfPoint contour = iterator.next();
//            areas[i] = Imgproc.contourArea(contour);
//            i++;
//        }
//        double max = areas[0];
//        int index = 0;
//        for (int t = 0; t < areas.length; t++)
//            if (areas[t] > max){
//                max = areas[t];
//                index = t;
//            }


//        MatOfPoint screenCnt = contours.get(index);

//        Mat warped = four_point_transform(orig, screenCnt,* ratio)


//        int max_id = Arrays.stream(areas).findAny()


//        Mat src=new Mat(bitmap.getHeight(), bitmap.getWidth(), CvType.CV_8UC3);    // 创建一个和被识别图片一样大小的新mat
//        Utils.bitmapToMat(bitmap, src);    // 将图片存入这个MAT 就相当于src=mat_cam
//
//        Imgproc.cvtColor(src, src, Imgproc.COLOR_BGR2GRAY);
//        Imgproc.threshold(src, src, 150, 255.0, Imgproc.THRESH_BINARY_INV);
//
//        bit = Bitmap.createBitmap(src.width(), src.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
//        Utils.matToBitmap(src, bit);
//        img.setImageBitmap(bit);
        }

    }

    private int[] arg(float[][] a){
        float[] sum = new float[4];
        float[] diff = new float[4];
        for(int k =0; k < 4; k++){
            sum[k] = a[k][0] + a[k][1];
            diff[k] = a[k][0] - a[k][1];
        }
        float min = sum[0], max = sum[0];
        int mi = 0, ma = 0, Mi = 0, Ma = 0;
        float Min = diff[0], Max = diff[0];
        int[] result = new int[4];
        for (int k = 0; k < 4; k ++){
            if (sum[k] < min){
                min = sum[k];
                mi = k;
            }
            if (sum[k] > max){
                max = sum[k];
                ma = k;
            }
            if (diff[k] < Min){
                Min = diff[k];
                Mi = k;
            }
            if (diff[k] > Max){
                Max = diff[k];
                Ma = k;
            }
        }
        result[0] = mi;
        result[1] = ma;
        result[2] = Mi;
        result[3] = Ma;

        return result;
    }


    private Mat four_point_transform(Mat image, float[][] pts) {
        Mat warp = image.clone();
        int[] arg = arg(pts);
        float[] tl = pts[arg[0]];   // 左上
        float[] br = pts[arg[1]];   // 右下
        float[] tr = pts[arg[3]];   // 右上
        float[] bl = pts[arg[2]];   // 左下

        float widthA = (float) Math.sqrt(Math.pow(br[0] - bl[0], 2) + Math.pow(br[1] - bl[1], 2));
        float widthB = (float) Math.sqrt(Math.pow(tr[0] - tl[0], 2) + Math.pow(tr[1] - tl[1], 2));
        double maxWidth = Math.max(widthA, widthB);

        float hightA = (float) Math.sqrt(Math.pow(tr[0] - br[0], 2) + Math.pow(tr[1] - br[1], 2));
        float hightB = (float) Math.sqrt(Math.pow(tl[0] - bl[0], 2) + Math.pow(tl[1] - bl[1], 2));
        float maxHight = Math.max((float) hightA, (float) hightB);

//        int[] a = new int[]{(int) tl[0], (int) tl[1], (int) tr[0],(int) tr[1], (int) br[0],(int) br[1], (int) bl[0],(int) bl[1]};
        Point tl_point = new Point();
        tl_point.x = (float) tl[0];
        tl_point.y = (float) tl[1];
        Point tr_point = new Point();
        tr_point.x = (float) tr[0];
        tr_point.y = (float) tr[1];
        Point br_point = new Point();
        br_point.x = (float) br[0];
        br_point.y = (float) br[1];
        Point bl_point = new Point();
        bl_point.x = (float) bl[0];
        bl_point.y = (float) bl[1];
        MatOfPoint2f rect = new MatOfPoint2f(tl_point, tr_point, br_point, bl_point);

        Point tl_dst = new Point();
        tl_dst.x = (float) 0;
        tl_dst.y = (float) 0;
        Point tr_dst = new Point();
        tr_dst.x = (float) maxWidth - 1;
        tr_dst.y = (float) 0;
        Point br_dst = new Point();
        br_dst.x = (float) maxWidth - 1;
        br_dst.y = (float) maxHight - 1;
        Point bl_dst = new Point();
        bl_dst.x = (float) 0;
        bl_dst.y = (float) maxHight - 1;
        MatOfPoint2f dst = new MatOfPoint2f(tl_dst, tr_dst, br_dst, bl_dst);


        Mat M = Imgproc.getPerspectiveTransform(rect, dst);
        Imgproc.warpPerspective(image, warp, M,  new Size(maxWidth, maxHight));

        return warp;
    }


    private void Rec_Image() {
            if (warp == null) {
                Toast.makeText(this, "请先进行图片矫正~", Toast.LENGTH_SHORT).show();
            } else {
                baseAPI = new TessBaseAPI();
                boolean success = baseAPI.init(datapath, "chi_sim");
                if (success) {
                    Log.i("OCR", "Successful!");
                } else {
                    Log.i("OCR", "Fail!");
                    Toast.makeText(this, "初始化中文字库失败！", Toast.LENGTH_SHORT).show();
                    return;
                }

//                baseAPI.setVariable(TessBaseAPI.VAR_CHAR_WHITELIST, "0123456789.");
                baseAPI.setVariable(TessBaseAPI.VAR_CHAR_BLACKLIST, "[email protected]#$%^&*()_+-[]}{;:'\"\\|~`,/<>?");
                baseAPI.setPageSegMode(TessBaseAPI.PageSegMode.PSM_AUTO);

                Mat gray = new Mat(warp.height(), warp.width(), CvType. CV_8UC1);
                Imgproc.cvtColor(warp, gray, Imgproc.COLOR_BGR2GRAY);     // 灰度化
//                Imgproc.adaptiveThreshold(gray, gray, 255, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY_INV, 35, -3);
                Imgproc.threshold(gray, gray, 150, 255.0, Imgproc.THRESH_BINARY_INV);
//                bit = Bitmap.createBitmap(gray.width(), gray.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
//                Utils.matToBitmap(gray, bit);
//                img.setImageBitmap(bit);


                int rows = gray.rows(), cols = gray.cols(), scale1 = 20, scale2 = 10;
                Mat kernel1 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(cols/scale1, 1));
//                System.out.println("************************");
//                System.out.println(cols|scale1);
                Mat kernel2 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, new Size(1, rows/scale2));
                Mat erode1 = gray.clone(), erode2 = gray.clone(), dilaterow = gray.clone(), dilatecol = gray.clone();

                Imgproc.erode(gray, erode1, kernel1);    // 识别横线
                Imgproc.dilate(erode1, dilaterow, kernel1);

//                bit = Bitmap.createBitmap(erode1.width(), erode1.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
//                Utils.matToBitmap(dilaterow, bit);
//                img.setImageBitmap(bit);

                Imgproc.erode(gray, erode2, kernel2);    // 识别竖线
                Imgproc.dilate(erode2, dilatecol, kernel2);

                Mat bitwiseAnd = dilatecol.clone();
                Core.bitwise_and(dilaterow, dilatecol, bitwiseAnd);

//                bit = Bitmap.createBitmap(dilatecol.width(), dilatecol.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
//                Utils.matToBitmap(bitwiseAnd, bit);
//                img.setImageBitmap(bit);


//                byte[] data = new byte[1];
//                byte[] data1 = new byte[1];
//                byte[] data2 = new byte[1];
//                for (int i=0; i<bitwiseAnd.height(); i++){
//                    for (int j=0; j<bitwiseAnd.width(); j++){
//                        dilaterow.get(i, j, data1);
//                        dilatecol.get(i, j, data2);
//                        bitwiseAnd.get(i, j, data);
//                        if ((data1[0] !=0)&(data2[0] !=0)) {
//                            System.out.println(data1[0]);
//                            System.out.println(data2[0]);
//                            System.out.println(data[0]);
//                            System.out.println("+++++++++++++++++++++++++++++++++++++");
//                        }
//
//
//
//                        }
//                    }

//
//                bit = Bitmap.createBitmap(bitwiseAnd.width(), bitwiseAnd.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
//                Utils.matToBitmap(bitwiseAnd, bit);
//                img.setImageBitmap(bit);


                ArrayList<Integer> xs = new ArrayList<Integer>();
                ArrayList<Integer> ys = new ArrayList<Integer>();
                ArrayList<Integer> mylistx = new ArrayList<Integer>();
                ArrayList<Integer> mylisty = new ArrayList<Integer>();

                byte[] data = new byte[1];
                for (int i=0; i<bitwiseAnd.height(); i++){
                    for (int j=0; j<bitwiseAnd.width(); j++){
                        bitwiseAnd.get(i, j, data);
                        if (data[0] != 0){
                            xs.add(j);
                            ys.add(i);
                        }
                    }
                }

                Integer[] myxs = xs.toArray(new Integer[xs.size()]);
                Arrays.sort(myxs);
                Integer[] myys = ys.toArray(new Integer[ys.size()]);
                Arrays.sort(myys);

                for (int i=0; i<myxs.length-1; i++){
                    if (myxs[i+1] - myxs[i] > 10){
                        mylistx.add(myxs[i]);
                    }
                }
                mylistx.add(myxs[myxs.length-1]);

                for (int i=0; i<myys.length-1; i++){
                    if (myys[i+1] - myys[i] > 10){
                        mylisty.add(myys[i]);
                    }
                }
                mylisty.add(myys[myys.length-1]);

//                AssetManager assetManager = getAssets();
//                File dir = new File(datapath + "tessdata");

//                try {
//                    opentxt_outputstream = new FileOutputStream("tessdata/result.txt");
//                } catch (FileNotFoundException e) {
//                    e.printStackTrace();
//                }


                StringBuilder result = new StringBuilder();
                for (int i=0; i < mylisty.size() -1; i++){
                    int k = 0;
                    for (int j=0; j < mylistx.size() -1; j++){
                        Rect roi = new Rect(mylistx.get(j) + 1, mylisty.get(i) + 1, mylistx.get(j + 1) - mylistx.get(j) - 2, mylisty.get(i + 1) - mylisty.get(i) - 2);
                        Mat img_clip = new Mat(gray, roi);
                        bit = Bitmap.createBitmap(img_clip.width(), img_clip.height(), Bitmap.Config.ARGB_8888);   // 将bit_cam重新创建为裁剪后的大小
                        Utils.matToBitmap(img_clip, bit);
                        baseAPI.setImage(bit);
                        String res = baseAPI.getUTF8Text();
                        result.append(res);
//                        System.out.println(res);

//                        try {
//                            opentxt_outputstream.write(res.getBytes());
//                        } catch (IOException e) {
//                            e.printStackTrace();
//                        }
//
//                        if (k < mylistx.size()-1){
//                            try {
//                                opentxt_outputstream.write("\t".getBytes());
//                            } catch (IOException e) {
//                                e.printStackTrace();
//                            }
//                        }
                        k ++;
                    }
                    result.append("\t");
//                    try {
//                        opentxt_outputstream.write("\n".getBytes());
//                    } catch (IOException e) {
//                        e.printStackTrace();
//                    }
                }
//                try {
//                    opentxt_outputstream.close();
//                } catch (IOException e) {
//                    e.printStackTrace();
//                }
                result.append("\n");
                text.setText(result);


//                baseAPI.setImage(bit);
//                String res = baseAPI.getUTF8Text();
//                text.setText(res);
            }
        }


}




















//package com.example.opencv_ocr;
//
//import androidx.appcompat.app.AppCompatActivity;
//
//import android.os.Bundle;
//import android.util.Log;
//import android.view.Gravity;
//import android.widget.Toast;
//
//import org.opencv.android.BaseLoaderCallback;
//import org.opencv.android.LoaderCallbackInterface;
//import org.opencv.android.OpenCVLoader;
//
//public class MainActivity extends AppCompatActivity {
//    private static final String TAG = "MainActivity";
//    //OpenCV库加载并初始化成功后的回调函数
//    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
//        @Override
//        public void onManagerConnected(int status) {
//            // TODO Auto-generated method stub
//            switch (status){
//                case BaseLoaderCallback.SUCCESS:
//                    Log.i(TAG, "成功加载opencv");
//                    Toast toast = Toast.makeText(getApplicationContext(),
//                            "成功加载opencv！", Toast.LENGTH_LONG);
//                    toast.setGravity(Gravity.CENTER, 0, 0);
//                    toast.show();
//                    break;
//                default:
//                    super.onManagerConnected(status);
//                    Log.i(TAG, "加载失败");
//                    Toast toast1 = Toast.makeText(getApplicationContext(),
//                            "加载失败！", Toast.LENGTH_LONG);
//                    toast1.setGravity(Gravity.CENTER, 0, 0);
//                    toast1.show();
//                    break;
//            }
//
//        }
//    };
//
//    @Override
//    protected void onCreate(Bundle savedInstanceState) {
//        super.onCreate(savedInstanceState);
//        setContentView(R.layout.activity_main);
//    }
//
//    @Override
//    public void onResume()
//    {
//        super.onResume();
//        if (!OpenCVLoader.initDebug()) {
//            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
//            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
//        } else {
//            Log.d(TAG, "OpenCV library found inside package. Using it!");
//            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
//        }
//    }
//}