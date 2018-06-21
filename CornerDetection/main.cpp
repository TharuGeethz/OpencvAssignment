#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;

using namespace std;

Mat img_1, img_2, img_1_gray, img_2_gray;
Mat dst001, dst199, detected_edges001, detected_edges199;

int edgeThresh = 1;
int lowThreshold001;
int lowThreshold199;
int const max_lowThreshold001 = 100;
int const max_lowThreshold199 = 100;
char* window_name001 = "Edge Map for Image 001";
char* window_name199 = "Edge Map for Image 199";
int ratio = 3;
int kernel_size = 3;

void CannyThreshold(int, void*)
{
    blur( img_1_gray, detected_edges001, Size(3,3) );
    blur( img_2_gray, detected_edges199, Size(3,3) );

    Canny( detected_edges001, detected_edges001, lowThreshold001, lowThreshold001*ratio, kernel_size );
    Canny( detected_edges199, detected_edges199, lowThreshold199, lowThreshold199*ratio, kernel_size );

    dst001 = Scalar::all(0);
    dst199 = Scalar::all(0);

    img_1.copyTo( dst001, detected_edges001);
    img_1.copyTo( dst199, detected_edges199);

    imshow( window_name001, dst001 );
    imshow( window_name199, dst199 );
    imwrite("../output2/output3.jpg",dst001);
    imwrite("../output2/output4.jpg",dst199);
}

int main(int argc, char *argv[])
{
    img_1 = imread( "../data-dir/Fish/img/0001.jpg" );
    img_2 = imread( "../data-dir/Fish/img/0199.jpg" );

    if( !img_1.data || !img_2.data)
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    dst001.create( img_1.size(), img_1.type() );
    dst199.create(img_2.size(), img_2.type());

    cvtColor( img_1, img_1_gray, COLOR_BGR2GRAY );
    cvtColor( img_2, img_2_gray, COLOR_BGR2GRAY );

    namedWindow( window_name001, WINDOW_AUTOSIZE );
    namedWindow( window_name199, WINDOW_AUTOSIZE );

    createTrackbar( "Min Threshold:", window_name001, &lowThreshold001, max_lowThreshold001, CannyThreshold );
    createTrackbar( "Min Threshold:", window_name199, &lowThreshold199, max_lowThreshold199, CannyThreshold );

    CannyThreshold(0, 0);
    cv::waitKey(0);
    return 0;
}
