#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

//Author Tharushi Geethma Abeysinghe

int main( int argc, char** argv ) {

    cv::Mat img_1;
    cv::Mat img_2;


    img_1 = cv::imread("../OpenCVAssignment/data-dir/Fish/img/0001.jpg" , CV_LOAD_IMAGE_COLOR);
    img_2 = cv::imread("../OpenCVAssignment/data-dir/Fish/img/0199.jpg" , CV_LOAD_IMAGE_COLOR);

    if(!img_1.data || !img_2.data ) {
        std::cout <<  "Could not open or find the image1" << std::endl ;
        return -1;
    }

    Mat midImage = img_1.clone();
    cv::rectangle(img_1, cv::Point(134, 55), cv::Point(134+60, 55+88), cv::Scalar(0, 255, 0));
    cv::Mat image1_crop = midImage(Rect(134,55,60,88));

    int minHessian = 400;

    Ptr<SURF> detector = SURF::create( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_1_crop;

    detector->detect( img_1, keypoints_1 );
    detector->detect( img_2, keypoints_2 );

    detector->detect( image1_crop, keypoints_1_crop );

    std::fstream features0001Write;
    std::fstream features0002Write;

    features0001Write.open( "../OpenCVAssignment/features0001.csv", std::ios::out);
    features0002Write.open( "../OpenCVAssignment/features0199.csv", std::ios::out);

    for ( size_t i = 0; i < keypoints_1.size(); ++i){
        features0001Write << keypoints_1[i].pt.x << ", " << keypoints_1[i].pt.y <<std::endl;
    }

    for ( size_t j = 0; j < keypoints_2.size(); ++j){
        features0002Write << keypoints_2[j].pt.x << ", " << keypoints_2[j].pt.y <<std::endl;
    }

    features0001Write.close();
    features0002Write.close();

    //Step 5 complete

    Mat descriptors_1, descriptors_1_crop;
    Ptr<SURF> extractor = SURF::create();

    extractor->compute( img_1, keypoints_1, descriptors_1 );
    extractor->compute( image1_crop, keypoints_1_crop, descriptors_1_crop );

    BFMatcher matcher(NORM_L2);
    std::vector< DMatch > matches;

    matcher.match(descriptors_1_crop, descriptors_1, matches );

    Mat img_matches;

    drawMatches( image1_crop, keypoints_1_crop, img_1, keypoints_1, matches, img_matches );

    imshow("Show detected matches", img_matches );
    string fileOutputPath = "../OpenCVAssignment/output/output_matching.jpg";
    imwrite(fileOutputPath, img_matches);

    //Step 7 ends from here

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors_1_crop.rows; i++ )
    {
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;

    for( int i = 0; i < descriptors_1_crop.rows; i++ )
    {
        if( matches[i].distance <= 3*min_dist )
        {
            good_matches.push_back( matches[i]);
        }
    }

    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( size_t i = 0; i < good_matches.size(); i++ )
    {

        obj.push_back( keypoints_1_crop[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_1[ good_matches[i].trainIdx ].pt );
    }
    Mat H = findHomography( scene,obj, RANSAC );

    std::vector<Point2f> obj_corners(4);
    obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( image1_crop.cols, 0 );
    obj_corners[2] = cvPoint( image1_crop.cols, image1_crop.rows ); obj_corners[3] = cvPoint( 0, image1_crop.rows );
    std::vector<Point2f> scene_corners(4);
    perspectiveTransform( obj_corners, scene_corners, H);

    line( img_matches, scene_corners[0] + Point2f( image1_crop.cols, 0), scene_corners[1] + Point2f( image1_crop.cols, 0), Scalar(0, 255, 0), 4 );
    line( img_matches, scene_corners[1] + Point2f( image1_crop.cols, 0), scene_corners[2] + Point2f( image1_crop.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[2] + Point2f( image1_crop.cols, 0), scene_corners[3] + Point2f( image1_crop.cols, 0), Scalar( 0, 255, 0), 4 );
    line( img_matches, scene_corners[3] + Point2f( image1_crop.cols, 0), scene_corners[0] + Point2f( image1_crop.cols, 0), Scalar( 0, 255, 0), 4 );

    imshow( "Show Likely object", img_matches );

    cv::waitKey(0);
    return 0;
}
