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


  img_1 = cv::imread("../OpencvAssignment/data-dir/Fish/img/0001.jpg" , CV_LOAD_IMAGE_COLOR);
  img_2 = cv::imread("../OpencvAssignment/data-dir/Fish/img/0199.jpg" , CV_LOAD_IMAGE_COLOR);

  if(!img_1.data || !img_2.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }


    cv::Mat image1_crop =img_1(Rect(134,55,60,88));
    int minHessian = 400;

    Ptr<SURF> detector = SURF::create( minHessian );

    std::vector<KeyPoint> keypoints_1, keypoints_2, keypoints_1_crop;
    detector->detect( img_1, keypoints_1 );
    detector->detect( img_2, keypoints_2 );
    detector->detect( img_1, keypoints_1_crop );

    std::fstream features0001Write;
    std::fstream features0002Write;

    features0001Write.open( "../OpencvAssignment/features0001.csv", std::ios::out);
    features0002Write.open( "../OpencvAssignment/features0199.csv", std::ios::out);

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

    matcher.match( descriptors_1, descriptors_1_crop, matches );

    Mat img_matches;
//    drawMatches( img_1, keypoints_1, img_2, keypoints_2, matches, img_matches );
    drawMatches( img_1, keypoints_1, image1_crop, keypoints_1_crop, matches, img_matches );

      //-- Show detected matches
    imshow("Matches", img_matches );
    string fileOutputPath = "../OpencvAssignment/output/output_matching.jpg";
    imwrite(fileOutputPath, img_matches);

    double max_dist = 0; double min_dist = 100;

    for( int i = 0; i < descriptors_1.rows; i++ )
      { double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
      }


    printf("-- Max dist : %f \n", max_dist );
    printf("-- Min dist : %f \n", min_dist );

    std::vector< DMatch > good_matches;
    std::vector<Point2f> obj;
    std::vector<Point2f> scene;

    for( int i = 0; i < descriptors_1.rows; i++ )
      { if( matches[i].distance <= 3*min_dist )
         { good_matches.push_back( matches[i]); }
      }

    for( size_t i = 0; i < good_matches.size(); i++ )
      {
        //-- Get the keypoints from the good matches
        obj.push_back( keypoints_1[ good_matches[i].queryIdx ].pt );
        scene.push_back( keypoints_1_crop[ good_matches[i].trainIdx ].pt );
      }
    Mat H = findHomography( obj, scene, RANSAC );

    //-- Get the corners from the image_1 ( the object to be "detected" )
      std::vector<Point2f> obj_corners(4);
      obj_corners[0] = cvPoint(0,0); obj_corners[1] = cvPoint( img_1.cols, 0 );
      obj_corners[2] = cvPoint( img_1.cols, img_1.rows ); obj_corners[3] = cvPoint( 0, img_1.rows );
      std::vector<Point2f> scene_corners(4);
      perspectiveTransform( obj_corners, scene_corners, H);
      //-- Draw lines between the corners (the mapped object in the scene - image_2 )
      line( img_matches, scene_corners[0] + Point2f( img_1.cols, 0), scene_corners[1] + Point2f( img_1.cols, 0), Scalar(0, 255, 0), 4 );
      line( img_matches, scene_corners[1] + Point2f( img_1.cols, 0), scene_corners[2] + Point2f( img_1.cols, 0), Scalar( 0, 255, 0), 4 );
      line( img_matches, scene_corners[2] + Point2f( img_1.cols, 0), scene_corners[3] + Point2f( img_1.cols, 0), Scalar( 0, 255, 0), 4 );
      line( img_matches, scene_corners[3] + Point2f( img_1.cols, 0), scene_corners[0] + Point2f( img_1.cols, 0), Scalar( 0, 255, 0), 4 );
      //-- Show detected matches
//      imshow( "Good Matches & Object detection", img_matches );

    cv::waitKey(0);
    return 0;
}
