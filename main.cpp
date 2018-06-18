#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include "opencv2/xfeatures2d.hpp"

using namespace cv;
using namespace cv::xfeatures2d;
using namespace std;

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
  std::vector<KeyPoint> keypoints_1, keypoints_2;
  detector->detect( img_1, keypoints_1 );
  detector->detect( img_2, keypoints_2 );


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

  cv::waitKey(0);
  return 0;
}
