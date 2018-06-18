#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {

  cv::Mat img_1;
  cv::Mat img_2;

  img_1 = cv::imread("../OpencvAssignment/data-dir/Fish/img/0001.jpg" , CV_LOAD_IMAGE_COLOR);
  img_2 = cv::imread("../OpencvAssignment/data-dir/Fish/img/0199.jpg" , CV_LOAD_IMAGE_COLOR);

  if(! img_1.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }
  if(! img_2.data ) {
      std::cout <<  "Could not open or find the image" << std::endl ;
      return -1;
    }


//  cv::namedWindow( "Display window", cv::WINDOW_AUTOSIZE );
//  cv::imshow( "Display window 1", img_1 );
//  cv::imshow( "Display window 2", img_2 );

  cv::waitKey(0);
  return 0;
}
