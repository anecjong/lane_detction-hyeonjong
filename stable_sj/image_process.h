#include "opencv2/opencv.hpp"
#include <vector>
#include <cmath>

cv::Mat binarization(cv::Mat frame);
cv::Mat hough_detect(cv::Mat frame_bin_warp);