#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <algorithm>

std::vector<int> black_val(cv::Mat frame_bin, int offset);
std::vector<int> blacks_from_inside(cv::Mat frame_bin, int offset);