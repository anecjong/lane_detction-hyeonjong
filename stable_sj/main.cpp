#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <fstream>
#include "image_process.h"
#include "detecting_algorithm.h"

int main(void)
{
    std::ofstream outfile;
    outfile.open("test.csv", std::ios::out);
    int count = 0;
    
    cv::VideoCapture cap("../resources/subProject.avi");
    //cv::VideoCapture cap("../resources/output.avi");
    if (!cap.isOpened()){
        std::cout << "Video load failed!" << std::endl;
        return -1;
    }

    cv::Mat frame, frame_bin;
    std::vector<cv::Mat> channels(3);

    cv::Mat frame_bin_warp, line_board;
    std::vector<int> blacks;
    // roi matrix

    std::vector<cv::Point2f> src_pts(4);
    std::vector<cv::Point2f> dst_pts(4);

    int roi_w = 640, roi_h = 100, sp = 350;
    src_pts[0] = cv::Point2f(0, sp);
    src_pts[1] = cv::Point2f(roi_w-1, sp);
	src_pts[2] = cv::Point2f(roi_w-1, sp+roi_h);
    src_pts[3] = cv::Point2f(0, sp+roi_h);

    int w = 640, h = 240;
	dst_pts[0] = cv::Point2f(0, 0);
    dst_pts[1] = cv::Point2f(w-1, 0);
	dst_pts[2] = cv::Point2f(w-1, h-1);
    dst_pts[3] = cv::Point2f(0, h-1);

    cv::Mat M_pt = cv::getPerspectiveTransform(src_pts,dst_pts);
    cv::Mat M_inv = cv::getPerspectiveTransform(dst_pts,src_pts);

    while (true){
        cap >> frame;
        count++;
        if (frame.empty()){
            std::cout << "empty frame!" << std::endl;
            break;
        }

        frame_bin = binarization(frame);
        cv::warpPerspective(frame_bin, frame_bin_warp, M_pt, cv::Size(w,h));
        line_board = hough_detect(frame_bin_warp);

        cv::Mat roi;
        frame_bin_warp.copyTo(roi, line_board);
        cv::resize(roi, roi, cv::Size(roi_w,roi_h));
        roi = cv::Scalar(255) - roi;

        blacks = blacks_from_inside(roi, 400-sp);

        if (count == 30){
            for (auto& item : blacks){
                outfile << item << ",";
            }
            outfile << std::endl;
            count = 0;
        }
        cv::imshow("frame",frame);
        cv::imshow("frame_bin", frame_bin);
        cv::imshow("line_board", line_board);
        cv::imshow("roi",roi);
        
        if (cv::waitKey(1) == 27) break;

    }

    cap.release();
    cv::destroyAllWindows();
    outfile.close();

    return 0;
}