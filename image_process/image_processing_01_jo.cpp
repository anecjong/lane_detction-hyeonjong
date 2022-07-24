#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>

void onChange(int pos, void* userdata);
int thresh = 128;
cv::TickMeter tm;



int main(void)
{
    cv::VideoCapture cap("../resources/subProject.avi");
    if (!cap.isOpened()){
        std::cout << "Video load failed!" << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> src_pts(4);
    std::vector<cv::Point2f> dst_pts(4);

    src_pts[0] = cv::Point2f(-200, 119);
    src_pts[1] = cv::Point2f(0, 0);
    src_pts[2] = cv::Point2f(639, 0);
    src_pts[3] = cv::Point2f(839, 119);

    dst_pts[0] = cv::Point2f(0, 479);
    dst_pts[1] = cv::Point2f(0, 0);
    dst_pts[2] = cv::Point2f(639, 0);
    dst_pts[3] = cv::Point2f(639, 479);

    cv::Mat per_mat = cv::getPerspectiveTransform(src_pts,dst_pts);
    cv::Mat frame, gray, dst, hist, roi, canny, bin, bin_dst;
    cv::namedWindow("bin");
    cv::createTrackbar("thresh: ", "bin", 0, 255, onChange, 0);
    while (true){
        cap >> frame;
        
        if (frame.empty()){
            std::cout << "empty frame!" << std::endl;
            return -1;
        }

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray,hist);
        roi = hist(cv::Rect(0,350,640,120));
        tm.start();
        cv::fastNlMeansDenoising(roi, dst);
        tm.stop();
        std::cout << tm.getTimeMilli() << std::endl;
        tm.reset();
        tm.start();
        bin = (dst < thresh);
        cv::Canny(bin, canny, 50, 150);
        cv::warpPerspective(bin,bin_dst,per_mat,cv::Size(640,480));
        cv::imshow("dst",dst);
        cv::imshow("bin",bin);
        cv::imshow("canny",canny);
        cv::imshow("bin_dst",bin_dst);

        if (cv::waitKey(3) == 27) break;

    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

void onChange(int pos, void* userdata){
    thresh = pos;
}