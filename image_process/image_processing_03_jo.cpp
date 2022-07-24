#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>

void onChange(int pos, void* userdata);
cv::TickMeter tm;

int h_up = 255, s_up = 255, v_up = 120;
int h_low = 0, s_low = 0, v_low = 0;


cv::Mat frame, frame_hsv;
cv::Mat frame_eq(480,640,CV_8UC3);
cv::Mat frame_inrange(480,640,CV_8UC1);
cv::Mat roi, roi_canny;
std::vector<cv::Mat> channels(3);

int main(void)
{
    cv::VideoCapture cap("../resources/subProject.avi");
    if (!cap.isOpened()){
        std::cout << "Video load failed!" << std::endl;
        return -1;
    }

    cv::namedWindow("frame_inrange");
    cv::namedWindow("frame");
    cv::createTrackbar("h_low: ","frame_inrange", &h_low, 255, onChange, 0);
    cv::createTrackbar("h_up: ","frame_inrange", &h_up, 255, onChange, 0);
    cv::createTrackbar("s_low: ","frame_inrange", &s_low, 255, onChange, 0);
    cv::createTrackbar("s_up: ","frame_inrange", &s_up, 255, onChange, 0);
    cv::createTrackbar("v_low: ","frame_inrange", &v_low, 255, onChange, 0);
    cv::createTrackbar("v_up: ","frame_inrange", &v_up, 255, onChange, 0);

    while (true){
        cap >> frame;
        
        if (frame.empty()){
            std::cout << "empty frame!" << std::endl;
            return -1;
        }
        // to minimize effect of patterns on floor
        cv::medianBlur(frame, frame, 3);
        cv::medianBlur(frame, frame, 3);
        cv::medianBlur(frame, frame, 3);

        cv::Scalar upper_bound(h_up,s_up,v_up);
        cv::Scalar lower_bound(h_low,s_low,v_low);
        cv::cvtColor(frame, frame_hsv,cv::COLOR_BGR2HSV);
        cv::split(frame_hsv, channels);

        cv::equalizeHist(channels[2], channels[2]);
        cv::GaussianBlur(channels[2],channels[2],cv::Size(),3);
        cv::merge(channels, frame_eq);
        onChange(0,0);
        roi = frame_inrange(cv::Rect(0,350,640,100));
        cv::Canny(roi, roi_canny, 10, 30);

        cv::imshow("frame",frame);
        cv::imshow("frame_inrange",frame_inrange);
        cv::imshow("roi_canny",roi_canny);
        if (cv::waitKey(0) == 27) break;

    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

void onChange(int pos, void* userdata){
    cv::Scalar upper_bound(h_up,s_up,v_up);
    cv::Scalar lower_bound(h_low,s_low,v_low);
    cv::inRange(frame_eq, lower_bound, upper_bound, frame_inrange);
    cv::imshow("frame_inrange",frame_inrange);
}