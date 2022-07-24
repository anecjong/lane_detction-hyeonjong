#include <iostream>
#include "opencv2/opencv.hpp"
#include <vector>

cv::Mat frame, frame_hsv, frame_bin;
cv::Mat roi, roi_canny;
std::vector<cv::Mat> channels(3);

int main(void)
{
    cv::VideoCapture cap("../resources/subProject.avi");
    if (!cap.isOpened()){
        std::cout << "Video load failed!" << std::endl;
        return -1;
    }

    while (true){
        cap >> frame;
        
        if (frame.empty()){
            std::cout << "empty frame!" << std::endl;
            return -1;
        }
        // removing blue lines
        for (int y = 399; y < 401; ++y){
            for (int x = 0; x < 640*3; ++x){
                frame.at<uchar>(y,x) = 0;
            }
        }
        cv::medianBlur(frame,frame,5);
        cv::cvtColor(frame, frame_hsv,cv::COLOR_BGR2HSV);
        cv::split(frame_hsv, channels);

        // to minimize effect of patterns on floor
        cv::medianBlur(channels[2], channels[2], 5);
        cv::medianBlur(channels[2], channels[2], 5);

        // luminance adjustment to avoid bright region error
        channels[2] = (channels[2]-20)*0.5;
        channels[2] = 2*channels[2];

        // otsu algorithm
        cv::threshold(channels[2], frame_bin, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
        roi = frame_bin(cv::Rect(0,350,640,100));
        cv::Canny(roi, roi_canny, 60, 70);

        cv::imshow("roi", roi);
        cv::imshow("frame",frame);
        cv::imshow("roi_canny",roi_canny);
        if (cv::waitKey(1) == 27) break;

    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}