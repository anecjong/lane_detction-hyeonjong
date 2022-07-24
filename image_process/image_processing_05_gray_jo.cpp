#include <iostream>
#include "opencv2/opencv.hpp"

cv::Mat frame, frame_gray, frame_bin;
cv::Mat roi, roi_canny;

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
        cv::cvtColor(frame, frame_gray,cv::COLOR_BGR2GRAY);

        // to minimize effect of patterns on floor
        cv::medianBlur(frame_gray, frame_gray, 5);
        cv::medianBlur(frame_gray, frame_gray, 5);

        // luminance adjustment to avoid bright region error
        frame_gray = (frame_gray - 20) * 0.5;
        frame_gray = frame_gray * 2;

        // otsu algorithm
        cv::threshold(frame_gray, frame_bin, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
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