#include "image_process.h"

cv::Mat binarization(cv::Mat frame){
    // frame: cv_8uc3

    cv::Mat frame_hsv, frame_bin;
    std::vector<cv::Mat> channels(3);

    // removing blue lines
    for (int y = 399; y < 401; ++y){
        for (int x = 0; x < 640*3; ++x){
            frame.at<uchar>(y,x) = 0;
        }
    }
    cv::medianBlur(frame,frame,5);

    //Adjustment
    cv::Mat mask(480,640,CV_8UC1);
    mask.setTo(0);
    std::vector<cv::Point> mask_pts;
    mask_pts.push_back(cv::Point(0,300));
    mask_pts.push_back(cv::Point(640,300));
    mask_pts.push_back(cv::Point(640,440));
    mask_pts.push_back(cv::Point(0,440));
    cv::fillPoly(mask, mask_pts, cv::Scalar(255));

    auto m = cv::mean(frame, mask);
    cv::add(frame, cv::Scalar(70-(int)m[0], 70-(int)m[1], 70-(int)m[2]), frame, mask);


    // bgr 2 hsv
    cv::cvtColor(frame, frame_hsv,cv::COLOR_BGR2HSV);
    cv::split(frame_hsv, channels);

    // to minimize effect of patterns on floor
    cv::medianBlur(channels[2], channels[2], 5);
    cv::medianBlur(channels[2], channels[2], 5);
    cv::GaussianBlur(channels[2], channels[2], cv::Size(), 5);

    // luminance adjustment to avoid bright region error
    //channels[2] = channels[2] + (cv::Scalar(128) - cv::mean(channels[2]));
    //channels[2] = (channels[2]-20)*0.5;
    //channels[2] = 2.0*channels[2];

    // otsu algorithm
    //cv::threshold(channels[2], frame_bin, 0, 255, cv::THRESH_OTSU | cv::THRESH_BINARY);
    
    // range
    cv::threshold(channels[2], frame_bin, 56, 255, cv::THRESH_BINARY_INV);

    return frame_bin;
}

cv::Mat hough_detect(cv::Mat frame_bin_warp){
    std::vector<cv::Vec4i> lines;
    cv::Mat canny;
    int lx=0, ly=0, rx=0, ry=0;
    double rd=0, ld=0, m, b;
    double rx_mean=0,ry_mean=0,lx_mean=0,ly_mean=0;
    int lcount=0, rcount=0;

    cv::Mat line_board(frame_bin_warp.rows, frame_bin_warp.cols, CV_8UC1);
    line_board.setTo(0);
    cv::Canny(frame_bin_warp,canny,10,70);
    cv::HoughLinesP(canny, lines, 1, CV_PI/360, 50, 40, 10);

    for( int i = 0; i < lines.size(); i++ ){
        if (lines[i][0]-lines[i][2] == 0) continue;
        else m = (0.0 + lines[i][1] - lines[i][3]) / (lines[i][0] - lines[i][2]);

        if ( 0.25 < std::fabs(m) && std::fabs(m) < 1000){
            if (m > 0 && (lines[i][2] > 290) && (lines[i][0] > 290 - 10)){
                rx += (lines[i][0] + lines[i][2])/2;
                ry += (lines[i][1] + lines[i][3])/2;
                rd += m;
                rcount +=1;
            }
            else if (m < 0 && (lines[i][0] < 290) && (lines[i][2] < 290 + 10)){
                lx += (lines[i][0] + lines[i][2])/2;
                ly += (lines[i][1] + lines[i][3])/2;
                ld += m;
                lcount +=1;
            }
        }
    }

    if (rcount!=0){
        rx_mean = ((double)rx)/rcount;
        ry_mean = ((double)ry)/rcount;

        m = rd/rcount;
        b = -m * rx_mean + ry_mean;

        int pt1_x, pt2_x, pt1_y, pt2_y;
        pt1_y = 0;
        pt2_y = line_board.rows;
        pt1_x = ((double)pt1_y - b) / m;
        pt2_x = ((double)pt2_y - b) / m;
        cv::line( line_board, cv::Point(pt1_x, pt1_y), cv::Point(pt2_x,pt2_y), cv::Scalar(255),80);
    }
    if (lcount!=0){
        lx_mean = ((double)lx)/lcount;
        ly_mean = ((double)ly)/lcount;

        m = ld/lcount;
        b = -m * lx_mean + ly_mean;

        int pt1_x, pt2_x, pt1_y, pt2_y;
        pt1_y = 0;
        pt2_y = line_board.rows;
        pt1_x = ((double)pt1_y - b) / m;
        pt2_x = ((double)pt2_y - b) / m;
        cv::line( line_board, cv::Point(pt1_x, pt1_y), cv::Point(pt2_x,pt2_y), cv::Scalar(255),80);
    }

    return line_board;
}