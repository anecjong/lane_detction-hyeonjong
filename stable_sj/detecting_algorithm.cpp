#include "detecting_algorithm.h"

std::vector<int> black_val(cv::Mat frame_bin, int offset){
    std::vector<int> blacks;
    blacks.reserve(10);
    int prev = (int)frame_bin.at<uchar>(offset,0);

    if (prev == 0){
        blacks.push_back(0);
    }

    for (int i = 1; i < frame_bin.cols; ++i){
        if (prev!=(int)frame_bin.at<uchar>(offset,i)){
            blacks.push_back(i);
            prev = (int)frame_bin.at<uchar>(offset,i);
        }
    }

    if ((int)frame_bin.at<uchar>(offset,639)==0 && *(blacks.end()-1)!=639){
        blacks.push_back(639);
    }

    return blacks;
}

std::vector<int> blacks_from_inside(cv::Mat frame_bin, int offset){
    std::vector<int> blacks;
    blacks.reserve(4);

    int l_l=0, l_r=0, r_l=640, r_r=640;
    int l_last = -1, r_last = -1;
    for (int i = 290; i > 0; --i){
        int this_val = (int)frame_bin.at<uchar>(offset,i);
        if (this_val == 0){
            if (l_r==0) l_r = i;
            l_last = i;
        }
    }
    if (l_last != -1) l_l = l_last;

    for (int i = 290; i < 640; ++i){
        int this_val = (int)frame_bin.at<uchar>(offset,i);
        if (this_val == 0){
            if (r_l==640) r_l = i;
            r_last = i;
        }
    }
    if (r_last != -1) r_r = r_last;

    blacks.push_back(l_l);
    blacks.push_back(l_r);
    blacks.push_back(r_l);
    blacks.push_back(r_r);

    return blacks;
}
