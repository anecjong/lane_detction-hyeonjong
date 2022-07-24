#include <iostream>
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;



int main()
{
	VideoCapture cap("subProject.avi");

	if (!cap.isOpened())
	{
		cerr << "Image open failed!\n";

		return -1;
	}

	int col = cvRound(cap.get(CAP_PROP_FRAME_WIDTH));
	int row = cvRound(cap.get(CAP_PROP_FRAME_HEIGHT));
	
	// ����ũ �̹���
	Mat mask(480, 640, CV_8UC1, Scalar(0));
	Mat warp_mask(480, 640, CV_8UC1, Scalar(0));
	
	// �������� Point. offset : 400
	vector<Point> pts(4), warp_pts(4);
	pts[0] = Point(0, 300);
	pts[1] = Point(640, 300);
	pts[2] = Point(640, 440);
	pts[3] = Point(0, 440);

	/*
	vector<Point2f> warp_src(4), warp_dst(4);

	warp_src[0] = Point2f(40, 360);
	warp_src[1] = Point2f(600, 360);
	warp_src[2] = Point2f(640, 440);
	warp_src[3] = Point2f(0, 440);
	
	warp_dst[0] = Point2f(0, 0);
	warp_dst[2] = Point2f(320, 0);
	warp_dst[3] = Point2f(320, 240);
	warp_dst[1] = Point2f(0, 240);
	*/

	// ���ɿ��� ��� ĥ�ϱ�
	cv::fillPoly(mask, pts, Scalar(255));

	UMat frame, hls, dst, tmp, warp_img;
	vector<UMat> planes;
	cv::KalmanFilter KF();

	while (true)
	{
		cap >> frame;

		if (frame.empty())
		{
			cerr << "frame is empty!\n";
			break;
		}

		// ���� �ν�, ������ : ���̴ٿ� ��ġ�� �κ�
		Scalar m = mean(frame, mask);

		cv::add(frame, -m + Scalar(128, 128, 128), frame, mask);

		cv::cvtColor(frame, hls, COLOR_BGR2HLS);

		// ����þ� ����
		cv::GaussianBlur(hls, dst, Size(), 3.5);
		
		// HLS ä�� �и�
		cv::split(dst, planes);

		// ����ũ ����
		cv::bitwise_and(planes[1], mask, tmp);

		// ���̴� ����� 
		cv::rectangle(tmp, Point(320 - 90, 390), Point(320 + 95, 440), Scalar(255), -1);

		cv::threshold(tmp, tmp, 115, 255, THRESH_BINARY);
		
		// ��ġ ����, �����̵� �����츦 ����ϸ� �����ٵ�
		
		/*
		Mat M = cv::getPerspectiveTransform(warp_src, warp_dst);
		Mat Minv = cv::getPerspectiveTransform(warp_dst, warp_src);

		warpPerspective(tmp, warp_img, M, Size(320, 240));
		*/



		cv::imshow("tmp", tmp);
		
		if (waitKey(20) == 27)
		{
			break;
		}


	}
}
