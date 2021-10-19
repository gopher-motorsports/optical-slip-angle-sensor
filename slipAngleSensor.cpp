#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include<time.h>

using namespace cv;
using namespace std;

const int fps = 20;
//#define DEBUG_PRINT
void recenterDft(Mat& frame) {
	int centerX = frame.cols / 2;
	int centerY = frame.rows / 2;
	Mat q1(frame, Rect(0, 0, centerX, centerY));
	Mat q2(frame, Rect(centerX, 0, centerX, centerY));
	Mat q3(frame, Rect(0, centerY, centerX, centerY));
	Mat q4(frame, Rect(centerX, centerY, centerX, centerY));

	Mat swapMap;

	q1.copyTo(swapMap);
	q4.copyTo(q1);
	swapMap.copyTo(q4);

	q2.copyTo(swapMap);
	q3.copyTo(q2);
	swapMap.copyTo(q3);
}
void take_dft(Mat& frame) {
	//setting up DFT
	Mat frame_dft;
	Mat frameComplex[2] = { frame,Mat::zeros(frame.size(),CV_32F) };
	merge(frameComplex, 2, frame_dft);
	Mat fdft;
	//taking dft
	dft(frame_dft, frame, DFT_COMPLEX_INPUT);
	recenterDft(frame);
}


void invDft(Mat& frame) {
	Mat inv;
	dft(frame, frame, DFT_INVERSE | DFT_REAL_OUTPUT | DFT_SCALE);
}

void showDFT(Mat& frame) {
	Mat splitArray[2] = { Mat::zeros(frame.size(),CV_32F),Mat::zeros(frame.size(),CV_32F) };
	split(frame, splitArray);
	Mat dftMag;
	magnitude(splitArray[0], splitArray[1], dftMag);
	dftMag += Scalar::all(1);
	log(dftMag, dftMag);
	normalize(dftMag, dftMag, 0, 1, NORM_MINMAX);
}

void showIDFT(Mat& frame) {
	Mat splitArray[2] = { Mat::zeros(frame.size(),CV_32F),Mat::zeros(frame.size(),CV_32F) };
	split(frame, splitArray);
	Mat dftMag;
	magnitude(splitArray[0], splitArray[1], dftMag);
	dftMag += Scalar::all(1);
	log(dftMag, dftMag);
	normalize(dftMag, dftMag, 0, 1, NORM_MINMAX);
	imshow("inv_dft", dftMag);
}

void showCDFT(Mat& frame) {
	Mat splitArray[2] = { Mat::zeros(frame.size(),CV_32F),Mat::zeros(frame.size(),CV_32F) };
	split(frame, splitArray);
	Mat dftMag;
	magnitude(splitArray[0], splitArray[1], dftMag);
	dftMag += Scalar::all(1);
	log(dftMag, dftMag);
	normalize(dftMag, dftMag, 0, 1, NORM_MINMAX);
}

void imgcrop(Mat& frame) {
	//getting the centre of the image
	int crows = frame.size().width / 2;
	int ccols = frame.size().height / 2;
	int x = 30;

	frame(cv::Rect(crows - (x / 2), ccols - (x / 2), x, x)) = 0;
	showCDFT(frame);
}
void imgPrep(Mat& frame) {

	Mat img;
	if (frame.empty()) {
		printf(" Error opening image\n");
		return;
	}
	//frame.assignTo(frame, CV_8U);
	//cv::GaussianBlur(frame, frame, cv::Size(1, 1), 3);
	//cv::addWeighted(frame, 1.5, frame, -0.5, 0, frame);
	//cvtColor(frame, frame, COLOR_BGR2GRAY);
	imshow("blurrssss ", frame);
	//cout << frame;;

	int dSize = 1; //use a number between 2-6
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dSize + 1, 2 * dSize + 1),
		Point(-1, -1));
	morphologyEx(frame, frame, MORPH_OPEN, element, Point(-1, -1), 1);
	imshow("morph", frame);
	//cout << "frameeeeeeeee" << frame <<endl;
	//image thresholding
	cv::absdiff(frame, cv::Scalar::all(0), frame);
	imshow("aasdfa",frame);
	//cout << frame;
	threshold(frame, frame, 0.1, 1.0, THRESH_BINARY); //asdadsdsdafsadfsdfsafasfasfsdfsadfasdfasfdasffasdfasdf
	imshow("threshold", frame);
	//morphing the image {open(erosion+dialation)+dialation}

	//int dSize = 3; //use a number between 2-6   ##################################################### need to tweak this 
	//Mat element = getStructuringElement(MORPH_RECT,
	//	Size(2 * dSize + 1, 2 * dSize + 1),
	//	Point(-1, -1));
	//morphologyEx(frame, frame, MORPH_OPEN, element, Point(-1, -1), 1);

	imshow("prepImage", frame);
	//frame.assignTo(frame, CV_32F);
}

string type2str(int type) {
	string r;
	uchar depth = type & CV_MAT_DEPTH_MASK;
	uchar chans = 1 + (type >> CV_CN_SHIFT);
	switch (depth) {
	case CV_8U:  r = "8U"; break;
	case CV_8S:  r = "8S"; break;
	case CV_16U: r = "16U"; break;
	case CV_16S: r = "16S"; break;
	case CV_32S: r = "32S"; break;
	case CV_32F: r = "32F"; break;
	case CV_64F: r = "64F"; break;
	default:     r = "User"; break;
	}
	r += "C";
	r += (chans + '0');
	return r;
}

void LineDet(Mat& frame, Mat& dst) {//https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
	Mat img, cdst, cdstP, dst1;

	if (frame.empty()) {
		printf(" Error opening image\n");
		return;
	}
	//edge detection idk what affect do the values have
	//imshow("linedet111111111111111111111", frame);
	string ty = type2str(frame.type());
	printf("Matrix: %s %dx%d \n", ty.c_str(), frame.cols, frame.rows);
	frame.assignTo(frame, CV_8U); //############################### the issueeeeeeeeeeeeeee
	//imshow("linedet22222222222222222", frame);
	Canny(frame, dst, 50, 200, 3);
	Canny(frame, dst1, 50, 200, 3);
	//imshow("canny", dst1);

	// Copy edges to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cvtColor(dst1, cdst, COLOR_GRAY2BGR);
	//imshow("cvtC", cdst);
	cdstP = cdst.clone();

	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst, lines, 1, CV_PI / 180, 60, 0, 0); // runs the actual detection
	cout << "shits working";
	
	// Draw the lines
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1];
		Point pt1, pt2;
		double a = cos(theta), b = sin(theta);
		double x0 = a * rho, y0 = b * rho;
		pt1.x = cvRound(x0 + 1000 * (-b));
		pt1.y = cvRound(y0 + 1000 * (a));
		pt2.x = cvRound(x0 - 1000 * (-b));
		pt2.y = cvRound(y0 - 1000 * (a));
		line(cdst, pt1, pt2, Scalar(0, 0, 255), 3, LINE_AA);

	}
	imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst);

	vector<Vec4i> linesP; // will hold the results of the detection
	HoughLinesP(dst1, linesP, 1, CV_PI / 180, 50, 50, 10); // runs the actual detection
	// Draw the lines
	cout << "\n number of lines:" << linesP.size();
	float sum = 0;
	for (size_t i = 0; i < linesP.size(); i++)
	{
		Vec4i l = linesP[i];
		line(cdstP, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 3, LINE_AA);
		float angle = atan2((l[3] - l[1]), (l[2] - l[0])) * 180 / CV_PI;

		if (angle < 0) {
			angle += 180;
		}
		angle -= 90;
#ifdef DEBUG_PRINT
		//cout << "\nCoordinates: " << Point(l[0], l[1]) << " " << Point(l[2], l[3]);
		//cout << "\nangle from P: " << angle;
#endif // DEBUG


		sum += angle;
	}
#ifdef DEBUG_PRINT
	//cout << "\nsum: " << sum;
#endif // DEBUG

	//cout << "\navg: " << sum / linesP.size();

	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
}


int main()
{
	// setting up video capture

	VideoCapture vid("GolfCartRun.mp4");
	if (!vid.isOpened()) {
		cout << "Error opening video stream or file" << endl;
		return -1;
	}

	int frameCount=0;
	int empty_frame_count = 0;
	for (;;)
	{
		Mat frame,frame1;
		vid >> frame; // get a new frame from camera
		frame.copyTo(frame1);
		if (frame.empty())
		{
			empty_frame_count++;
			if (empty_frame_count > 20) break;
			frame = Mat::zeros(480, 640, CV_8UC3);
			waitKey(100);
		}
		else if (frameCount > 5) {

			Mat frameF;
			cvtColor(frame, frame, COLOR_BGR2GRAY);
			frame.convertTo(frameF, CV_32F, 1.0 / 255.0);
			take_dft(frameF);
			showDFT(frameF);
			imgcrop(frameF);
			invDft(frameF);
			showIDFT(frameF);


			imgPrep(frameF);
			imshow("prepImage", frameF);
			imwrite("output.png", frameF);

			LineDet(frameF, frame1);

			imshow("Line Image", frame1);

			frameCount++;
		}
		frameCount++;
		if (waitKey(30) >= 0) break;

	}
	return 1;
}
