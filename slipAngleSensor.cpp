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
	//waitKey();
}

void showDFT(Mat& frame) {
	Mat splitArray[2] = { Mat::zeros(frame.size(),CV_32F),Mat::zeros(frame.size(),CV_32F) };
	split(frame, splitArray);
	Mat dftMag;
	magnitude(splitArray[0], splitArray[1], dftMag);
	dftMag += Scalar::all(1);
	log(dftMag, dftMag);
	normalize(dftMag, dftMag, 0, 1, NORM_MINMAX);
	//recenterDft(dftMag);
	//imshow("dft", dftMag);
	//waitKey();
}

void showIDFT(Mat& frame) {
	Mat splitArray[2] = { Mat::zeros(frame.size(),CV_32F),Mat::zeros(frame.size(),CV_32F) };
	split(frame, splitArray);
	Mat dftMag;
	magnitude(splitArray[0], splitArray[1], dftMag);
	dftMag += Scalar::all(1);
	log(dftMag, dftMag);
	normalize(dftMag, dftMag, 0, 1, NORM_MINMAX);
	//recenterDft(dftMag);
	//imshow("inv_dft", dftMag);
	//waitKey();
}

void showCDFT(Mat& frame) {
	Mat splitArray[2] = { Mat::zeros(frame.size(),CV_32F),Mat::zeros(frame.size(),CV_32F) };
	split(frame, splitArray);
	Mat dftMag;
	magnitude(splitArray[0], splitArray[1], dftMag);
	dftMag += Scalar::all(1);
	log(dftMag, dftMag);
	normalize(dftMag, dftMag, 0, 1, NORM_MINMAX);
	//recenterDft(dftMag);
	//imshow("crop_dft", dftMag);
	//waitKey();

}

void imgcrop(Mat& frame) {
	//getting the centre of the image
	int crows = frame.size().width / 2;
	int ccols = frame.size().height / 2;
	int x = 30;

	//cv::Mat img(100, 100, CV_8U, cv::Scalar(255));
	frame(cv::Rect(crows - (x / 2), ccols - (x / 2), x, x)) = 0;
	//change the width and height of the image accordingly
	//Mat img = frame(Rect(crows-(x/2), ccols-(x/2), x, x));
	showCDFT(frame);
}
void imgPrep(Mat& frame) {

	Mat img;
	//Converting to 8bit integer
	//frame.assignTo(frame, CV_8U);
	//sharpening image 
	//using unsharp filter(https://homepages.inf.ed.ac.uk/rbf/HIPR2/unsharp.htm) this filter does produce poop results with noise, 
	//might have to use Laplacian of gaussion filter(dk what is it)
	//imshow("before sharp", frame);
	//cole is using this to sharpen the imgae but its not working well (https://answers.opencv.org/question/216383/how-could-do-sharpness-images/)
	/*
	Mat sharp;
	Mat sharpening_kernel = (Mat_<double>(3, 3) << -1, -1, -1,
		-1, 9, -1,
		-1, -1, -1);
	filter2D(frame, sharp, -1, sharpening_kernel);
	imshow("after sharp", frame);

	double sigma = 1, amount = 1;
	Mat blurry, sharp;
	GaussianBlur(frame, blurry, Size(), sigma);
	addWeighted(frame, 1 + amount, blurry, -amount, 0, sharp);
	Mat lowContrastMask = abs(frame - blurry) < 100000; //experiment with values
	sharp = frame * (1 + amount) + blurry * (-amount); //this is the same as addWeighted - is addWeightd obsolete??
	frame.copyTo(sharp, lowContrastMask);
	imshow("after sharp", sharp);
	*/

	
	cv::GaussianBlur(frame, frame, cv::Size(1, 1), 3);
	cv::addWeighted(frame, 1.5, frame, -0.5, 0, frame);
	
	//image thresholding
	threshold(frame, frame, 0, 255, THRESH_BINARY);
	imshow("threshold", frame);
	//morphing the image {open(erosion+dialation)+dialation}
	
	int dSize = 3; //use a number between 2-6
	Mat element = getStructuringElement(MORPH_RECT,
		Size(2 * dSize + 1, 2 * dSize + 1),
		Point(-1, -1));
	morphologyEx(frame, frame, MORPH_OPEN, element,Point(-1,-1),1);

	//dilate(frame, frame, element,Point(-1,-1),1);
}

void LineDet(Mat& frame, Mat& dst) {//https://docs.opencv.org/3.4/d9/db0/tutorial_hough_lines.html
	Mat img, cdst, cdstP,dst1;

	if (frame.empty()) {
		printf(" Error opening image\n");
		return;
	}

	//edge detection idk what affect do the values have
	frame.assignTo(frame, CV_8U);
	Canny(frame, dst, 50, 200, 3);
	Canny(frame, dst1, 50, 200, 3);
	//imshow("canny", dst);

	// Copy edges to the images that will display the results in BGR
	cvtColor(dst, cdst, COLOR_GRAY2BGR);
	cvtColor(dst1, cdst, COLOR_GRAY2BGR);
	//imshow("cvtC", cdst);
	cdstP = cdst.clone();


	vector<Vec2f> lines; // will hold the results of the detection
	HoughLines(dst, lines, 1, CV_PI / 180, 60, 0, 0); // runs the actual detection
	cout << "shits working";
	// Draw the lines
	//float rho, theta;
	//cout << "\n number of lines:" << lines.size();
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
		//cout << lines[i];
		//cout << "\nthe angles: " << theta;

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
		cout << "\nCoordinates: " << Point(l[0], l[1]) << " " << Point(l[2], l[3]);
		cout << "\nangle from P: " << angle;
#endif // DEBUG


		sum += angle;
	}
#ifdef DEBUG_PRINT
	cout << "\nsum: " << sum;
#endif // DEBUG

	cout << "\navg: " << sum / linesP.size();

	imshow("Detected Lines (in red) - Probabilistic Line Transform", cdstP);
}



int main()
{
	Mat frame = imread("road.png", IMREAD_GRAYSCALE);
	Mat frame1 = imread("road.png", IMREAD_GRAYSCALE);
	//Mat frame = imread("input.jpg", IMREAD_GRAYSCALE);
	//imshow("image", frame);
	// setting up video capture
	/*
	VideoCapture vid;
	int deviceID = 0;
	int apiID = cv::CAP_ANY;
	if (!vid.isOpened()) {
		cerr << "ERROR! Unable to open camera\n";
		//return -1;
	}
	*/
	int frameCount = 0;
	int noLineCount = 0;
	int lineCount = 0;
	//create a stack for angleHistogram and allAngles MAYBE

	//time_t start, end;
		//Defining variables to take the DFT
	Mat frameF;
	Mat frameDft;
	//Video Capture begins
	//for (;;) {
		//reading in each frame
		//vid.read(frame);
		//resizing the image ( the last 2 parameters are for scaling )
		//cv::resize(frame, frame, cv::Size(), 0.60, 0.60);
		//convert each image to grayscale for DFT to work
		//cvtColor(frame, frame, COLOR_BGR2GRAY);

		//time(&start);
	//imshow("img", frame);

	//Starting DFT

	frame.convertTo(frameF, CV_32F, 1.0 / 255.0);
	take_dft(frameF);

	showDFT(frameF);
	Mat inv;
	imgcrop(frameF);


	invDft(frameF);
	showIDFT(frameF);

	imgPrep(frameF);
	
	imshow("prepImage", frameF);
	imwrite("output.png",frameF);
	
	
	LineDet(frameF,frame1);

	//imshow("Line Image", frame1);

	frameCount++;
	
	
	waitKey();
	//time(&end);

//}

	return 1;
}

