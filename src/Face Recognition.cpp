//============================================================================
// Name        : Face.cpp
// Author      : Yang Zhou
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// import for LBP library
#include "lbp.hpp"
#include "histogram.hpp"

using namespace std;
using namespace cv;

void drawBoundingBox();
void drawSiftFeatures();
void drawLBPFeatures();


int main() {

	// to store the input file names
//	drawSiftFeatures();
//	drawBoundingBox();
	drawLBPFeatures();
	cout << "The end of the program" << endl; // prints !!!Hello World!!!
	return 0;
}

void drawBoundingBox(){

	char * filename = new char[100];
	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");

	Mat input;
	Mat input_gray;
	RNG rng(12345);

	input = imread(filename);

	if(!input.empty()){
		// Convert RGB image to Gray Scale
		cout << "[drawBoundingBox] Converting the original image to gray scale" << endl;
		cvtColor(input, input_gray,CV_BGR2GRAY);
		blur(input_gray, input_gray, Size(3,3));
	}
	else{
		cout << "[drawBoundingBox] Could not read the image" << endl;
	}

	Mat threshold_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;


 	int thresh = 100;

	// Detect edges using Threshold
	threshold(input_gray, threshold_output, thresh, 255, THRESH_BINARY);
	// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Approximate contours to polygons
	vector<vector<Point> > contours_poly(contours.size());
	vector<Rect> boundRect(contours.size());
	vector<Point2f> center(contours.size());
	vector<float> radius(contours.size());

	for(int i = 0; i < contours.size(); i++){
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);
		boundRect[i] = boundingRect(Mat(contours_poly[i]));
		minEnclosingCircle((Mat)contours_poly[i], center[i], radius[i]);
	}

	// Draw polygon contour + bonding rects
	Mat drawing = Mat::zeros(threshold_output.size(), CV_8UC3);
	for(int i =0; i < contours.size(); i++){
		Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
		drawContours( drawing, contours_poly, i, color, 1, 8, vector<Vec4i>(), 0, Point() );
		rectangle( drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0 );
//		circle( drawing, center[i], (int)radius[i], color, 2, 8, 0 );
	}
	/// Show in a window
	  namedWindow( "Contours", CV_WINDOW_AUTOSIZE );
	  imshow( "Contours", drawing );
	  waitKey(0);
}

void drawSiftFeatures() {
	// to store the input file names
	char* filename = new char[100];
	// to store the current input image
	Mat input;
	// To store keypoints
	vector<KeyPoint> keypoints;
	// To store the SIFT descriptor of current image
	cv::Mat descriptor;
	// To store all descriptor
	cv::Mat featureUnclustered;
	cv::SiftDescriptorExtractor detector;
	// Construct image name
	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");
	input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	// detect feature points
	detector.detect(input, keypoints);
	detector.compute(input, keypoints, descriptor);
	featureUnclustered.push_back(descriptor);
	int dictionarySize = 200;
	cv::TermCriteria tc(CV_TERMCRIT_ITER, 100, 0.001);
	// number of retries
	int retries = 1;
	// K means center initialization
	int flag = cv::KMEANS_PP_CENTERS;
	// Create the BoW trainer
	cv::BOWKMeansTrainer bowTrainer(dictionarySize, tc, retries, flag);
	// Cluster feature vectors
	cv::Mat dictonary = bowTrainer.cluster(featureUnclustered);
	// store the vocabulary
	cv::FileStorage fs("dictionary.yml", cv::FileStorage::WRITE);
	fs << "vocabulart" << dictonary;
	fs.release();
	//	imshow("Test", input);
	Mat output;
	Mat output_resize;
	drawKeypoints(input, keypoints, output);
	resize(output, output_resize, Size(192,240));
	imshow("SIFT_Feature", output_resize);
	waitKey(0);
}

void drawLBPFeatures(){

	Mat input;
	// The size of small window
	int N = 100;
	vector<Mat> tiles;
	char *filename = new char[100];
	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");
	input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
//	imshow("input", input);
//	waitKey(0);

	cout << "[drawLBPFeature] The size of input " << input.rows << " x " << input.cols << endl;
	for(int x = 0; x < input.cols- N; x += N){

		for(int y = 0; y < input.rows - N; y += N){
			Mat tile = input(Rect(x, y, N, N));
			tiles.push_back(tile);
		}
	}

	imshow("tile", tiles.at(25));

	cout << "[drawLBPFeatures] the size of tiles array: " << tiles.size() << endl;
	waitKey(0);

}
