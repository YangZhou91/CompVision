//============================================================================
// Name        : Face.cpp
// Author      : Yang Zhou
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>
#include <stdio.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// import for LBP library

// import another LBP library


using namespace std;
using namespace cv;

void drawBoundingBox();
void drawSiftFeatures();
void drawLBPFeatures();
int computeLbpCode(unsigned char seq[9]);
int	*computeLbpHist(Mat &image, int *lbpHist);
int *extractLBPFeatures(int *outputFeatures);


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
	char* filename = new char[100];
		// to store the current input image
	Mat input;
	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");
	input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	Mat patch = input(Rect(300,200,60,60));
	cout << "patch =" << endl;
	cout << patch << endl;

	int hist[256];
	computeLbpHist(patch, hist);
//	cout << "[drawLBPFeatures] hist = " << hist[2] << endl;
	int histSize = 256;
	int *feature = new int[256];
	extractLBPFeatures(feature);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double)hist_w/histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

	for (int i = 1; i < histSize; i++){
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(hist[i-1])),
				Point( bin_w*(i), hist_h - cvRound(hist[i-1]) ),
				Scalar( 255, 0, 0), 2, 8, 0  );
		cout << "[drawLBPFeature] hist " << i -1  << " "<< feature[i-1] << endl;
	}

	  /// Display
	  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	  imshow("calcHist Demo", histImage );


	  waitKey(0);

}


//Compute an single lbp value from a pixel
int computeLbpCode(unsigned char seq[9]){


	bool bin[8] = {false};
	int base = seq[0];
	int result = 0, one = 1, final;
	// Compare each element with the center element, and update the binary value
	for(int i = 0; i < 8; i++){
		if(base >= seq[i+1]){
			bin[i] = 0;

		}
		else{
			bin[i] = 1;
		}
	}

	// Concatenate the binary number
	for(int i = 0; i < 8; i++){
//		decimal = decimal << 1 | array[i];
		result = result << 1 | bin[i];
	}
	return result;
}

//
int* computeLbpHist(Mat &image, int* lbpHist){


	unsigned char locP[9];
	// The 58 different uniform pattern
	// The 256 different pattern without uniform pattern
	for(int i = 0; i < 256; i++){
		lbpHist[i] = 0;
	}
	// for the each row and column, and avoid corners
	for(int i = 2; i < image.rows -2; i++){

		for(int j = 2; j < image.cols -2; j++){

			locP[0] = image.at<unsigned char>(i,j);
			locP[1] = image.at<unsigned char>(i-1,j);
			locP[2] = image.at<unsigned char>(i-1,j-1);
			locP[3] = image.at<unsigned char>(i, j-1);
			locP[4] = image.at<unsigned char>(i+1, j-1);
			locP[5] = image.at<unsigned char>(i+1, j);
			locP[6] = image.at<unsigned char>(i+1, j+1);
			locP[7] = image.at<unsigned char>(i, j+1);
			locP[8] = image.at<unsigned char>(i-1, j+1);
			lbpHist[computeLbpCode(locP)] ++;
		}
	}


	return lbpHist;
}

int* extractLBPFeatures(int *outputFeature){

	Mat input;
	int width = 10;
	int height = 10;
	int N = 10;

	vector<Mat> tiles;
	char *filename = new char[100];
	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");
	input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	cout<< "[extractLBPFeatures] size of image: " << input.rows << " x " << input.cols << endl;
	for(int x = 0; x < input.cols- N; x += N){
		for(int y = 0; y < input.rows - N; y += N){
			Mat tile = input(Rect(x, y, N, N));
			tiles.push_back(tile);
		}
	}

	cout << "[extractLBPFeatures] size of tiles: " << tiles.size() << endl;
	int count = 0;
	// not uniform pattern
	int hist[259];
	// For each tile, compute the histogram
	for(int i = 0; i < tiles.size(); i++){
		computeLbpHist(tiles.at(i), hist);
		for(int j = 0; j < 256; j++){
			outputFeature[j] = hist[j];
		}
	}
}

