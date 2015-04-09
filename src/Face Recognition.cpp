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


using namespace std;

int main() {

	// to store the input file names
	char * filename = new char[100];

	// to store the current input image
	cv::Mat input;

	// To store keypoints
	vector<cv::KeyPoint> keypints;
	// To store the SIFT descriptor of current image
	cv::Mat descriptor;
	// To store all descriptor
	cv::Mat featureUnclustered;

	cv::SiftDescriptorExtractor detector;

	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");
	input = cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	cv::imshow("Test", input);
	cv::waitKey(0);

	cout << "The end of the program" << endl; // prints !!!Hello World!!!
	return 0;
}
