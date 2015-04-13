//============================================================================
// Name        : Face.cpp
// Author      : Yang Zhou
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

// Get rid of errors for using sprintf
//#define _CRT_SECURE_NO_WARNINGS

#include <iostream>
#include <stdio.h>
#include "dirent.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace cv;

struct MGHData
{
	Mat image;
	string subject;
	string distance;
	int angle;
	Mat histogram;
	Rect roi;
};

Mat bulkExtractSiftFeatures(vector<MGHData> data);
Mat bulkExtractLBPFeatures(vector<MGHData> data);

int computeLbpCode(unsigned char seq[9]);
int* computeLbpHist(Mat &image, int *lbpHist);
int* computeSiftHist(Mat &image, const Mat &codeWords);
Mat extractLBPFeatures(Mat image, Mat &outputFeatures);
Mat computeCodeWords(Mat descriptors, int K);

bool MGHDataLoader(vector<MGHData> &trainingdataset, vector<MGHData> &testingdataset, vector<MGHData> &groupdataset, string directory);

int main() {

	vector<MGHData> trainingdata, testingdata, groupdata;

	MGHDataLoader(trainingdata, testingdata, groupdata, "Images/");

	Mat sift_features, lbp_features;
	cout << "Computing Sift features for training imgs..." << endl;
	sift_features = bulkExtractSiftFeatures(trainingdata);
	cout << "Computing LBP features for training imgs..." << endl;
	lbp_features = bulkExtractLBPFeatures(trainingdata);

	Mat feature_clusters;
	cout << "Computing code words for training imgs..." << endl;
	feature_clusters = computeCodeWords(sift_features, 5);

	

	//computeRecognition(Mat input_hist, vector<Mat> training_hist);

	cout << ">>>>>>>>>>>>>End of the program" << endl;
	return 0;
}

// Load training and testing images
bool MGHDataLoader(vector<MGHData> &trainingdataset, vector<MGHData> &testingdataset, vector<MGHData> &groupdataset, string directory)
{
	cout << "Loading Images" << endl;

	string trainingDir = directory + "/Training";
	string testingDir = directory + "/Testing";
	string groupDir = directory + "/Group";

	string delimiter = "_";
	string delimeterExtension = ".";
	DIR *dir;
	struct dirent *ent;

	// Training images
	if ((dir = opendir(trainingDir.c_str())) != NULL) {

		while ((ent = readdir(dir)) != NULL) {
			
			string imgname = ent->d_name;

			if (imgname.find(".jpg") != string::npos) {
				
				std::cout << "Loading " << imgname << endl;

				vector<string> tokens;
				size_t pos = 0;
				std::string token;

				while ((pos = imgname.find(delimiter)) != string::npos) {
					token = imgname.substr(0, pos);
					tokens.push_back(token);
					imgname.erase(0, pos + delimiter.length());
				}
				pos = imgname.find(delimeterExtension);
				token = imgname.substr(0, pos);
				tokens.push_back(token);

				Mat img = imread(trainingDir + "/" + ent->d_name, CV_LOAD_IMAGE_GRAYSCALE);
				MGHData data;
				data.image = img;
				data.subject = tokens[0];
				data.distance = tokens[1];
				data.angle = stoi(tokens[2]);

				trainingdataset.push_back(data);
			}

		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		cerr << "Unable to open image directory " << trainingDir << endl;
		return false;
	}

	// Testing images
	if ((dir = opendir(testingDir.c_str())) != NULL) {

		while ((ent = readdir(dir)) != NULL) {
			string imgname = ent->d_name;

			if (imgname.find(".jpg") != string::npos) {
				std::cout << "Loading " << imgname << endl;
				
				vector<string> tokens;
				size_t pos = 0;
				std::string token;

				while ((pos = imgname.find(delimiter)) != string::npos) {
					token = imgname.substr(0, pos);
					tokens.push_back(token);
					imgname.erase(0, pos + delimiter.length());
				}
				pos = imgname.find(delimeterExtension);
				token = imgname.substr(0, pos);
				tokens.push_back(token);

				Mat img = imread(testingDir + "/" + ent->d_name, CV_LOAD_IMAGE_GRAYSCALE);
				MGHData data;
				data.image = img;
				data.subject = tokens[0];
				data.distance = tokens[1];
				data.angle = stoi(tokens[2]);

				testingdataset.push_back(data);
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		cerr << "Unable to open image directory " << testingDir << endl;
		return false;
	}

	//Group data
	if ((dir = opendir(groupDir.c_str())) != NULL) {

		while ((ent = readdir(dir)) != NULL) {
			string imgname = ent->d_name;

			if (imgname.find(".jpg") != string::npos) {
				std::cout << "Loading " << imgname << endl;

				Mat img = imread(groupDir + "/" + ent->d_name, CV_LOAD_IMAGE_GRAYSCALE);
				MGHData data;
				data.image = img;
				data.subject = "group";
				data.distance = "group";
				data.angle = 0;

				groupdataset.push_back(data);
			}
		}
		closedir(dir);
	}
	else {
		/* could not open directory */
		cerr << "Unable to open image directory " << testingDir << endl;
		return false;
	}
	return true;
}

// Extract SIFT Features for image list
Mat bulkExtractSiftFeatures(vector<MGHData> data) {

	// To store keypoints
	vector<KeyPoint> keypoints;
	// To store the SIFT descriptor of current image
	cv::Mat descriptor;
	// To store all descriptor
	cv::Mat featureUnclustered;
	cv::SiftDescriptorExtractor detector;
	// Construct image name

	for (int i = 0; i < data.size(); i++) {
		MGHData tempData = data.at(i);
		Mat tempImg = tempData.image;
		Mat img_resize;
		resize(tempImg, img_resize, Size(100, 200));
		//Mat tempImg_gary;
		// convert to gray scale image
		//cvtColor(tempImg, tempImg_gary, CV_RGB2GRAY);

		// detect feature points
		detector.detect(img_resize, keypoints);
		detector.compute(img_resize, keypoints, descriptor);
		featureUnclustered.push_back(descriptor);

		// Display part
		//Mat output;
		//Mat output_resize;
		//drawKeypoints(tempImg, keypoints, output);
		//resize(output, output_resize, Size(192, 240));
		//imshow(format("SIFT_Feature_%i", i), output_resize);
		cout << "[SiftFeatures] i=" << i << endl;
	}

	computeCodeWords(featureUnclustered, 10);

	return featureUnclustered;
}

// Extract LBP Features for image list
Mat bulkExtractLBPFeatures(vector<MGHData> data){
	char* filename = new char[100];
		// to store the current input image
	Mat input;
	sprintf(filename, "/Users/Gavin/Desktop/CompVision/Training Images/1.jpg");
	input = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);

	Mat patch = input(Rect(300,200,60,60));


	int hist[256];
	computeLbpHist(patch, hist);
//	cout << "[drawLBPFeatures] hist = " << hist[2] << endl;
	int histSize = 256;
	Mat features;
	Mat featureUncluster;
	for (int i = 0; i < data.size(); i ++){
		MGHData tempData = data.at(i);
		Mat tempImg = tempData.image;

		extractLBPFeatures(tempImg, features);

		features.copyTo(featureUncluster.row(i));
	}
	
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double)hist_w/histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0,0,0));

	for (int i = 1; i < histSize; i++){
		line(histImage, Point(bin_w*(i-1), hist_h - cvRound(hist[i-1])),
				Point( bin_w*(i), hist_h - cvRound(hist[i-1]) ),
				Scalar( 255, 0, 0), 2, 8, 0  );
		//cout << "[drawLBPFeature] hist " << i -1  << " "<< feature[i-1] << endl;
	}

	  /// Display
	  namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE );
	  imshow("calcHist Demo", histImage );

	  return featureUncluster;

	  waitKey(0);

}

// Compute an single lbp value from a pixel
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

// Compute LBP histogram for given image
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

// Compute SIFT histogram for given image
int* computeSiftHist(MGHData data, const Mat &codeWords)
{
	int* hist;
	// extract features from image
	
	// find nearest code word

	// return histogram
	return hist;
}

// Extract LBP Features for given image
Mat extractLBPFeatures(Mat image, Mat &outputFeature){

	Mat input;
	int width = 10;
	int height = 10;
	int N = 10;

	vector<Mat> tiles;
	
	image.copyTo(input);

	for(int x = 0; x < input.cols- N; x += N){
		for(int y = 0; y < input.rows - N; y += N){
			Mat tile = input(Rect(x, y, N, N));
			tiles.push_back(tile);
		}
	}

	cout << "[extractedLBPFeatures] size of tiles: " << tiles.size() << endl;
	int row = tiles.size();
	// not uniform pattern
	int hist[256];
	Mat histMat;
	// For each tile, compute the histogram
	for(int i = 0; i < tiles.size(); i++){
		computeLbpHist(tiles.at(i), hist);
		Mat temp = Mat(1, 256, CV_64F, hist);
		temp.copyTo(histMat.row(i));
	}

	return histMat;
}

// Compute code words for given descriptors
Mat computeCodeWords(Mat descriptors, int K){
	// Change to K later on
	int clusterCount = 10;
	Mat labels;
	Mat centers;
	TermCriteria criteria{ TermCriteria::COUNT, 100, 1 };

	kmeans(descriptors, clusterCount, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);

	cout << "[computedCodeWords] The size of centers: " << centers.rows << " x " << centers.cols << endl;

	return centers;
}

// Return closest subject match
string computeRecognition(Mat input_hist, vector<Mat> training_hist)
{
	string closest_subject = "None";



	return closest_subject;
}