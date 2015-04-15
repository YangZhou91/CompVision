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
#include <string>
#include <map>
#include "dirent.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <opencv2/nonfree/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv2/objdetect/objdetect.hpp"

using namespace std;
using namespace cv;

struct MGHData
{
	Mat image;
	string subject;
	string distance;
	int angle;
	Mat sift_histogram;
	Mat lbp_histogram;
	Rect roi;
};

Mat bulkExtractSiftFeatures(vector<MGHData> data);
vector<Mat> bulkExtractLBPFeatures(vector<MGHData> data);

// For the box selection
//bool got_roi = false;
//Point pt1, pt2;
//Mat cap_img;

// Feature Extraction method
int computeLbpCode(unsigned char seq[9]);
int* computeLbpHist(Mat &image, int *lbpHist);
void computeSiftCodewordHist(MGHData &data, const Mat &codeWords, Mat features);
void computeLBPCodewordHist(MGHData &data, const Mat &codeWords, Mat features);
void computeCodewordHist(MGHData &data, const Mat &codeWords, Mat features, string feature_mode);
Mat extractLBPFeatures(Mat image, Mat &outputFeatures);
Mat extractLBPFeatures(Mat image);
Mat extractSiftFeatures(Mat image);
Mat computeCodeWords(Mat descriptors, int K);
MGHData computeSiftClassification(MGHData testingdata, vector<MGHData> trainingdata);
MGHData computeLBPClassification(MGHData testingdata, vector<MGHData> trainingdata);
MGHData computeClassification(MGHData testingdata, vector<MGHData> trainingdata, string feature_mode, int compare_method);
double computeRecognitionRate(vector<MGHData> testingdata, vector<MGHData> trainingdata, vector<Mat> testing_features, vector<Mat> training_features, const Mat codewords, Mat &confusion_matrix, string feature_mode);

Mat detectAndLabel(Mat frame, vector<MGHData> trainingdata, const Mat &codewords, vector<Mat> training_features, string feature_method);

/** Global variables */
String face_cascade_name = "haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;

// Helper methods
Mat getROI(MGHData data);
void drawSIFTImage(int i, vector<KeyPoint> keypoints, Mat input);
void drawLBPHistogram();

void drawSIFTCodeword();
void drawLBPCodeword();

// Image loader part
bool MGHDataLoader(vector<MGHData> &trainingdataset, vector<MGHData> &testingdataset, vector<MGHData> &groupdataset, string directory);
void mouse_click(int event, int x, int y, int flags, void *param);

vector<MGHData> trainingdata, testingdata, groupdata;

int main() 
{

	cout << "Loading Images..." << endl;
	MGHDataLoader(trainingdata, testingdata, groupdata, "Images/");

	vector<Mat> sift_features_training, sift_features_testing;
	vector<Mat> lbp_features_training, lbp_features_testing;

	/* PART 2 */
	cout << "Computing Sift features for training images..." << endl;
	for (int i = 0; i < trainingdata.size(); i++)
		sift_features_training.push_back(extractSiftFeatures(getROI(trainingdata.at(i))));

	cout << "Computing LBP features for training images..." << endl;
	for (int i = 0; i < trainingdata.size(); i++)
		lbp_features_training.push_back(extractLBPFeatures(getROI(trainingdata.at(i))));

	map<int, Mat> sift_feature_clusters, lbp_feature_clusters;

	cout << "Computing SIFT code words for training images..." << endl;
	// create one big matrix to contain all SIFT features from all training images
	Mat sift_features_training_mat;
	for (int i = 0; i < sift_features_training.size(); i++)
		sift_features_training_mat.push_back(sift_features_training.at(i));

	sift_feature_clusters.insert(pair<int, Mat>(5, computeCodeWords(sift_features_training_mat, 5)));
	sift_feature_clusters.insert(pair<int, Mat>(10, computeCodeWords(sift_features_training_mat, 15)));
	sift_feature_clusters.insert(pair<int, Mat>(20, computeCodeWords(sift_features_training_mat, 20)));

	// update histogram field in each training data 
	for (int i = 0; i < trainingdata.size(); i++)
		computeSiftCodewordHist(trainingdata[i], sift_feature_clusters[5], sift_features_training[i]);

	cout << "Computing LBP code words for training images..." << endl;
	// create one big matrix to contain all LBP features from all training images
	Mat lbp_features_training_mat;
	for (int i = 0; i < lbp_features_training.size(); i++){
		lbp_features_training_mat.push_back(lbp_features_training.at(i));
	}

	cout << "[Main] lbp_features_mat: " << lbp_features_training_mat.rows << " x " << lbp_features_training_mat.cols << endl;
	
	lbp_feature_clusters.insert(pair<int, Mat>(5, computeCodeWords(lbp_features_training_mat, 5)));
	lbp_feature_clusters.insert(pair<int, Mat>(10, computeCodeWords(lbp_features_training_mat, 15)));
	lbp_feature_clusters.insert(pair<int, Mat>(20, computeCodeWords(lbp_features_training_mat, 20)));

	// update histogram field in each training data 
	for (int i = 0; i < trainingdata.size(); i++)
		computeLBPCodewordHist(trainingdata[i], lbp_feature_clusters[5], lbp_features_training[i]);

	/* PART 3 */
	cout << "Computing Sift features for testing images..." << endl;
	for (int i = 0; i < testingdata.size(); i++)
		sift_features_testing.push_back(extractSiftFeatures(getROI(testingdata.at(i))));

	cout << "Computing LBP features for testing images..." << endl;
	for (int i = 0; i < testingdata.size(); i++)
		lbp_features_testing.push_back(extractLBPFeatures(getROI(testingdata.at(i))));

	for (int number_of_clusters = 5; number_of_clusters <= 20; number_of_clusters *= 2)
	{
		cout << "Number of clusters = " << number_of_clusters << endl;

	//bulkExtractSiftFeatures(trainingdata);
	//bulkExtractLBPFeatures(trainingdata);

	//drawLBPHistogram();

		drawSIFTCodeword();
		drawLBPCodeword();

		double sift_testing_recognition_performance, lbp_testing_recognition_performance, sift_training_recognition_performance, lbp_training_recognition_performance;
		Mat sift_testing_confusion_matrix, lbp_testing_confusion_matrix, sift_training_confusion_matrix, lbp_training_confusion_matrix;
		string sift = "sift";
		string lbp = "lbp";

		// evaluate testing and training data with SIFT

		// update histogram field in each training data 
		for (int i = 0; i < trainingdata.size(); i++)
			computeCodewordHist(trainingdata[i], sift_feature_clusters[number_of_clusters], sift_features_training[i], sift);

		sift_testing_recognition_performance = computeRecognitionRate(testingdata, trainingdata, sift_features_testing, sift_features_training, sift_feature_clusters[number_of_clusters], sift_testing_confusion_matrix, sift);
		cout << "SIFT recognition performance [Testing] = " << sift_testing_recognition_performance << "%" << " (k=" << number_of_clusters << ")" << endl;
		cout << "SIFT Confusion matrix [Testing] = " << endl << " " << sift_testing_confusion_matrix << endl << endl;

		sift_training_recognition_performance = computeRecognitionRate(trainingdata, trainingdata, sift_features_training, sift_features_training, sift_feature_clusters[number_of_clusters], sift_training_confusion_matrix, sift);
		cout << "SIFT recognition performance [Training] = " << sift_training_recognition_performance << "%" << " (k=" << number_of_clusters << ")" << endl;
		cout << "SIFT Confusion matrix [Training] = " << endl << " " << sift_training_confusion_matrix << endl << endl;

		// update histogram field in each training data 
		for (int i = 0; i < trainingdata.size(); i++)
			computeCodewordHist(trainingdata[i], lbp_feature_clusters[number_of_clusters], lbp_features_training[i], lbp);

		// evaluate testing and training data with LBP

		lbp_testing_recognition_performance = computeRecognitionRate(testingdata, trainingdata, lbp_features_testing, lbp_features_training, lbp_feature_clusters[number_of_clusters], lbp_testing_confusion_matrix, lbp);
		cout << "LBP recognition performance [Testing] = " << lbp_testing_recognition_performance << "%" << " (k=" << number_of_clusters << ")" << endl;
		cout << "LBP Confusion matrix [Testing] = " << endl << " " << lbp_testing_confusion_matrix << endl << endl;

		lbp_training_recognition_performance = computeRecognitionRate(trainingdata, trainingdata, lbp_features_training, lbp_features_training, lbp_feature_clusters[number_of_clusters], lbp_training_confusion_matrix, lbp);
		cout << "LBP recognition performance [Training] = " << lbp_training_recognition_performance << "%" << " (k=" << number_of_clusters << ")" << endl;
		cout << "LBP Confusion matrix [Training] = " << endl << " " << lbp_training_confusion_matrix << endl << endl;
	}

	/* PART 4 */
	for (int i = 0; i < groupdata.size(); i++)
	{
		imshow("group", groupdata[i].image);
		waitKey(0);

		if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading face cascade\n"); return -1; };

		detectAndLabel(groupdata[i].image, trainingdata, sift_feature_clusters[10], sift_features_training, "sift");
		waitKey(0);
	}
	cout << ">>>>>>>>>>>>>DONE." << endl;
	getchar();
	return 0;
}

// Load training and testing images
bool MGHDataLoader(vector<MGHData> &trainingdataset, vector<MGHData> &testingdataset, vector<MGHData> &groupdataset, string directory)
{
	string trainingDir = directory + "/Training";
	string testingDir = directory + "/Testing";
	string groupDir = directory + "/Group";

	string delimiter = "_";
	string delimeterExtension = ".";
	DIR *dir;
	struct dirent *ent;
	
	// Training images
	if ((dir = opendir(trainingDir.c_str())) != NULL) 
	{
		while ((ent = readdir(dir)) != NULL) 
		{
			string imgname = ent->d_name;
			if (imgname.find(".jpg") != string::npos)
			{
				cout << "Loading " << imgname << endl;
				vector<string> tokens;
				size_t pos = 0;
				std::string token;

				while ((pos = imgname.find(delimiter)) != string::npos) 
				{
					token = imgname.substr(0, pos);
					tokens.push_back(token);
					imgname.erase(0, pos + delimiter.length());
				}
				pos = imgname.find(delimeterExtension);
				token = imgname.substr(0, pos);
				tokens.push_back(token);

				Mat img = imread(trainingDir + "/" + ent->d_name, CV_LOAD_IMAGE_GRAYSCALE);

				Point point1(stoi(tokens[3]), stoi(tokens[4]));
				Point point2(stoi(tokens[5]), stoi(tokens[6]));
				MGHData data;
				data.image = img;
				data.subject = tokens[0];
				data.distance = tokens[1];
				data.angle = stoi(tokens[2]);
				data.roi = Rect(point1, point2);

				trainingdataset.push_back(data);
			}
		}
		closedir(dir);
	}
	else 
	{
		cerr << "Unable to open image directory " << trainingDir << endl;
		return false;
	}
	
	// Testing images
	if ((dir = opendir(testingDir.c_str())) != NULL) 
	{
		while ((ent = readdir(dir)) != NULL) 
		{
			string imgname = ent->d_name;

			if (imgname.find(".jpg") != string::npos) 
			{
				cout << "Loading " << imgname << endl;
				vector<string> tokens;
				size_t pos = 0;
				std::string token;

				while ((pos = imgname.find(delimiter)) != string::npos) 
				{
					token = imgname.substr(0, pos);
					tokens.push_back(token);
					imgname.erase(0, pos + delimiter.length());
				}
				pos = imgname.find(delimeterExtension);
				token = imgname.substr(0, pos);
				tokens.push_back(token);

				Mat img = imread(testingDir + "/" + ent->d_name, CV_LOAD_IMAGE_GRAYSCALE);

				Point point1(stoi(tokens[4]), stoi(tokens[5]));
				Point point2(stoi(tokens[6]), stoi(tokens[7]));

				MGHData data;
				data.image = img;
				data.subject = tokens[0];
				data.distance = tokens[1];
				data.angle = stoi(tokens[2]);
				data.roi = Rect(point1, point2);

				testingdataset.push_back(data);
			}
		}
		closedir(dir);
	}
	else 
	{
		cerr << "Unable to open image directory " << testingDir << endl;
		return false;
	}

	//Group data
	if ((dir = opendir(groupDir.c_str())) != NULL) 
	{
		while ((ent = readdir(dir)) != NULL) 
		{
			string imgname = ent->d_name;
			if (imgname.find(".jpg") != string::npos) 
			{
				cout << "Loading " << imgname << endl;

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
	else 
	{
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
		Mat tempImg = getROI(tempData);

		//Mat tempImg_gary;
		// convert to gray scale image
		//cvtColor(tempImg, tempImg_gary, CV_RGB2GRAY);

		// detect feature points
		detector.detect(tempImg, keypoints);
		detector.compute(tempImg, keypoints, descriptor);
		featureUnclustered.push_back(descriptor);

		// For the first three image
		if (i < 3){
			drawSIFTImage(i, keypoints, tempImg);
			
		}

	}
	waitKey(0);

	computeCodeWords(featureUnclustered, 10);

	return featureUnclustered;
}

// Extract LBP Features for image list
vector<Mat> bulkExtractLBPFeatures(vector<MGHData> data){
	char* filename = new char[100];
	// to store the current input image
	Mat input;

	input = data.at(1).image;
	int height = input.rows;
	int width = input.cols;
	int N = 10;
	Mat patch = input(Rect(300, 200, 60, 60));

	int hist[256];
	computeLbpHist(patch, hist);
	//	cout << "[drawLBPFeatures] hist = " << hist[2] << endl;
	int histSize = 256;

	int numOfRow = data.size()*width*height / N / N;
	cout << "[bulkExtractLBPFeatures] num of Rows: " << numOfRow << endl;
	Mat features;
	vector <Mat> featureUncluster;
	for (int i = 0; i < data.size(); i++){
		MGHData tempData = data.at(i);
		Mat tempImg = getROI(tempData);

		extractLBPFeatures(tempImg, features);

		featureUncluster.push_back(features);
	}


	return featureUncluster;

	waitKey(0);
}

// Compute an single lbp value from a pixel
int computeLbpCode(unsigned char seq[9]){
	bool bin[8] = { false };
	int base = seq[0];
	int result = 0, one = 1, final;
	// Compare each element with the center element, and update the binary value
	for (int i = 0; i < 8; i++){
		if (base >= seq[i + 1]){
			bin[i] = 0;
		}
		else{
			bin[i] = 1;
		}
	}

	// Concatenate the binary number
	for (int i = 0; i < 8; i++){
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
	for (int i = 0; i < 256; i++){
		lbpHist[i] = 0;
	}
	// for the each row and column, and avoid corners
	for (int i = 2; i < image.rows - 2; i++){

		for (int j = 2; j < image.cols - 2; j++){

			locP[0] = image.at<unsigned char>(i, j);
			locP[1] = image.at<unsigned char>(i - 1, j);
			locP[2] = image.at<unsigned char>(i - 1, j - 1);
			locP[3] = image.at<unsigned char>(i, j - 1);
			locP[4] = image.at<unsigned char>(i + 1, j - 1);
			locP[5] = image.at<unsigned char>(i + 1, j);
			locP[6] = image.at<unsigned char>(i + 1, j + 1);
			locP[7] = image.at<unsigned char>(i, j + 1);
			locP[8] = image.at<unsigned char>(i - 1, j + 1);
			lbpHist[computeLbpCode(locP)] ++;
		}
	}

	return lbpHist;
}

// Compute SIFT code word histogram for given image
void computeSiftCodewordHist(MGHData &data, const Mat &codeWords, Mat features)
{
	Mat histogram = Mat::zeros(1, codeWords.rows, CV_8UC1);

	// build histogram
	for (int i = 0; i < features.rows; i++)
	{
		double min_dist = numeric_limits<double>::infinity();
		int code_word = -1;
		for (int j = 0; j < codeWords.rows; j++)
		{
			double dist = norm(codeWords.row(j), features.row(i), NORM_L2);
			if (dist < min_dist)
			{
				min_dist = dist;
				code_word = j;
			}
		}
		histogram.data[code_word] += 1;
	}
	data.sift_histogram = histogram;
}

// Compute LBP code word histogram for given image
void computeLBPCodewordHist(MGHData &data, const Mat &codeWords, Mat features)
{
	Mat histogram = Mat::zeros(1, codeWords.rows, CV_8UC1);

	// build histogram
	for (int i = 0; i < features.rows; i++)
	{
		double min_dist = numeric_limits<double>::infinity();
		int code_word = -1;
		for (int j = 0; j < codeWords.rows; j++)
		{
			double dist = norm(codeWords.row(j), features.row(i), NORM_L2);
			if (dist < min_dist)
			{
				min_dist = dist;
				code_word = j;
			}
		}
		histogram.data[code_word] += 1;
	}
	data.lbp_histogram = histogram;
}

// Compute code word histogram for given image
void computeCodewordHist(MGHData &data, const Mat &codeWords, Mat features, string feature_mode)
{
	Mat histogram = Mat::zeros(1, codeWords.rows, CV_8UC1);

	// build histogram
	for (int i = 0; i < features.rows; i++)
	{
		double min_dist = numeric_limits<double>::infinity();
		int code_word = -1;
		for (int j = 0; j < codeWords.rows; j++)
		{
			double dist = norm(codeWords.row(j), features.row(i), NORM_L2);
			if (dist < min_dist)
			{
				min_dist = dist;
				code_word = j;
			}
		}
		histogram.data[code_word] += 1;
	}
	if (feature_mode.compare("sift"))
		data.sift_histogram = histogram;
	else if (feature_mode.compare("lbp"))
		data.lbp_histogram = histogram;
}

// Extract LBP Features for given image
Mat extractLBPFeatures(Mat image, Mat &outputFeature){

	Mat input;
	int width = 10;
	int height = 10;
	int N = 10;

	vector<Mat> tiles;

	image.copyTo(input);

	for (int x = 0; x < input.cols - N; x += N){
		for (int y = 0; y < input.rows - N; y += N){
			Mat tile = input(Rect(x, y, N, N));
			tiles.push_back(tile);
		}
	}

	//cout << "[extractLBPFeatures] size of tiles: " << tiles.size() << endl;
	int numOfRow = tiles.size();
	// not uniform pattern
	int hist[256];
	Mat histMat(numOfRow, 256, CV_32F);

	// For each tile, compute the histogram
	for (int i = 0; i < tiles.size(); i++){
		computeLbpHist(tiles.at(i), hist);
		Mat temp = Mat(1, 256, CV_32F, hist);
		temp.copyTo(histMat.row(i));
	}

	return histMat;
}

// Compute code words for given descriptors
Mat computeCodeWords(Mat descriptors, int K){
	Mat labels;
	Mat centers;
	TermCriteria criteria{ TermCriteria::COUNT, 100, 1 };

	//descriptors.convertTo(descriptors, CV_32F);
	kmeans(descriptors, K, labels, criteria, 1, KMEANS_RANDOM_CENTERS, centers);

	cout << "[computeCodeWords] The size of centers: " << centers.rows << " x " << centers.cols << endl;

	return centers;
}

// do classification by finding nearest neighbor to its sift histogram of code words
MGHData computeSiftClassification(MGHData testingdata, vector<MGHData> trainingdata)
{
	MGHData closest_subject;
	double min_dist = numeric_limits<double>::infinity();

	for (int i = 0; i < trainingdata.size(); i++)
	{
		testingdata.sift_histogram.convertTo(testingdata.sift_histogram, CV_32F);
		trainingdata[i].sift_histogram.convertTo(trainingdata[i].sift_histogram, CV_32F);
		double dist = compareHist(testingdata.sift_histogram, trainingdata[i].sift_histogram, CV_COMP_CHISQR);
		if (dist < min_dist)
		{
			min_dist = dist;
			closest_subject = trainingdata[i];
		}
	}
	return closest_subject;
}

// do classification by finding nearest neighbor to its LBP histogram of code words
MGHData computeLBPClassification(MGHData testingdata, vector<MGHData> trainingdata)
{
	MGHData closest_subject;
	double min_dist = numeric_limits<double>::infinity();

	for (int i = 0; i < trainingdata.size(); i++)
	{
		testingdata.lbp_histogram.convertTo(testingdata.lbp_histogram, CV_32F);
		trainingdata[i].lbp_histogram.convertTo(trainingdata[i].lbp_histogram, CV_32F);
		double dist = compareHist(testingdata.lbp_histogram, trainingdata[i].lbp_histogram, CV_COMP_CHISQR);
		if (dist < min_dist)
		{
			min_dist = dist;
			closest_subject = trainingdata[i];
		}
	}
	return closest_subject;
}

// do classification by finding nearest neighbor to its histogram of code words
MGHData computeClassification(MGHData testingdata, vector<MGHData> trainingdata, string feature_mode, int compare_method = CV_COMP_CHISQR)
{
	string sift = "sift";
	string lbp = "lbp";

	MGHData closest_subject;
	double min_dist = numeric_limits<double>::infinity();

	for (int i = 0; i < trainingdata.size(); i++)
	{
		double dist;
		if (feature_mode.compare(sift))
		{
			testingdata.sift_histogram.convertTo(testingdata.sift_histogram, CV_32F);
			trainingdata[i].sift_histogram.convertTo(trainingdata[i].sift_histogram, CV_32F);
			dist = compareHist(testingdata.sift_histogram, trainingdata[i].sift_histogram, compare_method);
		}

		else if (feature_mode.compare(lbp))
		{
			testingdata.lbp_histogram.convertTo(testingdata.lbp_histogram, CV_32F);
			trainingdata[i].lbp_histogram.convertTo(trainingdata[i].lbp_histogram, CV_32F);
			dist = compareHist(testingdata.lbp_histogram, trainingdata[i].lbp_histogram, compare_method);
		}
		if (dist < min_dist)
		{
			min_dist = dist;
			closest_subject = trainingdata[i];
		}
	}
	return closest_subject;
}

// Override method for extracting LBP features
Mat extractLBPFeatures(Mat image){
	Mat output;
	Mat features = extractLBPFeatures(image, output);

	return features;
}

Mat extractSiftFeatures(Mat image){
	// To store keypoints
	vector<KeyPoint> keypoints;
	// To store the SIFT descriptor of current image
	cv::Mat descriptor;

	cv::SiftDescriptorExtractor detector;

	// detect feature points
	detector.detect(image, keypoints);
	detector.compute(image, keypoints, descriptor);

	return descriptor;
}

// Get Region of Interest
Mat getROI(MGHData data){
	MGHData tempData = data;
	Mat tempMat = tempData.image;
	Mat tempROI = tempMat(tempData.roi);

	return tempROI;
}

void drawSIFTImage(int i, vector<KeyPoint> keypoints, Mat input){
	Mat output;
	drawKeypoints(input, keypoints, output);
	imshow(format("SIFT_Features_%i", i), output);
}

void drawLBPHistogram(){

	Mat input = trainingdata.at(1).image;
	
	int height = input.rows;
	int width = input.cols;
	int N = 40;
	int histSize = 256;
	Mat patch_1 = input(Rect(1, 1, N, N));
	//imshow("input", patch_1);
	waitKey(0);
	Mat patch_2 = input(Rect(10, 10, N, N));
	Mat patch_3 = input(Rect(50, 50, N, N));
	int hist_1[256];
	int hist_2[256];
	int hist_3[256];
	computeLbpHist(patch_1, hist_1);
	computeLbpHist(patch_2, hist_2);
	computeLbpHist(patch_3, hist_3);

	// width and height for the window
	int hist_w = 256; int hist_h = 100;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage_1(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImage_2(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	Mat histImage_3(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 1; i < histSize; i++){
		line(histImage_1, Point(bin_w*(i - 1), hist_h - cvRound(hist_1[i - 1])),
			Point(bin_w*(i), hist_h - cvRound(hist_1[i - 1])),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage_2, Point(bin_w*(i - 1), hist_h - cvRound(hist_2[i - 1])),
			Point(bin_w*(i), hist_h - cvRound(hist_2[i - 1])),
			Scalar(255, 0, 0), 2, 8, 0);
		line(histImage_3, Point(bin_w*(i - 1), hist_h - cvRound(hist_3[i - 1])),
			Point(bin_w*(i), hist_h - cvRound(hist_3[i - 1])),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(format("LBP_Hist_%i", 1), CV_WINDOW_AUTOSIZE);
	namedWindow(format("LBP_Hist_%i", 2), CV_WINDOW_AUTOSIZE);
	namedWindow(format("LBP_Hist_%i", 3), CV_WINDOW_AUTOSIZE);
	imshow(format("LBP_Hist_%i", 1), histImage_1);
	imshow(format("LBP_Hist_%i", 2), histImage_2);
	imshow(format("LBP_Hist_%i", 3), histImage_3);
	waitKey(0);
}

void drawSIFTCodeword(){
	
	MGHData tempData = trainingdata.at(2);
	Mat siftCodeword = tempData.sift_histogram;

	// number of codewords
	int histSize = 5;
	// Size of display window
	int hist_w = 256; int hist_h = 100;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage_1(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));
	
	for (int i = 1; i < histSize; i++){
		line(histImage_1, Point(bin_w*(i - 1), hist_h - cvRound(siftCodeword.at<char>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(siftCodeword.at<char>(i - 1))),
			Scalar(255, 0, 0), 2, 8, 0);
		
	}

	/// Display
	namedWindow(format("SIFT_Hist_%i", 1), CV_WINDOW_AUTOSIZE);

	imshow(format("SIFT_Hist_%i", 1), histImage_1);

	waitKey(0);
}

void drawLBPCodeword(){

	MGHData tempData = trainingdata.at(2);
	Mat siftCodeword = tempData.lbp_histogram;

	// number of codewords
	int histSize = 5;
	// Size of display window
	int hist_w = 256; int hist_h = 100;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage_1(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 1; i < histSize; i++){
		line(histImage_1, Point(bin_w*(i - 1), hist_h - cvRound(siftCodeword.at<char>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(siftCodeword.at<char>(i - 1))),
			Scalar(255, 0, 0), 2, 8, 0);

	}


	/// Display
	namedWindow(format("LBP_Hist_%i", 1), CV_WINDOW_AUTOSIZE);

	imshow(format("LBP_Hist_%i", 1), histImage_1);

	waitKey(0);
}
	// 
double computeRecognitionRate(vector<MGHData> testingdata, vector<MGHData> trainingdata, vector<Mat> testing_features, vector<Mat> training_features, const Mat codewords, Mat &confusion_matrix_output, string feature_mode)
{
	map<int, int> angle_to_position;
	angle_to_position.insert(pair<int, int>(-45, 0));
	angle_to_position.insert(pair<int, int>(-30, 1));
	angle_to_position.insert(pair<int, int>(0, 2));
	angle_to_position.insert(pair<int, int>(30, 3));
	angle_to_position.insert(pair<int, int>(45, 4));

	// update histogram field in each testing data 
	for (int i = 0; i < testingdata.size(); i++)
		computeCodewordHist(testingdata[i], codewords, testing_features[i], feature_mode);

	// update histogram field in each training data 
	for (int i = 0; i < trainingdata.size(); i++)
		computeCodewordHist(trainingdata[i], codewords, training_features[i], feature_mode);

	// evaluate recognition performance

	double recognition_performance = 0.0;
	Mat confusion_matrix = Mat::zeros(5, 5, CV_64F);
	Mat normalized_confusion_matrix;
	for (int i = 0; i < testingdata.size(); i++)
	{
		MGHData actual, expected;
		actual = computeClassification(testingdata[i], trainingdata, feature_mode);
		expected = testingdata[i];

		//cout << "Actual = " << actual.subject << ", " << "Expected = " << expected.subject << endl;

		int i_idx, j_idx;
		i_idx = angle_to_position[actual.angle];
		j_idx = angle_to_position[testingdata[i].angle];
		confusion_matrix.at<double>(i_idx, j_idx) += 1;

		if (actual.subject.compare(expected.subject))
			recognition_performance += 1;
	}

	// normalize confusion matrix
	for (int k = 0; k < 5; k++)
	{
		Mat row;
		double sum = norm(confusion_matrix.row(k), NORM_L1);
		row = confusion_matrix.row(k) / sum;
		normalized_confusion_matrix.push_back(row);
	}
	cout << "[computeRecognitionRate] Number of recognized images = " << recognition_performance << endl;
	confusion_matrix_output = normalized_confusion_matrix;
	recognition_performance = (recognition_performance / testingdata.size()) * 100;

	//cout << "SIFT recognition performance [Testing] = " << recognition_performance << "%" << endl;
	cout << "[computeRecognitionRate] " << feature_mode << " UNNORMALIZED Confusion matrix [Testing] = " << endl << " " << confusion_matrix << endl << endl;

	return recognition_performance;
}

Mat detectAndLabel(Mat frame, vector<MGHData> trainingdata, const Mat &codewords,  vector<Mat> training_features, string feature_method)
{
	vector<Rect> faces;
	Mat frame_gray, frame_with_bounded_boxes;

	frame_gray = frame;
	frame.copyTo(frame_with_bounded_boxes);
	equalizeHist(frame_gray, frame_gray);

	String window_name = "Face detection";
	String window_name2 = "Face detection and labelling";

	// Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point c1(faces[i].x, faces[i].y);
		Point c2(faces[i].x + faces[i].width, faces[i].y + faces[i].height);
		rectangle(frame, c1, c2, Scalar(255, 0, 255));
		rectangle(frame_with_bounded_boxes, c1, c2, Scalar(255, 0, 255));

		// perform classification
		MGHData nearest_neighbor;
		MGHData testingdata;
		testingdata.image = frame;
		testingdata.roi = Rect(c1, c2);

		Mat testing_features;
		testing_features.push_back(extractSiftFeatures(getROI(testingdata)));

		// update histogram field in each testing data 
		computeCodewordHist(testingdata, codewords, testing_features, feature_method);

		// update histogram field in each training data 
		for (int i = 0; i < trainingdata.size(); i++)
			computeCodewordHist(trainingdata[i], codewords, training_features[i], feature_method);

		nearest_neighbor = computeClassification(testingdata, trainingdata, feature_method);

		putText(frame, nearest_neighbor.subject, c1, FONT_HERSHEY_PLAIN, 0.75, Scalar(255, 0, 255));
		Mat faceROI = frame_gray(faces[i]);
	}

	imshow(window_name2, frame);
	imshow(window_name, frame_with_bounded_boxes);
	return frame;
}
//Callback for mousclick event, the x-y coordinate of mouse button-up and button-down 
//are stored in two points pt1, pt2.
/*
void mouse_click(int event, int x, int y, int flags, void *param)
{

	switch (event)
	{
	case CV_EVENT_LBUTTONDOWN:
	{
		std::cout << "Mouse Pressed" << std::endl;

		pt1.x = x;
		pt1.y = y;

		break;
	}
	case CV_EVENT_LBUTTONUP:
	{
		if (!got_roi)
		{
			Mat cl;
			std::cout << "Mouse LBUTTON Released" << std::endl;

			pt2.x = x;
			pt2.y = y;
			std::cout << "PT1" << pt1.x << ", " << pt1.y << std::endl;
			std::cout << "PT2" << pt2.x << "," << pt2.y << std::endl;

			got_roi = true;
		}
		else
		{
			std::cout << "ROI Already Acquired" << std::endl;
		}
		break;
	}

	}

}
*/