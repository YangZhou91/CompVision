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

int computeLbpCode(unsigned char seq[9]);
int* computeLbpHist(Mat &image, int *lbpHist);
void computeSiftCodewordHist(MGHData &data, const Mat &codeWords, Mat features);
void computeLBPCodewordHist(MGHData &data, const Mat &codeWords, Mat features);
Mat extractLBPFeatures(Mat image, Mat &outputFeatures);
Mat extractLBPFeatures(Mat image);
Mat extractSiftFeatures(Mat image);
Mat computeCodeWords(Mat descriptors, int K);
MGHData computeSiftClassification(MGHData testingdata, vector<MGHData> trainingdata);
MGHData computeLBPClassification(MGHData testingdata, vector<MGHData> trainingdata);


Mat getROI(MGHData data);

bool MGHDataLoader(vector<MGHData> &trainingdataset, vector<MGHData> &testingdataset, vector<MGHData> &groupdataset, string directory);
void mouse_click(int event, int x, int y, int flags, void *param);

int main() 
{
	/* PART 1 */
	vector<MGHData> trainingdata, testingdata, groupdata;

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

	map<int, int> angle_to_position;
	angle_to_position.insert(pair<int, int>(-45, 0));
	angle_to_position.insert(pair<int, int>(-30, 1));
	angle_to_position.insert(pair<int, int>(0, 2));
	angle_to_position.insert(pair<int, int>(30, 3));
	angle_to_position.insert(pair<int, int>(45, 4));

	for (int number_of_clusters = 5; number_of_clusters <= 20; number_of_clusters *= 2)
	{
		cout << "Number of clusters = " << number_of_clusters << endl;

		// update histogram field in each testing data 
		for (int i = 0; i < testingdata.size(); i++)
			computeSiftCodewordHist(testingdata[i], sift_feature_clusters[number_of_clusters], sift_features_testing[i]);

		// update histogram field in each testing data 
		for (int i = 0; i < testingdata.size(); i++)
			computeLBPCodewordHist(testingdata[i], lbp_feature_clusters[number_of_clusters], lbp_features_testing[i]);

		// update histogram field in each training data 
		for (int i = 0; i < trainingdata.size(); i++)
			computeSiftCodewordHist(trainingdata[i], sift_feature_clusters[number_of_clusters], sift_features_training[i]);

		// update histogram field in each training data 
		for (int i = 0; i < trainingdata.size(); i++)
			computeLBPCodewordHist(trainingdata[i], lbp_feature_clusters[number_of_clusters], lbp_features_training[i]);

		// evaluate recognition performance
		cout << "Computing classification..." << endl;

		double sift_recognition_performance = 0.0;
		Mat sift_confusion_matrix = Mat::zeros(5, 5, CV_64F);
		Mat normalized_sift_confusion_matrix;
		for (int i = 0; i < testingdata.size(); i++)
		{
			MGHData actual, expected;
			actual = computeSiftClassification(testingdata[i], trainingdata);
			expected = testingdata[i];

			//cout << "Actual = " << actual.subject << ", " << "Expected = " << expected.subject << endl;

			int i_idx, j_idx;
			i_idx = angle_to_position[actual.angle];
			j_idx = angle_to_position[testingdata[i].angle];
			sift_confusion_matrix.at<double>(i_idx, j_idx) += 1;

			if (actual.subject.compare(expected.subject))
				sift_recognition_performance += 1;
		}

		// normalize confusion matrix
		for (int k = 0; k < 5; k++)
		{
			Mat row;
			double sum = norm(sift_confusion_matrix.row(k), NORM_L1);
			row = sift_confusion_matrix.row(k) / sum;
			normalized_sift_confusion_matrix.push_back(row);
		}

		sift_recognition_performance = (sift_recognition_performance / trainingdata.size()) * 100;
		cout << "SIFT recognition performance = " << sift_recognition_performance << "%" << " (k=" << number_of_clusters << ")" << endl;
		cout << "SIFT Confusion matrix = " << endl << " " << normalized_sift_confusion_matrix << endl << endl << "\n" << endl;

		double lbp_recognition_performance = 0.0;
		Mat lbp_confusion_matrix = Mat::zeros(5, 5, CV_64F);
		Mat lbp_normalized_confusion_matrix;
		for (int i = 0; i < testingdata.size(); i++)
		{
			MGHData actual, expected;
			actual = computeLBPClassification(testingdata[i], trainingdata);
			expected = testingdata[i];

			//cout << "Actual = " << actual.subject << ", " << "Expected = " << expected.subject << endl;

			int i_idx, j_idx;
			i_idx = angle_to_position[actual.angle];
			j_idx = angle_to_position[testingdata[i].angle];
			lbp_confusion_matrix.at<double>(i_idx, j_idx) += 1;

			if (actual.subject.compare(expected.subject))
				lbp_recognition_performance += 1;
		}

		// normalize confusion matrix
		for (int k = 0; k < 5; k++)
		{
			Mat row;
			double sum = norm(lbp_confusion_matrix.row(k), NORM_L1);
			row = lbp_confusion_matrix.row(k) / sum;
			lbp_normalized_confusion_matrix.push_back(row);
		}

		lbp_recognition_performance = (lbp_recognition_performance / trainingdata.size()) * 100;
		cout << "LBP recognition performance = " << lbp_recognition_performance << "%" << " (k=" << number_of_clusters << ")" << endl;
		cout << "LBP Confusion matrix = " << endl << " " << lbp_normalized_confusion_matrix << endl << endl << "\n" << endl;
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

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);
	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	for (int i = 1; i < histSize; i++){
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist[i - 1])),
			Point(bin_w*(i), hist_h - cvRound(hist[i - 1])),
			Scalar(255, 0, 0), 2, 8, 0);
		//cout << "[drawLBPFeature] hist " << i -1  << " "<< feature[i-1] << endl;
	}

	/// Display
	namedWindow("calcHist Demo", CV_WINDOW_AUTOSIZE);
	imshow("calcHist Demo", histImage);

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

// Compute SIFT code word histogram for given image
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

	cout << "[computedCodeWords] The size of centers: " << centers.rows << " x " << centers.cols << endl;

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

Mat getROI(MGHData data){
	MGHData tempData = data;
	Mat tempMat = tempData.image;
	Mat tempROI = tempMat(tempData.roi);

	return tempROI;
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