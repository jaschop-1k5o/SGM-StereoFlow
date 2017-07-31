#include "SGM.h"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include <vector>
//for automated filepath building:
#include <string>
#include <sstream>
//for runtime analysis:
#include <ctime>
#include <iostream>
#include <fstream>

//-- macros: STEREO|FLOW|STEREOFLOW|DRAWEPIPOLES
#define STEREO
#define FLOW
#define STEREOFLOW

void computeEpipoles(std::vector<cv::Vec3f> &lines, cv::Mat &x_sol);

int main(int argc, char *argv[]){

/*RUNTIME ANALYSIS*/std::clock_t t0 = std::clock();/*RUNTIME ANALYSIS*/

//-- set some default arguments for easy testing
	//-- path to local KITTI directory
	std::string kittiDir = "/home/johann/TUM/17S/Seminar-HWSWCodesign/KITTI";
	//-- image directory paths relative to KITTI directory
	//--- colored images (left & right camera)
	std::string colorLeftDir = "training/colored_0";
	std::string colorRightDir = "training/colored_1"; //currently not used
	//---grayscale images (left & right camera)
	std::string grayLeftDir = "training/image_0";
	std::string grayRightDir = "training/image_1";
	//-- KITTI image ID, filenames of current/last frame are {imgID}_11/{imgID}_10 resp.
	std::string imgID = "000000";
//-- parse arguments (RUDIMENTARY)
	if(argc>=2){
		imgID = argv[1];
	}

	std::cout<<"evaluating image "<<argv[1]<<std::endl;

//-- Read input images
	cv::Mat imageLeft, imageLeftLast;
	cv::Mat grayLeft, grayRight, grayLeftLast;
	//build filepaths via stringstream
	std::stringstream path;
	path.str("");path<<kittiDir<<"/"<<grayLeftDir<<"/"<<imgID<<"_11.png";
	grayLeft=cv::imread(path.str(),CV_LOAD_IMAGE_GRAYSCALE);

	path.str("");path<<kittiDir<<"/"<<grayRightDir<<"/"<<imgID<<"_11.png";
	grayRight=cv::imread(path.str(),CV_LOAD_IMAGE_GRAYSCALE);

	path.str("");path<<kittiDir<<"/"<<grayLeftDir<<"/"<<imgID<<"_10.png";
	grayLeftLast=cv::imread(path.str(),CV_LOAD_IMAGE_GRAYSCALE);

	path.str("");path<<kittiDir<<"/"<<colorLeftDir<<"/"<<imgID<<"_11.png";
	imageLeft=cv::imread(path.str(),CV_LOAD_IMAGE_COLOR);

	path.str("");path<<kittiDir<<"/"<<colorLeftDir<<"/"<<imgID<<"_10.png";
	imageLeftLast=cv::imread(path.str(),CV_LOAD_IMAGE_COLOR);

//--check if images were loaded correctly
	if(grayLeft.cols<=0 || grayLeft.rows<=0 || grayRight.cols<=0 || grayRight.rows<=0 ||
		grayLeftLast.cols<=0 || grayLeftLast.rows<=0 || imageLeft.cols<=0 || imageLeft.rows<=0){		 
		std::cout<<"ERROR: some images could not be loaded!"<<std::endl;
		return -1;
	}

/*RUNTIME ANALYSIS*/std::clock_t t1 = std::clock();/*RUNTIME ANALYSIS*/

#ifdef DRAWEPIPOLES
//-- colors for image drawing
	const Scalar colorBlue(225.0, 0.0, 0.0, 0.0);
	const Scalar colorRed(0.0, 0.0, 225.0, 0.0);
	const Scalar colorOrange(0.0, 69.0, 225.0, 0.0);
	const Scalar colorYellow(0.0, 255.0, 225.0, 0.0);
#endif
//-- Declare computation constants
	const int PENALTY1 = 400; //400 stereo
	const int PENALTY2 = 6000; //6600 stereo
	const int winRadius = 2;  //2 stereo

#ifdef STEREO
//-- Compute the stereo disparity
	std::cout<<std::endl<<"COMPUTING STEREO DISPARITY"<<std::endl;
	cv::Mat disparityStereo_(grayLeft.rows, grayLeft.cols, CV_8UC1);
	SGMStereo sgmstereo(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius);
	sgmstereo.runSGM(disparityStereo_);
	cv::Mat stereo_formatted(grayLeft.rows, grayLeft.cols, CV_16UC1);
	disparityStereo_.convertTo(stereo_formatted, CV_16U, 255);
	imwrite("../disparityStereo.png", stereo_formatted);
#endif

/*RUNTIME ANALYSIS*/std::clock_t t2 = std::clock();/*RUNTIME ANALYSIS*/

#if defined FLOW || defined STEREOFLOW
//-- precomputation for flow analysis
	std::cout<<"Computing epipoles: ";

	std::vector<KeyPoint> keypoints_1;
	std::vector<KeyPoint> keypoints_2;
	std::vector<DMatch> matches1to2;
	cv::Mat descriptors_1, descriptors_2;
	cv::Ptr<xfeatures2d::SIFT> sift = xfeatures2d::SIFT::create();
	sift->detectAndCompute(grayLeftLast,noArray(),keypoints_1,descriptors_1);
	sift->detectAndCompute(grayLeft,noArray(),keypoints_2,descriptors_2);
	cv:: FlannBasedMatcher matcher;
	matcher.match(descriptors_1, descriptors_2, matches1to2);


	double max_dist = 0; double min_dist = 100;
//-- Quick calculation of max and min distances between keypoints
  	for( int i = 0; i < matches1to2.size(); i++ ){ 
		double dist = matches1to2[i].distance;
    		if( dist < min_dist ) min_dist = dist;
    		if( dist > max_dist ) max_dist = dist;
		
  	}
	printf("-- Max dist : %f \n", max_dist );
  	printf("-- Min dist : %f \n", min_dist );	
//-- Draw only "good" matches (i.e. whose distance is less than C*min_dist )
 	std::vector< DMatch > good_matches;
	std::vector<Point2f> temp_keypoints_1;
	std::vector<Point2f> temp_keypoints_2;

 	for( int i = 0; i < matches1to2.size(); i++ ){ 
		if( matches1to2[i].distance < 16*min_dist ){ 
			good_matches.push_back( matches1to2[i]);
			temp_keypoints_1.push_back((Point2f)keypoints_1[matches1to2[i].queryIdx].pt);
			temp_keypoints_2.push_back((Point2f)keypoints_2[matches1to2[i].trainIdx].pt);

		}
  	}

	cv::Mat mask;
	cv::Mat Fmat;
	Fmat=findFundamentalMat(temp_keypoints_1, temp_keypoints_2, CV_FM_RANSAC , 3, 0.999 ,mask); //CV_FM_8POINT, CV_FM_RANSAC, CV_FM_LMEDS

	std::vector<Point2f> new_keypoints_1;
	std::vector<Point2f> new_keypoints_2;

	int num_inliers = 0;
	for (int i = 1; i < temp_keypoints_1.size(); i++){
		Point_<uchar> pp= mask.at<Point_<uchar> >(i,0);
		Point2f a=temp_keypoints_1[i];
		Point2f b=temp_keypoints_1[i-1];
		if((short)pp.x != 0 && (a.x != b.x) && (b.y != a.y)){
			new_keypoints_1.push_back(temp_keypoints_1[i]);
			new_keypoints_2.push_back(temp_keypoints_2[i]);
			num_inliers++;
		}
	}

	//updated 23.11.2017. The fundamental matrix must be recomputed again with CV_FM_8POINT by qualified matchings.
        cv::Mat newFmat = findFundamentalMat(new_keypoints_1, new_keypoints_2, CV_FM_8POINT);

	std::vector<cv::Vec3f> lines_1;
	std::vector<cv::Vec3f> lines_2;	

	cv::computeCorrespondEpilines(new_keypoints_2, 2, newFmat, lines_1);
	cv::computeCorrespondEpilines(new_keypoints_1, 1, newFmat, lines_2);	

	std::vector<Point2f> des_points_1;
	std::vector<Point2f> des_points_2;

//-- Compute epipoles by least square
	cv::Mat Epipole_1 = Mat::zeros(2, 1, CV_32F);
	computeEpipoles(lines_1, Epipole_1);

	cv::Mat Epipole_2 = Mat::zeros(2, 1, CV_32F);
	computeEpipoles(lines_2, Epipole_2);

	std::cout<<"done"<<std::endl;

#ifdef DRAWEPIPOLES
//-- draw epipolar lines on image 	
	des_points_2.push_back(cv::Point2f(Epipole_2.at<float>(0), Epipole_2.at<float>(1)));
	des_points_1.push_back(cv::Point2f(Epipole_1.at<float>(0), Epipole_1.at<float>(1)));

	for (int i=0; i < new_keypoints_1.size(); i++){			
		cv::line(imageLeftLast, new_keypoints_1[i], des_points_1[0], colorRed,1,8);
		cv::line(imageLeftLast, new_keypoints_2[i], des_points_2[0], colorBlue,1,8);
		cv::line(imageLeft, new_keypoints_2[i], des_points_2[0], colorBlue,1,8);
		cv::circle(imageLeftLast, new_keypoints_1[i],10,colorRed,1);
		cv::circle(imageLeft, new_keypoints_2[i],10,colorBlue,1);	
	}
	imwrite("../imageLeftLast_epipoles.png",imageLeftLast);
#endif
#endif

/*RUNTIME ANALYSIS*/std::clock_t t3 = std::clock();/*RUNTIME ANALYSIS*/

#ifdef FLOW
//-- Compute the flow disparity
	std::cout<<std::endl<<"COMPUTING FLOW DISPARITY"<<std::endl;
	cv::Mat disparityFlow_(grayLeft.rows, grayLeft.cols, CV_8UC1);
	SGMFlow sgmflow(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius, Epipole_1, Epipole_2, Fmat);
	sgmflow.runSGM(disparityFlow_);
	cv::Mat disFlowFlag_;
	sgmflow.copyDisflag(disFlowFlag_);
	imwrite("../disparityFlow.png", disparityFlow_);
	imwrite("../disFlowFlag.png", disFlowFlag_);
#endif

/*RUNTIME ANALYSIS*/std::clock_t t4 = std::clock();/*RUNTIME ANALYSIS*/

#ifdef STEREOFLOW
//-- Compute the stereoflow disparity
	cv::Mat disparityStereo, disparityFlow, disFlowFlag;
//-- Acquire results of Stereo&Flow computation 
#ifdef STEREO
	disparityStereo = disparityStereo_;
#else
	disparityStereo=cv::imread("../input/disparityStereo.png",CV_LOAD_IMAGE_GRAYSCALE );
#endif
#ifdef FLOW
	disparityFlow = disparityFlow_;
	disFlowFlag = disFlowFlag_;
#else
	disparityFlow=cv::imread("../input/disparityFlow.png",CV_LOAD_IMAGE_GRAYSCALE );
	disFlowFlag=cv::imread("../input/disFlowFlag.png",CV_LOAD_IMAGE_GRAYSCALE );	
#endif
	std::cout<<std::endl<<"COMPUTING STEREO-FLOW DISPARITY"<<std::endl;
	SGMStereoFlow sgmsf(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius, Epipole_1, Epipole_2, Fmat);

	sgmsf.setAlphaRansac(disparityStereo, disparityFlow, disFlowFlag);
	sgmsf.setEvidence(disparityStereo, disparityFlow, disFlowFlag);

	cv::Mat disparityStereoFlow(grayLeft.rows, grayLeft.cols, CV_8UC1);
	sgmsf.runSGM(disparityStereoFlow);

	//convert to 16Bit format (required by KITTI Benchmark)
	cv::Mat disparity_formatted(grayLeft.rows, grayLeft.cols, CV_16UC1);
	disparityStereoFlow.convertTo(disparity_formatted, CV_16U, 255);
	imwrite("../disparityStereoFlow.png",disparity_formatted);
#endif

/*RUNTIME ANALYSIS*/std::clock_t t5 = std::clock();/*RUNTIME ANALYSIS*/

	std::ofstream timers;
	timers.open("../timers.txt");

	double duration = double(t1 - t0)/CLOCKS_PER_SEC;
	timers<<"[loading   ]"<<duration<<std::endl;

	duration = double(t2 - t1)/CLOCKS_PER_SEC;
	timers<<"[stereo    ]"<<duration<<std::endl;

	duration = double(t3 - t2)/CLOCKS_PER_SEC;
	timers<<"[epipoles  ]"<<duration<<std::endl;

	duration = double(t4 - t3)/CLOCKS_PER_SEC;
	timers<<"[flow      ]"<<duration<<std::endl;

	duration = double(t5 - t4)/CLOCKS_PER_SEC;
	timers<<"[stereoflow]"<<duration<<std::endl;

	duration = double(t5 - t1)/CLOCKS_PER_SEC;
	timers<<"[comp.total]"<<duration<<std::endl;

	timers.close();

	return 0;
}


void computeEpipoles(std::vector<cv::Vec3f> &lines, cv::Mat &x_sol){

	const int cols = 2;
	const int rows = lines.size();	
	
	Mat A 	  = Mat::zeros(rows, cols, CV_32FC1);
	Mat b     = Mat::zeros(rows, 1, CV_32FC1);

	for(int i = 0; i < rows; i++){
		A.at<float>(i,0) = lines[i][0];
		A.at<float>(i,1) = lines[i][1];
		b.at<float>(i,0) = -1.0 * lines[i][2];
	}
	cv::solve(A,b,x_sol,DECOMP_QR);


}



