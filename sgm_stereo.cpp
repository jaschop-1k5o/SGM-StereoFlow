#include "SGM.h"
#include "opencv2/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/xfeatures2d/nonfree.hpp"
#include "opencv2/stitching/detail/matchers.hpp"
#include <vector>

void computeEpipoles(std::vector<cv::Vec3f> &lines, cv::Mat &x_sol);

int main(int argc, char *argv[]){


	cv::Mat imageLeft, imageRight, imageLeftLast;
	cv::Mat grayLeft, grayRight, grayLeftLast;
	grayLeft=cv::imread("/home/johann/TUM/17S/Seminar-HWSWCodesign/KITTI/training/image_0/000003_11.png",CV_LOAD_IMAGE_GRAYSCALE);
	grayRight=cv::imread("/home/johann/TUM/17S/Seminar-HWSWCodesign/KITTI/training/image_1/000003_11.png",CV_LOAD_IMAGE_GRAYSCALE);
	grayLeftLast=cv::imread("/home/johann/TUM/17S/Seminar-HWSWCodesign/KITTI/training/image_0/000003_10.png",CV_LOAD_IMAGE_GRAYSCALE);
	imageLeftLast=cv::imread("/home/johann/TUM/17S/Seminar-HWSWCodesign/KITTI/training/colored_0/000003_10.png",CV_LOAD_IMAGE_COLOR);
	imageLeft=cv::imread("/home/johann/TUM/17S/Seminar-HWSWCodesign/KITTI/training/colored_0/000003_11.png",CV_LOAD_IMAGE_COLOR);
	//cv::cvtColor(imageLeft,grayLeft,CV_BGR2GRAY);
	//cv::cvtColor(imageRight,grayRight,CV_BGR2GRAY);
	//cv::cvtColor(imageLeftLast,grayLeftLast,CV_BGR2GRAY);
	const Scalar colorBlue(225.0, 0.0, 0.0, 0.0);
	const Scalar colorRed(0.0, 0.0, 225.0, 0.0);
	const Scalar colorOrange(0.0, 69.0, 225.0, 0.0);
	const Scalar colorYellow(0.0, 255.0, 225.0, 0.0);

	const int PENALTY1 = 400; //400 stereo
	const int PENALTY2 = 6000; //6600 stereo
	const int winRadius = 2;  //2 stereo
//-- Compute the stereo part
/*	cv::Mat disparity(grayLeft.rows, grayLeft.cols, CV_8UC1);
	SGMStereo sgmstereo(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius);
	sgmstereo.runSGM(disparity);
	imwrite("../disparity.jpg", disparity);
	imshow("disparity", disparity);
	sgmstereo.writeDerivative();
*/


//-- compute the flow part

	std::cout<<"HEIGHT: "<<imageLeft.rows<<std::endl;
	std::cout<<"WIDTH: "<<imageLeft.cols<<std::endl;

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
		if( matches1to2[i].distance < 4*min_dist ){ 
			good_matches.push_back( matches1to2[i]);
			temp_keypoints_1.push_back((Point2f)keypoints_1[matches1to2[i].queryIdx].pt);
			temp_keypoints_2.push_back((Point2f)keypoints_2[matches1to2[i].trainIdx].pt);

		}
  	}


//-- Draw matching points
	//cv::drawMatches(grayLeftLast, keypoints_1, grayLeft, keypoints_2, good_matches, outImg);
	
	
	cv::Mat mask;
	cv::Mat Fmat;
	Fmat=findFundamentalMat(temp_keypoints_1, temp_keypoints_2, CV_FM_RANSAC , 3, 0.999 ,mask); //CV_FM_8POINT, CV_FM_RANSAC, CV_FM_LMEDS
	std::cout<<Fmat<<std::endl;


	
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

	std::cout<<"num_inliers : "<<num_inliers<<std::endl;
	std::vector<cv::Vec3f> lines_1;
	std::vector<cv::Vec3f> lines_2;
	

	cv::computeCorrespondEpilines(new_keypoints_2, 2, Fmat, lines_1);
	cv::computeCorrespondEpilines(new_keypoints_1, 1, Fmat, lines_2);	

	std::vector<Point2f> des_points_1;
	std::vector<Point2f> des_points_2;
	
	std::cout<<"new_keypoints_1.size() :"<<new_keypoints_1.size()<<std::endl;
	std::cout<<"new_keypoints_2.size() :"<<new_keypoints_2.size()<<std::endl;
	std::cout<<"mask.rows :"<<mask.rows<<std::endl;


//-- Compute epipoles by least square
	cv::Mat Epipole_1 = Mat::zeros(2, 1, CV_32F);
	computeEpipoles(lines_1, Epipole_1);
	std::cout<<"Epipole_1: "<<Epipole_1<<std::endl;

	cv::Mat Epipole_2 = Mat::zeros(2, 1, CV_32F);
	computeEpipoles(lines_2, Epipole_2);
	std::cout<<"Epipole_2: "<<Epipole_2<<std::endl;

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


	cv::Mat disparityFlow_(grayLeft.rows, grayLeft.cols, CV_8UC1);
	SGMFlow sgmflow(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius, Epipole_1, Epipole_2, Fmat);
	sgmflow.runSGM(disparityFlow_);
	imshow("disparityFlow_",disparityFlow_);
	imshow("imageLeftLast_",imageLeftLast);
	cv::Mat disFlowFlag;
	sgmflow.copyDisflag(disFlowFlag);
	imwrite("../disparityFlow.jpg", disparityFlow_);
	imwrite("../aviFlowFlag.jpg", disFlowFlag);
	sgmflow.writeDerivative();


//-- compute the stereoflow part
/*	cv::Mat disparityStereo, disparityFlow;
	disparityFlow=cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/results/flow/Flow106_8paths_40D_wo.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	disparityStereo=cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/results/stereo/disparity106_8paths_80D_wo.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	disFlowFlag=cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/results/flow/aviFlowFlag106.jpg",CV_LOAD_IMAGE_GRAYSCALE );	
	SGMStereoFlow sgmsf(grayLeftLast, grayLeft, grayRight, PENALTY1, PENALTY2, winRadius, Epipole_1, Epipole_2, Fmat);
	sgmsf.setAlphaRansac(disparityStereo, disparityFlow, disFlowFlag);

	cv::Mat disparityStereo_wOutliers = cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/results/stereoEvi/disparity106_evi30000.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	cv::Mat disparityFlow_wOutliers = cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/results/flow/Flow106_8paths_40D_wo.jpg",CV_LOAD_IMAGE_GRAYSCALE );
	sgmsf.setEvidence(disparityStereo_wOutliers, disparityFlow_wOutliers, disFlowFlag);

	cv::Mat disparityStereoFlow(grayLeft.rows, grayLeft.cols, CV_8UC1);
	sgmsf.runSGM(disparityStereoFlow);

	
	imshow("disparityStereoFlow",disparityStereoFlow);
	imwrite("../disparityStereoFlow.jpg",disparityStereoFlow);
*/
	waitKey(0);
	
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



