#ifndef _RANSAC_H_
#define _RANSAC_H_

#include <iostream>
#include <pcl/console/parse.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/sample_consensus/ransac.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/sac_model_sphere.h>
#include <boost/thread/thread.hpp>

#include "opencv2/highgui.hpp"
#include <opencv2/calib3d/calib3d.hpp>
#include <vector>

#include "SGM.h"

void computeAlpha(cv::Mat &disparity, cv::Mat &disparityFlow, std::vector<cv::Mat> &alpha, cv::Mat &disflag){

	const int HEIGHT = disparity.rows;
	const int WIDTH = disparity.cols;
	const int total_set = 50;
	const int dim = 1000;
	for(int i = 0; i < total_set; i++){
		cv::Mat A(dim, 3, CV_32FC1);
		cv::Mat b(dim, 1, CV_32FC1);
		cv::Mat x(3, 1, CV_32FC1);
		int Px;
		int Py;
		int counter =0;
		while(counter != dim){		
			Px = static_cast<int>(WIDTH * (rand ()/(RAND_MAX + 1.0)));
			Py = static_cast<int>(HEIGHT * (rand ()/(RAND_MAX + 1.0)));
			
			if(disflag.at<uchar>(Py,Px) != DISFLAG){	
				A.at<float>(counter,0) = static_cast<float>(Px * disparity.at<uchar>(Py, Px));
				A.at<float>(counter,1) = static_cast<float>(Py * disparity.at<uchar>(Py, Px));
				A.at<float>(counter,2) = static_cast<float>(disparity.at<uchar>(Py, Px));
				b.at<float>(counter,0) = static_cast<float>(disparityFlow.at<uchar>(Py, Px));
				counter ++;
			}
		}
		cv::solve(A,b,x,cv::DECOMP_QR);
		alpha.push_back(x);

	}
}

cv::Vec3f ransac(cv::Mat &disparity, cv::Mat &disparityFlow, cv::Mat disflag)
{

	std::vector<cv::Mat> alpha;
	computeAlpha(disparity, disparityFlow, alpha, disflag);
	const int alphaSize = alpha.size();

  	// initialize PointClouds
  	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud (new pcl::PointCloud<pcl::PointXYZ>);
  	pcl::PointCloud<pcl::PointXYZ>::Ptr finalcloud (new pcl::PointCloud<pcl::PointXYZ>);

  	// populate our PointCloud with points
  	cloud->width    = alphaSize;
  	cloud->height   = 1;
  	cloud->is_dense = false;
  	cloud->points.resize (cloud->width * cloud->height);

  	for (size_t i = 0; i < cloud->points.size (); ++i)
  	{
    		cloud->points[i].x = alpha[i].at<float>(0) ;
      		cloud->points[i].y = alpha[i].at<float>(1) ;
		cloud->points[i].z = alpha[i].at<float>(2) ;
 	}

  	std::vector<int> inliers;

  	// created RandomSampleConsensus object and compute the appropriated model
  
  	pcl::SampleConsensusModelPlane<pcl::PointXYZ>::Ptr
    	model_p (new pcl::SampleConsensusModelPlane<pcl::PointXYZ> (cloud));

    	pcl::RandomSampleConsensus<pcl::PointXYZ> ransac (model_p);
    	ransac.setDistanceThreshold (.00001);
    	ransac.computeModel();
    	ransac.getInliers(inliers);

  
  	// copies all inliers of the model computed to another PointCloud

  	pcl::copyPointCloud<pcl::PointXYZ>(*cloud, inliers, *finalcloud);

	const int finalcloudSize = finalcloud->points.size();
	double x = 0.0;
	double y = 0.0;
	double z = 0.0;


	for(int i = 0; i < finalcloudSize; i++){
		x += finalcloud->points[i].x;
		y += finalcloud->points[i].y;
		z += finalcloud->points[i].z;
	}
	cv::Vec3f avgAlpha(x/finalcloudSize,y/finalcloudSize, z/finalcloudSize);

	/* no longer needed	
	cv::Mat omega(disparity.rows, disparity.cols, CV_8UC1, cv::Scalar::all(0));
	cv::Mat diff(disparity.rows, disparity.cols, CV_8UC1, cv::Scalar::all(0));
	{
		const int HEIGHT = disparity.rows;
		const int WIDTH = disparity.cols;
		for(int x = DISP_RANGE+1; x < WIDTH; x++){
			for(int y =0; y < HEIGHT; y++){
				omega.at<uchar>(y,x) = static_cast<uchar>((avgAlpha[0]*x + avgAlpha[1]*y + avgAlpha[2])*disparity.at<uchar>(y,x));
				diff.at<uchar>(y,x) = static_cast<uchar>(abs(omega.at<uchar>(y,x) - disparityFlow.at<uchar>(y,x)));
			}
		}
	}
	imwrite("../omega.jpg", omega);
	imwrite("../diff.jpg", diff);
	*/

	return avgAlpha;
 }
#endif

