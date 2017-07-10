#ifndef _SGM_H_
#define _SGM_H_

#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <limits>
#include <iostream>
#define DISP_RANGE 100
#define DIS_FACTOR 6	
#define CENSUS_W 5
#define DISFLAG 100
#define Disthreshold 10000
#define Outlier 255
#define Vmax 0.3
#define disparityThreshold 2
#define Dinvd 0
using namespace cv;

class SGM
{

	protected:
		typedef Vec<float, DISP_RANGE> VecDf;
		int PENALTY1;
		int PENALTY2;
		const int winRadius;
		const cv::Mat &imgLeft; 
		const cv::Mat &imgRight;
		const cv::Mat &imgLeftLast;
		cv::Mat censusImageRight;
		cv::Mat censusImageLeft;
		cv::Mat censusImageLeftLast;
		cv::Mat cost;
		cv::Mat costRight;
		cv::Mat directCost;
		cv::Mat accumulatedCost;
		int HEIGHT;
		int WIDTH;


		void computeCensus(const cv::Mat &image, cv::Mat &censusImg);
		int  computeHammingDist(const uchar left, const uchar right);
		VecDf addPenalty(VecDf const& prior, VecDf &local, float path_intensity_gradient);
		void sumOverAllCost();
		virtual void createDisparity(cv::Mat &disparity);
		template <int DIRX, int DIRY> void aggregation(cv::Mat cost);
		virtual void computeDerivative();
		virtual void computeCost();
		virtual void computeCostRight();
		virtual void postProcess(cv::Mat &disparity);
		virtual void resetDirAccumulatedCost();
		virtual void consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity);
	public:
		SGM(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_);
		void setPenalty(const int penalty_1, const int penalty_2);	
		void runSGM(cv::Mat &disparity);

		virtual void writeDerivative();
		virtual ~SGM();	
	
};



class SGMStereo : public SGM
{
	
	protected:
		cv::Mat derivativeStereoLeft;
		cv::Mat derivativeStereoRight;
		cv::Mat halfPixelRightMin;
		cv::Mat halfPixelRightMax;
		void calcHalfPixelRight();
		virtual void computeDerivative();
		virtual void computeCost();
		virtual void computeCostRight();
		virtual void postProcess(cv::Mat &disparity);
		virtual void consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity);
	public:
		SGMStereo(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_);
		virtual ~SGMStereo();
		virtual void writeDerivative();
};

class SGMFlow : public SGM
{

	protected:
		cv::Mat imgRotation;
		cv::Mat EpipoleLeftLast;
		cv::Mat EpipoleLeft;
		cv::Mat translationLeftLast;
		cv::Mat translationLeft;
		cv::Mat fundamentalMatrix;
		cv::Mat derivativeFlowLeftLast;
		cv::Mat derivativeFlowLeft;
		cv::Mat disFlag;
		virtual void computeDerivative();
		virtual void computeCost();
		virtual void postProcess(cv::Mat &disparity);
		void computeRotation();
		void computeTranslation(cv::Mat &translation, cv::Mat &Epipole);
		virtual void resetDirAccumulatedCost();
	public:
		SGMFlow(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_, 
			cv::Mat &EpipoleLeftLast_, cv::Mat &EpipoleLeft_, cv::Mat &fundamentalMatrix_);
		virtual ~SGMFlow();
		virtual void writeDerivative();
		void copyDisflag(cv::Mat &M);
};

class SGMStereoFlow : public SGM
{
	protected:
		cv::Mat imgRotation;
		cv::Mat EpipoleLeftLast;
		cv::Mat EpipoleLeft;
		cv::Mat translationLeftLast;
		cv::Mat translationLeft;
		cv::Mat fundamentalMatrix;
		cv::Mat derivativeStereoLeft;
		cv::Mat derivativeStereoRight;
		cv::Mat derivativeFlowLeftLast;
		cv::Mat derivativeFlowLeft;
		cv::Mat disFlag;
		cv::Mat eviStereo;
		cv::Mat eviFlow;
		cv::Vec3f ransacAlpha;
		virtual void computeCost();
		virtual void computeDerivative();
		void computeRotation();
		void computeTranslation(cv::Mat &translation, cv::Mat &Epipole);
		virtual void postProcess(cv::Mat &disparity);
	public:
		SGMStereoFlow(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_,const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_, 
			cv::Mat &EpipoleLeftLast_, cv::Mat &EpipoleLeft_, cv::Mat &fundamentalMatrix_);
		void setAlphaRansac(cv::Mat &disparity, cv::Mat &disparityFlow, cv::Mat &disflag_);
		void setEvidence(cv::Mat &eviStereo_, cv::Mat &eviFlow_ ,cv::Mat &disflag_);		
	virtual ~SGMStereoFlow();


};


#endif
