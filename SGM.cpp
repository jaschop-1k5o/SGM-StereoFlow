#include "SGM.h"
#include "ransac.h"
#include <unistd.h>
#include "opencv2/photo.hpp"

SGM::SGM(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_):
imgLeftLast(imgLeftLast_), imgLeft(imgLeft_), imgRight(imgRight_), PENALTY1(PENALTY1_), PENALTY2(PENALTY2_), winRadius(winRadius_)
{
	this->WIDTH  = imgLeft.cols;
	this->HEIGHT = imgLeft.rows;
	censusImageLeft	 = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageRight = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	cost = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
	directCost = Mat::zeros(HEIGHT, WIDTH, CV_32SC(DISP_RANGE));
	accumulatedCost = Mat::zeros(HEIGHT, WIDTH, CV_32SC(DISP_RANGE));
std::cout<<"=========================================================="<<std::endl;
};

void SGM::writeDerivative(){}

void SGM::computeCensus(const cv::Mat &image, cv::Mat &censusImg){


	for (int y = winRadius; y < HEIGHT - winRadius; ++y) {
		for (int x = winRadius; x < WIDTH - winRadius; ++x) {
			unsigned char centerValue = image.at<uchar>(y,x);

			int censusCode = 0;
			for (int neiY = -winRadius; neiY <= winRadius; ++neiY) {
				for (int neiX = -winRadius; neiX <= winRadius; ++neiX) {
					censusCode = censusCode << 1;
					if (image.at<uchar>(y + neiY, x + neiX) >= centerValue) censusCode += 1;		
				}
			}
			
			censusImg.at<uchar>(y,x) = static_cast<unsigned char>(censusCode);
		}
	}
}


int SGM::computeHammingDist(const uchar left, const uchar right){

	int var = static_cast<int>(left ^ right);
	int count = 0;

	while(var){
		var = var & (var - 1);
		count++;
	}
	return count;
}




void SGM::sumOverAllCost(){

	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			accumulatedCost.at<SGM::VecDf>(y,x) += directCost.at<SGM::VecDf>(y,x);
			
		}
	}
}



void SGM::createDisparity(cv::Mat &disparity){
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float imax = std::numeric_limits<float>::max();
			int min_index = 0;
			SGM::VecDf vec = accumulatedCost.at<SGM::VecDf>(y,x);

			for(int d = 0; d < DISP_RANGE; d++){
				if(vec[d] < imax ){ imax = vec[d]; min_index = d;}
			}
			disparity.at<uchar>(y,x) = static_cast<uchar>(DIS_FACTOR*min_index);
		}
	}

}

void SGM::setPenalty(const int penalty_1, const int penalty_2){
	PENALTY1 = penalty_1;
	PENALTY2 = penalty_2;
}

void SGM::postProcess(cv::Mat &disparityIn, cv::Mat &disparity)
{
	disparity = disparityIn;
}

void SGM::runSGM(cv::Mat &disparity){
	std::cout<<"compute Census: ";
	computeCensus(imgLeft , censusImageLeft);
	computeCensus(imgLeftLast , censusImageLeftLast);
	computeCensus(imgRight, censusImageRight);
	std::cout<<"done"<<std::endl;
	std::cout<<"compute derivative: ";
	computeDerivative();
	std::cout<<"done"<<std::endl;
	std::cout<<"compute pixel-wise cost: ";
	computeCost();
	std::cout<<"done"<<std::endl;

	std::cout<<"aggregation starts:"<<std::endl;
	aggregation<1,0>(cost);
	sumOverAllCost();

	aggregation<0,1>(cost);
	sumOverAllCost();

	aggregation<0,-1>(cost);
	sumOverAllCost();

	aggregation<-1,0>(cost);
	sumOverAllCost();

/*	//---ENABLE FOR 8-PATH SGM
	aggregation<1,1>(cost);
	sumOverAllCost();

	aggregation<-1,1>(cost);
	sumOverAllCost();

	aggregation<1,-1>(cost);
	sumOverAllCost();

	aggregation<-1,-1>(cost);
	sumOverAllCost();
*/

/*	//---ENABLE FOR 16-PATH SGM
	aggregation<2,1>();
	sumOverAllCost();

	aggregation<2,-1>();
	sumOverAllCost();

	aggregation<-2,-1>();
	sumOverAllCost();

	aggregation<-2,1>();
	sumOverAllCost();

	aggregation<1,2>();
	sumOverAllCost();

	aggregation<-1,2>();
	sumOverAllCost();

	aggregation<1,-2>();
	sumOverAllCost();

	aggregation<-1,-2>();
	sumOverAllCost();
*/
	cv::Mat disparityTemp(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityTemp);
	postProcess(disparityTemp, disparity);

}

void SGM::computeCost(){}

SGM::~SGM(){
	censusImageRight.release();
	censusImageLeft.release();
	censusImageLeftLast.release();	
	cost.release();
	directCost.release();
	accumulatedCost.release();
}

void SGM::computeDerivative(){}

SGM::VecDf SGM::addPenalty(SGM::VecDf const& priorL,SGM::VecDf &localCost, float path_intensity_gradient ) {

	SGM::VecDf currL;
	float maxVal;

  	for ( int d = 0; d < DISP_RANGE; d++ ) {
    		float e_smooth = std::numeric_limits<float>::max();		
    		for ( int d_p = 0; d_p < DISP_RANGE; d_p++ ) {
      			if ( d_p - d == 0 ) {
        			// No penality
        			//e_smooth = std::min(e_smooth,priorL[d_p]);
				e_smooth = std::min(e_smooth,priorL[d]);
      			} else if ( abs(d_p - d) == 1 ) {
        			// Small penality
				e_smooth = std::min(e_smooth, priorL[d_p] + (PENALTY1));
      			} else {
        			// Large penality
				//maxVal=static_cast<float>(std::max((float)PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2));
        			//maxVal=std::max(PENALTY1, path_intensity_gradient ? PENALTY2/path_intensity_gradient : PENALTY2);				
				//e_smooth = std::min(e_smooth, (priorL[d_p] + maxVal));
				e_smooth = std::min(e_smooth, priorL[d_p] + PENALTY2);

      			}
    		}
    	currL[d] = localCost[d] + e_smooth;
  	}

	double minVal;
	cv::minMaxLoc(priorL, &minVal);

  	// Normalize by subtracting min of priorL cost
	for(int i = 0; i < DISP_RANGE; i++){
		currL[i] -= static_cast<float>(minVal);
	}

	return currL;
}

template <int DIRX, int DIRY>
void SGM::aggregation(cv::Mat cost) {

	if ( DIRX  == -1  && DIRY == 0) {
	std::cout<<"DIRECTION:(-1, 0) called,"<<std::endl;
      		// RIGHT MOST EDGE
      		for (int y = 0; y < HEIGHT; y++ ) {
			directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
			//std::cout<<y<<" :"<<directCost.at<SGM::VecDf>(y, WIDTH - 1)<<std::endl;
      		}
		//sleep(1);
      		for (int x = WIDTH - 2; x >= 0; x-- ) {
        		for ( int y = 0 ; y < HEIGHT ; y++ ) {
          			 directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
    	}	 

    	// Walk along the edges in a clockwise fashion
    	if ( DIRX == 1  && DIRY == 0) {
	std::cout<<"DIRECTION:( 1, 0) called,"<<std::endl;
      		// Process every pixel along this edge
     		for (int y = 0 ; y < HEIGHT ; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
			//std::cout<<y<<" :"<<directCost.at<SGM::VecDf>(y, 0)<<std::endl;			
    		}
		//sleep(1);
     		for (int x = 1 ; x < WIDTH; x++ ) {
      			for ( int y = 0; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 0 && DIRY == 1) {
	std::cout<<"DIRECTION:( 0, 1) called,"<<std::endl;
     		//TOP MOST EDGE	
      		for (int x = 0; x < WIDTH ; x++ ) {
			directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
			//std::cout<<x<<" :"<<directCost.at<SGM::VecDf>(0, x)<<std::endl;
			
      		}
		//sleep(1);
      		for (int y = 1 ; y < HEIGHT ; y++ ) {   
			for ( int x = 0; x < WIDTH; x++ ) {
          			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
    	} 
	
	if ( DIRX == 0 && DIRY == -1) {
	std::cout<<"DIRECTION:( 0,-1) called,"<<std::endl;
      		// BOTTOM MOST EDGE
     		 for ( int x = 0 ; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
			//std::cout<<x<<" :"<<directCost.at<SGM::VecDf>(0, x)<<std::endl;
      		}
		//sleep(1);
     		for ( int y = HEIGHT - 2; y >= 0; --y ) {
        		for ( int x = 0; x < WIDTH; x++ ) {
          			 directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
    	}

	if ( DIRX == 1  && DIRY == 1) {
	std::cout<<"DIRECTION:( 1, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 1; x < WIDTH; x++ ) {
      			for ( int y = 1; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == -1) {
	std::cout<<"DIRECTION:( 1,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 1; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 2; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == 1) {
	std::cout<<"DIRECTION:(-1, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 2; x >= 0; x-- ) {
      			for ( int y = 1; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == -1) {
	std::cout<<"DIRECTION:(-1,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 2; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 2; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 2  && DIRY == 1) {
	std::cout<<"DIRECTION:( 2, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == 2) {
	std::cout<<"DIRECTION:( 1, 2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -2  && DIRY == 1) {
	std::cout<<"DIRECTION:(-2, 1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					     // fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}


	if ( DIRX == -1  && DIRY == 2) {
	std::cout<<"DIRECTION:(-1, 2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(0, x) = cost.at<SGM::VecDf>(0, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = 2; y < HEIGHT; y++ ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 2  && DIRY == -1) {
	std::cout<<"DIRECTION:( 2,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 3; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == 1  && DIRY == -2) {
	std::cout<<"DIRECTION:( 1,-2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, 0) = cost.at<SGM::VecDf>(y, 0);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = 2; x < WIDTH; x++ ) {
      			for ( int y = HEIGHT - 3; y >= 0 ; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -2  && DIRY == -1) {
	std::cout<<"DIRECTION:(-2,-1) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 3; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}

	if ( DIRX == -1  && DIRY == -2) {
	std::cout<<"DIRECTION:(-1,-2) called,"<<std::endl;
      		for ( int y = 0; y < HEIGHT; y++ ) {
        		directCost.at<SGM::VecDf>(y, WIDTH - 1) = cost.at<SGM::VecDf>(y, WIDTH - 1);
    		}
		for ( int x = 0; x < WIDTH; x++ ) {
        		directCost.at<SGM::VecDf>(HEIGHT - 1, x) = cost.at<SGM::VecDf>(HEIGHT - 1, x);
    		}

     		for ( int x = WIDTH - 3; x >= 0; x-- ) {
      			for ( int y = HEIGHT - 3; y >= 0; y-- ) {
            			directCost.at<SGM::VecDf>(y,x) = addPenalty(directCost.at<SGM::VecDf>(y-DIRY,x-DIRX), cost.at<SGM::VecDf>(y,x), 0);
					      //fabs(derivativeLeft.at<float>(y,x)-derivativeLeft.at<float>(y-DIRY,x-DIRX)) );
        		}
      		}
	}
}


void SGMStereo::computeDerivative(){
	cv::Mat gradx(HEIGHT, WIDTH, CV_32FC1);
	cv::Sobel(imgLeft, derivativeStereoLeft, CV_32FC1,1,0);
	//derivativeStereoLeft=cv::abs(gradx);
	float sobelCapValue_ = 15;

	for (int y = 1; y < HEIGHT - 1; ++y) {
			for (int x = 1; x < WIDTH - 1; ++x) {
				float sobelValue = derivativeStereoLeft.at<float>(y,x);
				if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
				else if (sobelValue < -sobelCapValue_) sobelValue = 0;
				else sobelValue += sobelCapValue_;
				derivativeStereoLeft.at<float>(y,x) = sobelValue;
				//std::cout<<sobelValue<<std::endl;sleep(1);
			}
	}


	cv::Sobel(imgRight, derivativeStereoRight, CV_32FC1,1,0);
	//derivativeStereoRight=cv::abs(gradx);

	for (int y = 1; y < HEIGHT - 1; ++y) {
			for (int x = 1; x < WIDTH - 1; ++x) {
				float sobelValue = derivativeStereoRight.at<float>(y,x);
				if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
				else if (sobelValue < -sobelCapValue_) sobelValue = 0;
				else sobelValue += sobelCapValue_;
				derivativeStereoRight.at<float>(y,x) = sobelValue;
			}
	}


	gradx.release();
}

void SGMStereo::computeCost(){
	bool dispInvalid;
	float stereoAggrCost;
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			dispInvalid = false;
			for(int d = 0; d < DISP_RANGE; d++){
				if(dispInvalid){
					//save redundant computations: disparity(x) invalid => disparity(x') invalid for all x'>x
					abortNeiIteration:
					cost.at<SGM::VecDf>(y,x)[d] = stereoAggrCost;
					continue;
				}
				//reset aggregating variables
				stereoAggrCost = 0.0f;

				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						if(neiX-d >= winRadius){
							stereoAggrCost += fabs(derivativeStereoLeft.at<float>(neiY, neiX)- 
											    derivativeStereoRight.at<float>(neiY, neiX - d)) 
								+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), 
												censusImageRight.at<uchar>(neiY, neiX - d));
						}
						else{
							dispInvalid = true;
							stereoAggrCost = cost.at<SGM::VecDf>(y,x)[d-1];
							goto abortNeiIteration;
						}
					}
					
				}
				cost.at<SGM::VecDf>(y,x)[d] = stereoAggrCost;
			}
		}
	}
}

SGMStereo::SGMStereo(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_)
		:SGM(imgLeftLast_, imgLeft_, imgRight_, PENALTY1_, PENALTY2_, winRadius_){
		derivativeStereoLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeStereoRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
	}

SGMStereo::~SGMStereo(){
	derivativeStereoLeft.release();
	derivativeStereoRight.release();
}

void SGMStereo::postProcess(cv::Mat &disparityIn, cv::Mat &disparity){
	//denoising provided by OpenCV2
	fastNlMeansDenoising(disparityIn, disparity);

	/* NOT USED. BORDERS ARE RENDERED
	//Set disparity to 0 for borders & where full stereo cost could not be computed
	//left border (including stereo-blind area)
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < winRadius+DISP_RANGE; x++){
			disparity.at<uchar>(y,x) = static_cast<uchar>(0);
		}
	}
	//right border
	for(int y = 0; y < HEIGHT; y++){
		for(int x = WIDTH-winRadius; x < WIDTH; x++){
			disparity.at<uchar>(y,x) = static_cast<uchar>(0);
		}
	}
	//top border (w/o intersections)
	for(int y = 0; y < winRadius; y++){
		for(int x = winRadius+DISP_RANGE; x < WIDTH-winRadius; x++){
			disparity.at<uchar>(y,x) = static_cast<uchar>(0);
		}
	}
	//bottom border (w/o intersections)
	for(int y = HEIGHT-winRadius; y < HEIGHT; y++){
		for(int x = winRadius+DISP_RANGE; x < WIDTH-winRadius; x++){
			disparity.at<uchar>(y,x) = static_cast<uchar>(0);
		}
	}*/
}

void SGMStereo::writeDerivative(){
	imwrite("../derivativeStereoRight.jpg",derivativeStereoRight);
	imwrite("../derivativeStereoLeft.jpg",derivativeStereoLeft);
}

SGMFlow::SGMFlow(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_,
		 cv::Mat &EpipoleLeftLast_, cv::Mat &EpipoleLeft_, cv::Mat &fundamentalMatrix_)
		:SGM(imgLeftLast_, imgLeft_, imgRight_, PENALTY1_, PENALTY2_, winRadius_){
				
		EpipoleLeftLast_.copyTo(EpipoleLeftLast);
		EpipoleLeft_.copyTo(EpipoleLeft);
		fundamentalMatrix_.copyTo(fundamentalMatrix);
		imgRotation = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		computeRotation();

		translationLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		translationLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		derivativeFlowLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeFlowLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);

		disFlag = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
}

SGMStereoFlow::SGMStereoFlow(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_,
		 cv::Mat &EpipoleLeftLast_, cv::Mat &EpipoleLeft_, cv::Mat &fundamentalMatrix_)
		:SGM(imgLeftLast_, imgLeft_, imgRight_, PENALTY1_, PENALTY2_, winRadius_){
		//VVV STEREO variables VVV
		derivativeStereoLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeStereoRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);

		//VVV FLOW variables VVV
		EpipoleLeftLast_.copyTo(EpipoleLeftLast);
		EpipoleLeft_.copyTo(EpipoleLeft);
		fundamentalMatrix_.copyTo(fundamentalMatrix);
		imgRotation = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		computeRotation();

		translationLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		translationLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC2);
		derivativeFlowLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeFlowLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
}

SGMFlow::~SGMFlow(){
	imgRotation.release();
	EpipoleLeftLast.release();
	EpipoleLeft.release();
	translationLeftLast.release();
	translationLeft.release();
	fundamentalMatrix.release();
	disFlag.release();
	derivativeFlowLeftLast.release();
	derivativeFlowLeft.release();
}

SGMStereoFlow::~SGMStereoFlow(){
	imgRotation.release();
	EpipoleLeftLast.release();
	EpipoleLeft.release();
	translationLeftLast.release();
	translationLeft.release();
	fundamentalMatrix.release();
	derivativeStereoLeft.release();
	derivativeStereoRight.release();
	derivativeFlowLeftLast.release();
	derivativeFlowLeft.release();
}

void _computeRotation(cv::Mat &imgRotation, cv::Mat &fundamentalMatrix, int WIDTH, int HEIGHT){
	std::cout<<"Compute rotation: ";
	const double cx = (double)WIDTH/2.0;
	const double cy = (double)HEIGHT/2.0;
	cv::Mat A(WIDTH*HEIGHT,5,CV_64FC1,cv::Scalar(0.0));
	cv::Mat b(WIDTH*HEIGHT,1,CV_64FC1,cv::Scalar(0.0));
	cv::Mat x_sol;
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			double x_, y_;
			cv::Mat hom(3,1,CV_64FC1);
			x_ = x - cx;
			y_ = y - cy;
			//generate homogenous coord
			hom.at<double>(0,0) = (double)x;
			hom.at<double>(1,0) = (double)y;
			hom.at<double>(2,0) = 1.0;
			//calc epiline through pixel
			cv::Mat epi = fundamentalMatrix*hom;
			A.at<double>(y*WIDTH+x,0) = epi.at<double>(0,0);
			A.at<double>(y*WIDTH+x,1) = epi.at<double>(1,0);
			A.at<double>(y*WIDTH+x,2) = (epi.at<double>(1,0)*x_)-(epi.at<double>(0,0)*y_);
			A.at<double>(y*WIDTH+x,3) = (epi.at<double>(0,0)*x_*x_)+(epi.at<double>(1,0)*x_*y_);
			A.at<double>(y*WIDTH+x,4) = (epi.at<double>(0,0)*x_*y_)+(epi.at<double>(1,0)*y_*y_);
			b.at<double>(y*WIDTH+x,0) = -epi.at<double>(2,0)-(epi.at<double>(0,0)*x)-(epi.at<double>(1,0)*y);
		}
	}
	cv::solve(A,b,x_sol,DECOMP_QR);
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float x_, y_;
			x_ = x - cx;
			y_ = y - cy;
			
			imgRotation.at<Vec2f>(y,x)[0] = (float)(x_sol.at<double>(0,0)-(x_sol.at<double>(2,0)*y_)+(x_sol.at<double>(3,0)*x_*x_)+(x_sol.at<double>(4,0)*x_*y_));
			imgRotation.at<Vec2f>(y,x)[1] = (float)(x_sol.at<double>(1,0)+(x_sol.at<double>(2,0)*x_)+(x_sol.at<double>(3,0)*x_*y_)+(x_sol.at<double>(4,0)*y_*y_));
		}	
	}

	A.release();
	b.release();	
	x_sol.release();
	std::cout<<"done"<<std::endl;
}

void SGMFlow::computeRotation(){
	_computeRotation(imgRotation,fundamentalMatrix,WIDTH,HEIGHT);
}

void SGMStereoFlow::computeRotation(){
	_computeRotation(imgRotation,fundamentalMatrix,WIDTH,HEIGHT);
}

void SGMStereoFlow::setAlphaRansac(cv::Mat &disparity, cv::Mat &disparityFlow, cv::Mat &disFlag_)
{
	std::cout<<"Compute alpha: ";
	ransacAlpha = ransac(disparity,disparityFlow,disFlag_);
	std::cout<<"done"<<std::endl;
	std::cout<<"alpha="<<ransacAlpha<<std::endl;
}

void SGMStereoFlow::setEvidence(cv::Mat &eviStereo_, cv::Mat &eviFlow_ ,cv::Mat &disFlag_)
{
	eviStereo = eviStereo_;
	eviFlow = eviFlow_;
	disFlag = disFlag_;
}

void _computeTranslation(cv::Mat &translation, cv::Mat &Epipole, int WIDTH, int HEIGHT){
	for ( int x = 0; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {
			float delta_x = (x - Epipole.at<float>(0));
			float delta_y = (y - Epipole.at<float>(1));
			float nomi = sqrt(delta_x*delta_x + delta_y*delta_y);
			float dir_x = (delta_x/nomi);
			float dir_y = (delta_y/nomi);
			translation.at<Vec2f>(y,x)[0]=dir_x;
			translation.at<Vec2f>(y,x)[1]=dir_y;
		}
	}

}

void SGMFlow::computeTranslation(cv::Mat &translation, cv::Mat &Epipole){
	_computeTranslation(translation,Epipole,WIDTH,HEIGHT);
}

void SGMStereoFlow::computeTranslation(cv::Mat &translation, cv::Mat &Epipole){
	_computeTranslation(translation,Epipole,WIDTH,HEIGHT);
}


void SGMFlow::computeDerivative(){
	float sobelCapValue_ = 15;
	cv::Mat gradxLeftLast, gradyLeftLast;
	cv::Sobel(imgLeftLast, gradxLeftLast, CV_32FC1, 1, 0);

	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradxLeftLast.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradxLeftLast.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}


	cv::Sobel(imgLeftLast, gradyLeftLast, CV_32FC1, 0, 1);

	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradyLeftLast.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradyLeftLast.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}
	cv::Mat gradxLeft, gradyLeft;
	cv::Sobel(imgLeft, gradxLeft, CV_32FC1, 1, 0);
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradxLeft.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradxLeft.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}

	cv::Sobel(imgLeft, gradyLeft, CV_32FC1, 0, 1);
	 for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradyLeft.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradyLeft.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}
	
	//computeTranslation(translationLeftLast, EpipoleLeftLast);
	computeTranslation(translationLeft, EpipoleLeft);


	for ( int x = 0; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {

			derivativeFlowLeftLast.at<float>(y,x) = static_cast<float>(sqrt(pow(translationLeft.at<Vec2f>(y,x)[1]*gradyLeftLast.at<float>(y,x),2)+
										pow(translationLeft.at<Vec2f>(y,x)[0]*gradxLeftLast.at<float>(y,x),2)));

			derivativeFlowLeft.at<float>(y,x) = static_cast<float>(sqrt(pow(translationLeft.at<Vec2f>(y,x)[1]*gradyLeft.at<float>(y,x),2)+
										 pow(translationLeft.at<Vec2f>(y,x)[0]*gradxLeft.at<float>(y,x),2)));	
	
		}
	}

	gradxLeftLast.release();
	gradyLeftLast.release();
	gradxLeft.release();
	gradyLeft.release();


}

void SGMStereoFlow::computeDerivative(){
	//--- STEREO DERIVATIVES ---
	cv::Mat gradx(HEIGHT, WIDTH, CV_32FC1);
	cv::Sobel(imgLeft, derivativeStereoLeft, CV_32FC1,1,0);
	//derivativeStereoLeft=cv::abs(gradx);
	float sobelCapValue_ = 15;

	for (int y = 1; y < HEIGHT - 1; ++y) {
			for (int x = 1; x < WIDTH - 1; ++x) {
				float sobelValue = derivativeStereoLeft.at<float>(y,x);
				if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
				else if (sobelValue < -sobelCapValue_) sobelValue = 0;
				else sobelValue += sobelCapValue_;
				derivativeStereoLeft.at<float>(y,x) = sobelValue;
				//std::cout<<sobelValue<<std::endl;sleep(1);
			}
	}


	cv::Sobel(imgRight, derivativeStereoRight, CV_32FC1,1,0);
	//derivativeStereoRight=cv::abs(gradx);

	for (int y = 1; y < HEIGHT - 1; ++y) {
			for (int x = 1; x < WIDTH - 1; ++x) {
				float sobelValue = derivativeStereoRight.at<float>(y,x);
				if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
				else if (sobelValue < -sobelCapValue_) sobelValue = 0;
				else sobelValue += sobelCapValue_;
				derivativeStereoRight.at<float>(y,x) = sobelValue;
			}
	}


	gradx.release();

	//--- FLOW DERIVATIVES ---
	cv::Mat gradxLeftLast, gradyLeftLast;
	cv::Sobel(imgLeftLast, gradxLeftLast, CV_32FC1, 1, 0);

	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradxLeftLast.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradxLeftLast.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}


	cv::Sobel(imgLeftLast, gradyLeftLast, CV_32FC1, 0, 1);

	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradyLeftLast.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradyLeftLast.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}
	cv::Mat gradxLeft, gradyLeft;
	cv::Sobel(imgLeft, gradxLeft, CV_32FC1, 1, 0);
	for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradxLeft.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradxLeft.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}

	cv::Sobel(imgLeft, gradyLeft, CV_32FC1, 0, 1);
	 for (int y = 1; y < HEIGHT - 1; ++y) {
		for (int x = 1; x < WIDTH - 1; ++x) {
			float sobelValue = gradyLeft.at<float>(y,x);
			if (sobelValue > sobelCapValue_) sobelValue = 2*sobelCapValue_;
			else if (sobelValue < -sobelCapValue_) sobelValue = 0;
			else sobelValue += sobelCapValue_;
			gradyLeft.at<float>(y,x) = sobelValue;
			//std::cout<<sobelValue<<std::endl;sleep(1);
		}
	}
	
	//computeTranslation(translationLeftLast, EpipoleLeftLast);
	computeTranslation(translationLeft, EpipoleLeft);


	for ( int x = 0; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {

			derivativeFlowLeftLast.at<float>(y,x) = static_cast<float>(sqrt(pow(translationLeft.at<Vec2f>(y,x)[1]*gradyLeftLast.at<float>(y,x),2)+
										pow(translationLeft.at<Vec2f>(y,x)[0]*gradxLeftLast.at<float>(y,x),2)));

			derivativeFlowLeft.at<float>(y,x) = static_cast<float>(sqrt(pow(translationLeft.at<Vec2f>(y,x)[1]*gradyLeft.at<float>(y,x),2)+
										 pow(translationLeft.at<Vec2f>(y,x)[0]*gradxLeft.at<float>(y,x),2)));	
	
		}
	}

	gradxLeftLast.release();
	gradyLeftLast.release();
	gradxLeft.release();
	gradyLeft.release();
}

void SGMFlow::computeCost(){
	bool dispInvalid;
	float flowAggrCost;
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			dispInvalid = false;
			for(int w = 0; w < DISP_RANGE ; w++){
				if(dispInvalid){
					//save redundant computations: disparity(x) invalid => disparity(x') invalid for all x'>x
					abortNeiIteration:
					cost.at<SGM::VecDf>(y,x)[w] = flowAggrCost;
					continue;
				}

				flowAggrCost = 0.0f;
				//convert w to VZ-ratio
				float w_ = Vmax*(float)w/DISP_RANGE;

				for(int neiY = y - winRadius ; neiY <= y + winRadius; neiY++){				
					for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
						float newx = neiX+imgRotation.at<Vec2f>(neiY,neiX)[0];
						float newy = neiY+imgRotation.at<Vec2f>(neiY,neiX)[1];
						float distx = newx - EpipoleLeft.at<float>(0);
						float disty = newy - EpipoleLeft.at<float>(1);
						

						float L = sqrt(distx*distx + disty*disty);
						float d = L * w_/(1.0f-w_);
						
						
						int xx = round(newx + d*translationLeft.at<Vec2f>(newy,newx)[0]);
						int yy = round(newy + d*translationLeft.at<Vec2f>(newy,newx)[1]);

						if((xx>=winRadius) && (yy>=winRadius) && xx<(WIDTH-winRadius) && yy< (HEIGHT-winRadius)){	
							flowAggrCost += fabs(derivativeFlowLeftLast.at<float>(neiY,neiX)
							  - derivativeFlowLeft.at<float>(yy,xx)) + (float)CENSUS_W
							  * computeHammingDist(censusImageLeftLast.at<uchar>(neiY, neiX),
							  censusImageLeft.at<uchar>(yy, xx));
						}else{
							dispInvalid = true;
							disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
							flowAggrCost = cost.at<SGM::VecDf>(y,x)[w - 1];
							goto abortNeiIteration;
						}
					}
				}
				cost.at<SGM::VecDf>(y,x)[w] = flowAggrCost;
			}
		}
	}

	/* NOT USED. BORDERS ARE RENDERED
	//Set non-full costs to zero (border values are 0 by default)
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			if(disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){	
				for(int w = 0; w < DISP_RANGE ; w++){
					cost.at<SGM::VecDf>(y,x)[w] = 0.0;
				}
			}
		}
	}*/
}

void SGMFlow::postProcess(cv::Mat &disparityIn, cv::Mat &disparity){
	//denoising provided by OpenCV2
	fastNlMeansDenoising(disparityIn, disparity);

	/* NOT USED. BORDERS ARE RENDERED
	//Set disparity to 0 for borders & where full flow cost could not be computed
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			if(x < winRadius || x >= WIDTH-winRadius || y < winRadius || y >= HEIGHT-winRadius
				|| disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){	
				disparity.at<uchar>(y,x) = static_cast<uchar>(0);
			}
		}
	}*/
}

void SGMFlow::writeDerivative(){
	imwrite("../derivativeFlowLeft.jpg",derivativeFlowLeft);
	imwrite("../derivativeFlowLeftLast.jpg",derivativeFlowLeftLast);
}


void SGMFlow::copyDisflag(cv::Mat &M){

	disFlag.copyTo(M);

}

void SGMStereoFlow::computeCost(){
	bool dispInvalid,stereoInvalid,flowInvalid;
	float stereoLastCost,stereoAggrCost,flowLastCost,flowAggrCost;
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			dispInvalid = stereoInvalid = flowInvalid = false;
			stereoLastCost = flowLastCost = 0.0f;
			for(int d_st = 0; d_st < DISP_RANGE; d_st++){
				if(dispInvalid){
					//save redundant computations: disparity(x) invalid => disparity(x') invalid for all x'>x
					abortNeiIteration:
					cost.at<SGM::VecDf>(y,x)[d_st] = stereoLastCost + flowLastCost;
					continue;
				}
				//reset aggregating variables
				stereoAggrCost = flowAggrCost = 0.0f;
				//convert Stereo-disparity to VZ-ratio
				float w = d_st*(x*ransacAlpha[0]+y*ransacAlpha[1]+ransacAlpha[2]);
				w = (Vmax*w)/DISP_RANGE;

				for(int neiY = y - winRadius ; neiY <= y + winRadius; neiY++){				
					for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
						//check if stereo projection still in (census-)image
						stereoInvalid = stereoInvalid || neiX-d_st < winRadius;
						if(!stereoInvalid){
							stereoAggrCost += fabs(derivativeStereoLeft.at<float>(neiY, neiX)- 
								derivativeStereoRight.at<float>(neiY, neiX - d_st)) 
								+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), 
								censusImageRight.at<uchar>(neiY, neiX - d_st));
						}

						int xx,yy;
						if(!flowInvalid){
							//apply image rotation
							float newx = neiX+imgRotation.at<Vec2f>(neiY,neiX)[0];
							float newy = neiY+imgRotation.at<Vec2f>(neiY,neiX)[1];

							//compute Flow-disparity
							float distx = newx - EpipoleLeft.at<float>(0);
							float disty = newy - EpipoleLeft.at<float>(1);
							float L = sqrt(distx*distx + disty*disty);
							float d_fl = L * w/(1.0-w);

							//projected flow point
							xx = round(newx + d_fl*translationLeft.at<Vec2f>(newy,newx)[0]);
							yy = round(newy + d_fl*translationLeft.at<Vec2f>(newy,newx)[1]);

							//check if flow projection still in (census-)image
							flowInvalid = (xx<winRadius)||(yy<winRadius)||(xx>=WIDTH-winRadius)||(yy>=HEIGHT-winRadius);
						}
						if(!flowInvalid){
							flowAggrCost += fabs(derivativeFlowLeftLast.at<float>(neiY,neiX)
								- derivativeFlowLeft.at<float>(yy,xx))
								+ (float)CENSUS_W * computeHammingDist(censusImageLeftLast.at<uchar>(neiY, neiX), 
								censusImageLeft.at<uchar>(yy, xx));
						}
						else if(stereoInvalid){
							//cost not computeable for current disparity, abort neighbour iteration
							dispInvalid = true;
							goto abortNeiIteration;
						}
					}
				}
				//enter aggregated cost into cost field
				if(!stereoInvalid)
					stereoLastCost = stereoAggrCost;
				cost.at<SGM::VecDf>(y,x)[d_st] += stereoLastCost;

				if(!flowInvalid)
					flowLastCost = flowAggrCost;
				cost.at<SGM::VecDf>(y,x)[d_st] += flowLastCost;
			}
		}
	}
}

void SGMStereoFlow::postProcess(cv::Mat &disparityIn, cv::Mat &disparity)
{
	//denoising provided by OpenCV2
	fastNlMeansDenoising(disparityIn, disparity);

	/* NOT USED. BORDERS ARE RENDERED
	//Set disparity to 0 for borders & where neither stereo nor flow cost could not be computed
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			if(x < winRadius || x >= WIDTH-winRadius || y < winRadius || y >= HEIGHT-winRadius
				|| (x < winRadius+DISP_RANGE && disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG))){	
				disparity.at<uchar>(y,x) = static_cast<uchar>(0);
			}
		}
	}*/
}
