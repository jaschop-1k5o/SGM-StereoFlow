#include "SGM.h"
//#include "ransac.h"
#include <unistd.h>

SGM::SGM(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_):
imgLeftLast(imgLeftLast_), imgLeft(imgLeft_), imgRight(imgRight_), PENALTY1(PENALTY1_), PENALTY2(PENALTY2_), winRadius(winRadius_)
{
	this->WIDTH  = imgLeft.cols;
	this->HEIGHT = imgLeft.rows;
	censusImageLeft	 = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageRight = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	censusImageLeftLast = Mat::zeros(HEIGHT, WIDTH, CV_8UC1);
	cost = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
	costRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC(DISP_RANGE));
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
			//std::cout<<"x: "<<x<<" y: "<<y<<" "<<imax<<" |";
			//if(imax > Disthreshold){min_index = Outlier; /*std::cout<<"x: "<<x<<" y: "<<y<<" "<<imax<<" |";*/}
			disparity.at<uchar>(y,x) = static_cast<uchar>(DIS_FACTOR*min_index);
		}
	}

}

void SGM::setPenalty(const int penalty_1, const int penalty_2){
	PENALTY1 = penalty_1;
	PENALTY2 = penalty_2;
}

void SGM::postProcess(cv::Mat &disparity){}

void SGM::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity){}	

void SGM::resetDirAccumulatedCost(){}

void SGMFlow::resetDirAccumulatedCost(){

for (int x = 0 ; x < WIDTH; x++ ) {
      		for ( int y = 0; y < HEIGHT; y++ ) {
			if(disFlag.at<uchar>(y,x)==static_cast<uchar>(DISFLAG)){
				for(int d = 0; d < DISP_RANGE; d++){
            				directCost.at<SGM::VecDf>(y,x)[d] = 0.0;
				}
			}
					      
        	}
      	}


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
	resetDirAccumulatedCost();
	sumOverAllCost();

std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,1>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();

std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,-1>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<-1,0>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<1,1>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;	
	aggregation<-1,1>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<1,-1>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<-1,-1>(cost);
resetDirAccumulatedCost();
	sumOverAllCost();

/*
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
	createDisparity(disparity);
	postProcess(disparity);
/*	cv::Mat disparityLeft(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityLeft);
	postProcess(disparityLeft);
	imwrite("../disparityLeft.jpg", disparityLeft);


for(int y = 0; y < HEIGHT ; y++){
	for(int x = 0; x < WIDTH; x++){
		for(int d = 0; d < DISP_RANGE; d++){
			accumulatedCost.at<SGM::VecDf>(y,x)[d]=0.f;
			directCost.at<SGM::VecDf>(y,x)[d]=0.f;
		}
	}
}


	std::cout<<"compute costRight"<<std::endl;
	computeCostRight();
	std::cout<<"done"<<std::endl;

	std::cout<<"aggregation starts:"<<std::endl;
	aggregation<1,0>(costRight);
	resetDirAccumulatedCost();
	sumOverAllCost();

std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,1>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();

std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<0,-1>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<-1,0>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;

	aggregation<1,1>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;	
	aggregation<-1,1>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<1,-1>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();
std::cout<<accumulatedCost.at<SGM::VecDf>(250,126)<<std::endl;
	aggregation<-1,-1>(costRight);
resetDirAccumulatedCost();
	sumOverAllCost();


	cv::Mat disparityRight(HEIGHT, WIDTH, CV_8UC1);
	createDisparity(disparityRight);
	imshow("disparityRight", disparityRight);
	imwrite("../disparityRight.jpg", disparityRight);
	consistencyCheck(disparityLeft, disparityRight, disparity);

*/
	cv::Mat disparityRight =cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/disparityRight.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	cv::Mat disparityLeft =cv::imread("/home/sanyu/spsstereo/sanyu_local/sgm_lib/disparityLeft.jpg",CV_LOAD_IMAGE_GRAYSCALE);
	consistencyCheck(disparityLeft, disparityRight, disparity);

}

void SGM::computeCost(){}

void SGM::computeCostRight(){}

SGM::~SGM(){
	censusImageRight.release();
	censusImageLeft.release();
	censusImageLeftLast.release();	
	cost.release();
	costRight.release();
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

void SGMStereo::calcHalfPixelRight() {

	for(int y = 0; y < HEIGHT; ++y){
		for (int x = 0; x < WIDTH; ++x) {
			float centerValue = derivativeStereoRight.at<float>(y,x);
			float leftHalfValue = x > 0 ? (centerValue + derivativeStereoRight.at<float>(y,x-1))/2 : centerValue;
			float rightHalfValue = x < WIDTH - 1 ? (centerValue + derivativeStereoRight.at<float>(y,x+1))/2 : centerValue;
			float minValue = std::min(leftHalfValue, rightHalfValue);
			minValue = std::min(minValue, centerValue);
			float maxValue = std::max(leftHalfValue, rightHalfValue);
			maxValue = std::max(maxValue, centerValue);

			halfPixelRightMin.at<float>(y,x)=minValue;
			halfPixelRightMax.at<float>(y,x) = maxValue;
			//std::cout<<halfPixelRightMin.at<float>(y,x)<<" "<<halfPixelRightMax.at<float>(y,x)<<std::endl;
			//sleep(1);
		}
	}
}

void SGMStereo::consistencyCheck(cv::Mat disparityLeft, cv::Mat disparityRight, cv::Mat disparity){

//based on right
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = DISP_RANGE; x < WIDTH - winRadius - DISP_RANGE; x++){
			unsigned short disparityRightValue =  static_cast<unsigned short>(disparityRight.at<uchar>(y,x));
			unsigned short disparityLeftValue =  static_cast<unsigned short>(disparityLeft.at<uchar>(y,x + disparityRightValue));
			disparity.at<uchar>(y,x) = static_cast<uchar>(disparityRightValue*3);
			if(abs(disparityRightValue - disparityLeftValue) > disparityThreshold){
				disparity.at<uchar>(y,x) = static_cast<uchar>(Dinvd);
			}		
		}
	}

}

void SGMStereo::computeCostRight(){

for(int y = winRadius; y < HEIGHT - winRadius; y++){
	for(int x = winRadius; x < WIDTH - winRadius - DISP_RANGE; x++){
		for(int d = 0; d < DISP_RANGE; d++){
				
				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						costRight.at<SGM::VecDf>(y,x)[d] += fabs(derivativeStereoRight.at<float>(neiY, neiX)- 
										    derivativeStereoLeft.at<float>(neiY, neiX + d)) 
							+ CENSUS_W * computeHammingDist(censusImageRight.at<uchar>(neiY, neiX), censusImageLeft.at<uchar>(neiY, neiX + d));

					}
					
				}
			}

		}
	}



}

void SGMStereo::computeCost(){
	
//Mutual information matching
/*calcHalfPixelRight();
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = DISP_RANGE + winRadius; x < WIDTH - winRadius; x++){
			for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
			for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
			float leftCenterValue = derivativeStereoLeft.at<float>(neiY,neiX);
			float leftHalfLeftValue = x > 0 ? (leftCenterValue + derivativeStereoLeft.at<float>(neiY,neiX-1))/2 : leftCenterValue;
			float leftHalfRightValue = x < WIDTH - 1 ? (leftCenterValue + derivativeStereoLeft.at<float>(neiY,neiX+1))/2 : leftCenterValue;
			float leftMinValue = std::min(leftHalfLeftValue, leftHalfRightValue);
			leftMinValue = std::min(leftMinValue, leftCenterValue);
			float leftMaxValue = std::max(leftHalfLeftValue, leftHalfRightValue);
			leftMaxValue = std::max(leftMaxValue, leftCenterValue);
			for(int d = 0; d < DISP_RANGE; d++){

				float rightCenterValue = derivativeStereoRight.at<float>(neiY, neiX - d);
				float rightMinValue = halfPixelRightMin.at<float>(neiY, neiX - d);
				float rightMaxValue = halfPixelRightMax.at<float>(neiY, neiX - d);

				float costLtoR = std::max(0.f, leftCenterValue - rightMaxValue);
				costLtoR = std::max(costLtoR, rightMinValue - leftCenterValue);
				float costRtoL = std::max(0.f, rightCenterValue - leftMaxValue);
				costRtoL = std::max(costRtoL, leftMinValue - rightCenterValue);
				float costValue = std::min(costLtoR, costRtoL);

				cost.at<SGM::VecDf>(y,x)[d] += costValue+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), censusImageRight.at<uchar>(neiY, neiX - d));
				
			}
			}
			}
			
		}
	}

*/
/*	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int d = 0; d < DISP_RANGE; d++){
			for(int x = d + winRadius; x < WIDTH - winRadius; x++){
				
				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						cost.at<SGM::VecDf>(y,x)[d] += fabs(derivativeLeft.at<float>(y, x) - derivativeRight.at<float>(neiY, neiX - d)); 
							//+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(y, x), censusImageRight.at<uchar>(y, x - d));

					}

				}
					//std::cout<<cost.at<SGM::VecDf>(y,x)[d]<<std::endl;
					//std::cout<<derivativeLeft.at<float>(y, x)<<std::endl;
					//sleep(1);
			}
		}
	}
*/
//pixel intensity matching
for(int y = winRadius; y < HEIGHT - winRadius; y++){
	for(int x = DISP_RANGE + winRadius; x < WIDTH - winRadius; x++){
		for(int d = 0; d < DISP_RANGE; d++){
				
				for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
					for(int neiY = y - winRadius; neiY <= y + winRadius; neiY++){
						cost.at<SGM::VecDf>(y,x)[d] += fabs(derivativeStereoLeft.at<float>(neiY, neiX)- 
										    derivativeStereoRight.at<float>(neiY, neiX - d)) 
							+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), censusImageRight.at<uchar>(neiY, neiX - d));
						//std::cout<<derivativeStereoLeft.at<float>(y, x)<<std::endl;
						//sleep(1);
					}
					
				}
			}
			//std::cout<<cost.at<SGM::VecDf>(y,x)[0]<<std::endl;
			//		std::cout<<derivativeStereoLeft.at<float>(y, x)<<std::endl;
			//		sleep(1);
		}
	}

/*	for(int y = 0; y < HEIGHT -1; y++){
		for(int x = 0; x < DISP_RANGE - 1; x++){
			for(int d = x + 1; d < DISP_RANGE; d++){
				cost.at<SGM::VecDf>(y,x)[d] = cost.at<SGM::VecDf>(y,x)[x];
			}
		}
	}
*/
}

SGMStereo::SGMStereo(const cv::Mat &imgLeftLast_, const cv::Mat &imgLeft_, const cv::Mat &imgRight_, const int PENALTY1_, const int PENALTY2_, const int winRadius_)
		:SGM(imgLeftLast_, imgLeft_, imgRight_, PENALTY1_, PENALTY2_, winRadius_){
		derivativeStereoLeft = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		derivativeStereoRight = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		halfPixelRightMin = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
		halfPixelRightMax = Mat::zeros(HEIGHT, WIDTH, CV_32FC1);
	}

SGMStereo::~SGMStereo(){
	derivativeStereoLeft.release();
	derivativeStereoRight.release();
}

void SGMStereo::postProcess(cv::Mat &disparity){
	
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < DISP_RANGE; x++){
			disparity.at<uchar>(y,x)=static_cast<uchar>(0);
		}
	}

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
	disFlag.release();
	derivativeStereoLeft.release();
	derivativeStereoRight.release();
	derivativeFlowLeftLast.release();
	derivativeFlowLeft.release();
}

void SGMFlow::computeRotation(){

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
	
	std::cout<<"Rotation coef: "<<x_sol<<std::endl;
	for(int y = 0; y < HEIGHT; y++){
		for(int x = 0; x < WIDTH; x++){
			float x_, y_;
			x_ = x - cx;
			y_ = y - cy;
			
			imgRotation.at<Vec2f>(y,x)[0] = (float)(x_sol.at<double>(0,0)-(x_sol.at<double>(2,0)*y_)+(x_sol.at<double>(3,0)*x_*x_)+(x_sol.at<double>(4,0)*x_*y_));
			imgRotation.at<Vec2f>(y,x)[1] = (float)(x_sol.at<double>(1,0)+(x_sol.at<double>(2,0)*x_)+(x_sol.at<double>(3,0)*x_*y_)+(x_sol.at<double>(4,0)*y_*y_));
			//std::cout<<"loop(y="<<y<<",x="<<x<<") finished";
		}	
		//std::cout<<"loop(y="<<y<<") finished\n";	
	}

	A.release();
	b.release();	
	x_sol.release();
	std::cout<<"ComputeRotation finished";
}

void SGMStereoFlow::computeRotation(){
	//identical to SGMFlow::computeRotation()
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
	
	std::cout<<"Rotation coef: "<<x_sol<<std::endl;
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
}

void SGMStereoFlow::setAlphaRansac(cv::Mat &disparity, cv::Mat &disparityFLlow, cv::Mat &disflag_)
{
	//WHILE PCL BROKEN: set alpha(x,y) := 1
	ransacAlpha = cv::Vec3f(0.0,0.0,1.0);
}

void SGMStereoFlow::setEvidence(cv::Mat &eviStereo_, cv::Mat &eviFlow_ ,cv::Mat &disFlag_)
{
	eviStereo = eviStereo_;
	eviFlow = eviFlow_;
	disFlag = disFlag_;
}

void SGMFlow::computeTranslation(cv::Mat &translation, cv::Mat &Epipole){
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

void SGMStereoFlow::computeTranslation(cv::Mat &translation, cv::Mat &Epipole){
	//identical to SGMFlow::computeTranslation()
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
	//TODO
}

void SGMFlow::computeCost(){
	
//Scheme 1
/*
for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){

			float newx = x+imgRotation.at<Vec2f>(y,x)[0];
			float newy = y+imgRotation.at<Vec2f>(y,x)[1];
			float distx = newx - EpipoleLeft.at<float>(0);
			float disty = newy - EpipoleLeft.at<float>(1);
			float L = sqrt(distx*distx + disty*disty);			
			
		
			for(int w = 0; w < DISP_RANGE ; w++){
				float d = L * ((float)Vmax*(float)w/DISP_RANGE)/(1.0-((float)Vmax*(float)w/DISP_RANGE));
				int xx = (newx + d*translationLeft.at<Vec2f>(newy,newx)[0]);
				int yy = (newy + d*translationLeft.at<Vec2f>(newy,newx)[1]);
				for(int neiY = -winRadius ; neiY <= winRadius; neiY++){				
					for(int neiX = -winRadius; neiX <= winRadius; neiX++){

				
						if(((xx+neiX)>=winRadius) && ((yy+neiY)>=winRadius) && (xx+neiX)<(WIDTH-winRadius) && (yy+neiY)< (HEIGHT-winRadius)){	
							
							cost.at<SGM::VecDf>(y,x)[w] += fabs(derivativeFlowLeftLast.at<float>(y+neiY,x+neiX) - derivativeFlowLeft.at<float>(yy+neiY,xx+neiX))						
							+ (float)CENSUS_W * computeHammingDist(censusImageLeftLast.at<uchar>(y+neiY,x+neiX), censusImageLeft.at<uchar>(yy+neiY,xx+neiX));
							
										
						}else{
							disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
						}
				
					}
				}
			}
		}
	}
*/

//Scheme 2
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){

			for(int w = 0; w < DISP_RANGE ; w++){
			//for(int d_st = 0; d < DISP_RANGE; d++){ //use disp. range from stereo //STEREO_FLOW
				
				for(int neiY = y - winRadius ; neiY <= y + winRadius; neiY++){				
					for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
						float newx = neiX+imgRotation.at<Vec2f>(neiY,neiX)[0];
						float newy = neiY+imgRotation.at<Vec2f>(neiY,neiX)[1];
						float distx = newx - EpipoleLeft.at<float>(0);
						float disty = newy - EpipoleLeft.at<float>(1);
						

						float L = sqrt(distx*distx + disty*disty);
						//float w = d_st*computeAlpha(neix,neiy,alpha); //STEREO_FLOW
						float d = L * ((float)Vmax*(float)w/DISP_RANGE)/(1.0-((float)Vmax*(float)w/DISP_RANGE));
						
						
						int xx = round(newx + d*translationLeft.at<Vec2f>(newy,newx)[0]);
						int yy = round(newy + d*translationLeft.at<Vec2f>(newy,newx)[1]);

						//if(inScope(neiX + d_st,neiY))				       //STEREO_FLOW
						//	cost.at<...>(y,x)[d_st] += STEREO_COST(neiX,neiY,d_st) //STEREO_FLOW

						if((xx>=winRadius) && (yy>=winRadius) && xx<(WIDTH-winRadius) && yy< (HEIGHT-winRadius)){	
							
							cost.at<SGM::VecDf>(y,x)[w] += fabs(derivativeFlowLeftLast.at<float>(neiY,neiX) - derivativeFlowLeft.at<float>(yy,xx))						
							+ (float)CENSUS_W * computeHammingDist(censusImageLeftLast.at<uchar>(neiY, neiX), censusImageLeft.at<uchar>(yy, xx));
										
						}else{
							disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
						}
				
					}
				}
			}
		}
	}
		
//Set flag for image boundaries
	for(int y = 0; y < winRadius; y++){
		for(int x = 0; x < WIDTH; x++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}
	for(int y = HEIGHT - 1; y > HEIGHT - 1 - winRadius; y--){
		for(int x = 0; x < WIDTH; x++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = 0; x < winRadius; x++){
		for(int y = 0; y < HEIGHT; y++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = WIDTH - 1; x > WIDTH -1 - winRadius; x--){
		for(int y = 0; y < HEIGHT; y++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

//Set non-full costs to zero
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			if(disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){	
				for(int w = 0; w < DISP_RANGE ; w++){
					cost.at<SGM::VecDf>(y,x)[w] = 0.0;
				}
			}
		}
	}

	

/*	for(int x = winRadius; x < WIDTH - winRadius; x++){
		for(int y = winRadius; y < HEIGHT - winRadius; y++){
			//if(disFlag.at<uchar>(y,x) != static_cast<uchar>(DISFLAG)){
				for(int w = 1; w < DISP_RANGE; w++){
					//if(cost.at<SGM::VecDf>(y,x)[w] > 10000){printf("cost.at<SGM::VecDf>(%d,%d)[%d]: %f\n",y,x,w,cost.at<SGM::VecDf>(y,x)[w]);}
					if(cost.at<SGM::VecDf>(y,x)[w] == 0.0){cost.at<SGM::VecDf>(y,x)[w] = cost.at<SGM::VecDf>(y,x)[w-1];}
				}
			//}
		}
	}
*/
}

void SGMFlow::postProcess(cv::Mat &disparity){

	for(int x = 0; x < WIDTH; x++){
		for(int y = 0; y < HEIGHT; y++){
			if(disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){disparity.at<uchar>(y,x)=static_cast<uchar>(0);}
		}
	}

/*	for(int x = 0; x < WIDTH; x++){
		for(int y = 0; y < HEIGHT; y++){
			if(disFlag.at<uchar>(y,x) != static_cast<uchar>(DISFLAG)){
			if((short)disparity.at<uchar>(y,x) >= 39){
			
				std::cout<<"x: "<<x<<" y:"<<y<<std::endl;
				std::cout<<accumulatedCost.at<SGM::VecDf>(y,x)<<std::endl;
			}
		}
		}
	}
*/
//std::cout<<cost.at<SGM::VecDf>(320,1098)<<std::endl;
}

void SGMFlow::writeDerivative(){
	imwrite("../derivativeFlowLeft.jpg",derivativeFlowLeft);
	imwrite("../derivativeFlowLeftLast.jpg",derivativeFlowLeftLast);
}


void SGMFlow::copyDisflag(cv::Mat &M){

	disFlag.copyTo(M);

}

void SGMStereoFlow::computeCost(){	
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){

			for(int d_st = 0; d_st < DISP_RANGE; d_st++){ //use disp. range from stereo
				
				for(int neiY = y - winRadius ; neiY <= y + winRadius; neiY++){				
					for(int neiX = x - winRadius; neiX <= x + winRadius; neiX++){
						//apply image rotation
						float newx = neiX+imgRotation.at<Vec2f>(neiY,neiX)[0];
						float newy = neiY+imgRotation.at<Vec2f>(neiY,neiX)[1];
						float distx = newx - EpipoleLeft.at<float>(0);
						float disty = newy - EpipoleLeft.at<float>(1);
						
						//compute flow-d from stereo-d using alpha
						float L = sqrt(distx*distx + disty*disty);
						float w = d_st*(neiX*ransacAlpha[0]+neiY*ransacAlpha[1]+ransacAlpha[2]);
						float wMax = DISP_RANGE*(neiX*ransacAlpha[0]+neiY*ransacAlpha[1]+ransacAlpha[2]);
						float d_fl = L * ((float)Vmax*(float)w/wMax)/(1.0-((float)Vmax*(float)w/DISP_RANGE));
						
						//projected flow point
						int xx = round(newx + d_fl*translationLeft.at<Vec2f>(newy,newx)[0]);
						int yy = round(newy + d_fl*translationLeft.at<Vec2f>(newy,newx)[1]);
						
						cost.at<SGM::VecDf>(y,x)[d_st] += fabs(derivativeStereoLeft.at<float>(neiY, neiX)- 
										derivativeStereoRight.at<float>(neiY, neiX - d_st)) 
							+ CENSUS_W * computeHammingDist(censusImageLeft.at<uchar>(neiY, neiX), 
										censusImageRight.at<uchar>(neiY, neiX - d_st));

						if((xx>=winRadius) && (yy>=winRadius) && xx<(WIDTH-winRadius) && yy< (HEIGHT-winRadius)){	
							
							cost.at<SGM::VecDf>(y,x)[d_st] += fabs(derivativeFlowLeftLast.at<float>(neiY,neiX) - 
											derivativeFlowLeft.at<float>(yy,xx))
								+ (float)CENSUS_W * computeHammingDist(censusImageLeftLast.at<uchar>(neiY, neiX), 
											censusImageLeft.at<uchar>(yy, xx));
										
						}else{
							disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
						}
				
					}
				}
			}
		}
	}

		
	//Set flag for image boundaries
	for(int y = 0; y < winRadius; y++){
		for(int x = 0; x < WIDTH; x++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}
	for(int y = HEIGHT - 1; y > HEIGHT - 1 - winRadius; y--){
		for(int x = 0; x < WIDTH; x++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = 0; x < winRadius; x++){
		for(int y = 0; y < HEIGHT; y++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	for(int x = WIDTH - 1; x > WIDTH -1 - winRadius; x--){
		for(int y = 0; y < HEIGHT; y++){
			disFlag.at<uchar>(y,x)=static_cast<uchar>(DISFLAG);
		}
	}

	//Set non-full costs to zero
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			if(disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){	
				for(int w = 0; w < DISP_RANGE ; w++){
					cost.at<SGM::VecDf>(y,x)[w] = 0.0;
				}
			}
		}
	}

}

void SGMStereoFlow::postProcess(cv::Mat &disparity)
{
	//Set flagged disparities to zero
	for(int y = winRadius; y < HEIGHT - winRadius; y++){
		for(int x = winRadius; x < WIDTH - winRadius; x++){
			if(disFlag.at<uchar>(y,x) == static_cast<uchar>(DISFLAG)){	
				disparity.at<SGM::VecDf>(y,x) = 0.0;
			}
		}
	}
}
