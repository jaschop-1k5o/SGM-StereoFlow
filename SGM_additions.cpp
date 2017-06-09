//TO DEFINE: alpha, DISP_RANGE_FLOW, disFlag_flow

void SGMStereoFlow::computeCost(){	
//Scheme 2
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
						float w = d_st*(neix*alpha[0]+neiy*alpha[1]+alpha[2]);
						float d_fl = L * ((float)Vmax*(float)w/DISP_RANGE_FLOW)/(1.0-((float)Vmax*(float)w/DISP_RANGE));
						
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

