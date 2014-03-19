/****************Program to train a SVM classifier fusing bag-of -words histograms*********/
/** Create a folder with all the leaf responses for SIFT and shapecontext naming them as Leaf_no.xml  and Leaf_no_shape.xml
 * For e.g. : Leaf1.xml -> SIFT responses
 * 			  Leaf1_shape.xml - > Shape responses
 * Then create a text file with the names of all Leaf response(SIFT) files
 *******************************************************************************************/ 			

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <string>
#include <fstream>
#define LABELS 32
using namespace cv;
using namespace cvflann;
using namespace std;

int main()
{
	
	/* code for reading file with ORB features*/

	Mat trainData,labels;
	chdir("SURFVLAD");
	
	string fName;
	char fName_Shape[20];
	ifstream f1("list.txt");
	//ifstream f2("response_shape.txt");
	int count = 0, rowCount,i,j;
	while(count!=5)		//Pushes the training data into a matrix
	{
		f1 >> fName;
		cout<<"\n Name:"<<fName;
		
		//chdir("Leaf1");
		FileStorage fs1(fName,FileStorage::READ);
		//FileStorage fs1(fName_Shape,FileStorage::READ);
		
		Mat response,response_shape;
		fs1["responseHist"]>>response;
		//fs2["responseHist"]>>response_shape;

		rowCount=response.rows;
		
		//~ Mat responseMerged(response.rows, 2000,CV_32F);
		//~ for(i=0;i<response.rows;i++);
		//~ {
			//~ for(j=0;j<1000;j++)
			//~ {
				//~ responseMerged.at<double>(i,j)=response.at<double>(i,j);
				//~ responseMerged.at<double>(i,j+1000) = response_shape.at<double>(i,j);
			//~ }
		//~ }
		trainData.push_back(response);
		Mat reslabels=Mat::zeros(rowCount,1,CV_32SC1);
		reslabels.setTo((float)(count+1));
		labels.push_back(reslabels);
		cout<<"\n Response rows:"<<rowCount;
		cout<<"trainData:"<<trainData.rows<<","<<trainData.cols<<"\n Label:"<<labels.rows;
		fs1.release();
		count++;
		cout<<"\n COUNT:"<<count;
	}
	f1.close();
	
	
	cout<<"\n LAbels:\n"<<labels;
	

	
	
	
	/*
	CvSVMParams params(CvSVM::NU_SVC, CvSVM::POLY, 1, 1, 3,
	params.svm_type = CvSVM::NU_SVC;
	params.kernel_type = CvSVM::POLY;
	params.gamma = 1;
	params.coef0 = 1;
	params.degree = 3; 
	params.term_crit =  cvTermCriteria(CV_TERMCRIT_ITER, 100, 1e-6);
	*/
	
	CvSVMParams params( 
	CvSVM::C_SVC, CvSVM::RBF, 1,
	1, 0, 1, 0.1, 0.2, 0, cvTermCriteria( CV_TERMCRIT_ITER+CV_TERMCRIT_EPS, 1000, FLT_EPSILON ));
	//Train array:DD array with responsehistograms values
	
	Mat svmData;
	trainData.convertTo(svmData,CV_32F);
	Mat labelData;
	labelData=labels.t(); //labels: Array of labels
	
	cout<<"\n Training!";
	
	//SVM training:
	
	CvSVM svm(svmData, labelData, Mat(), Mat(), params);
	//params=svm.get_params();
	//cout<<"\n Svm parameters: "<<svm.get_params();
	CvParamGrid Cgrid=CvSVM::get_default_grid(CvSVM::C);
	CvParamGrid gammaGrid=CvSVM::get_default_grid(CvSVM::GAMMA);
	CvParamGrid pGrid=CvSVM::get_default_grid(CvSVM::P);
	CvParamGrid nuGrid=CvSVM::get_default_grid(CvSVM::NU);
	CvParamGrid coeffGrid=CvSVM::get_default_grid(CvSVM::COEF);
	CvParamGrid degreeGrid=CvSVM::get_default_grid(CvSVM::DEGREE);
	
	svm.train_auto(svmData, labelData, Mat(), Mat(), params,10, Cgrid, gammaGrid,  pGrid, nuGrid,  coeffGrid, degreeGrid, false);
	svm.save("classifier_SURF_VLAD_C200_RBF.xml","SURF");
	
	//End of training

	
	return 0;
	
}
	
	
