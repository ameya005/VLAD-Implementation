//SVM test

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <string>
#include <fstream>
//#include "HarrisDetector.cpp"
//#include "highDensity.cpp"

using namespace cv;
using namespace std;

int main()
{

//Test
	CvSVM svm;
	svm.load("classifier_SURF_VLADNU_LINEAR.xml","SURF");
	//Mat testImg = imread("Test");
	
	Mat vocabulary;
	
	FileStorage fs3("vocabulary_surf_vlad.xml", FileStorage::READ);
	cout<<"Reading Vocabulary from file";
	
	fs3["vocabulary"]>>vocabulary;
	fs3.release();
	chdir("Images");
	ifstream f1;
	f1.open("list.txt");
	ofstream f2("output_SURFVLAD_Linear.txt");
    Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	
    Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
    BOWImgDescriptorExtractor bowide(extractor,matcher);
    bowide.setVocabulary(vocabulary);
	while(!f1.eof())
	{
		string imgName;
		f1>>imgName;
		cout<<"\n"<<imgName;
		Mat img=imread(imgName);
		if(!img.data)
			continue;
		
		vector<KeyPoint> keypoints;
		Mat desc1(0,30,CV_64FC1);
		//Mat responseHist;
        //Mat responseHist(1, 1000, CV_32FC1, Scalar::all(0.0));
		cvtColor(img,img,CV_BGR2GRAY);
		SURF surf;
		vector<Point2f> points;
		//harrisFeatures(img, points);
		//cout<<"\n POints size:"<<points.size();
		int i;
		/*
		for(i=0;i<points.size();i++)
		{
				KeyPoint temp(points[i],10,-1,0,0,-1);
				keypoints.push_back(temp);
				//cout<<"\n Point "<<i<<" "<< keypoints[i].pt.x <<" "<<keypoints[i].pt.y; 
		}
        */
		surf(img,img,keypoints,desc1,false);
		//bowide.compute(img,keypoints,responseHist);
		Mat desc1_32f;
        desc1.convertTo(desc1_32f,CV_32F);
         /* **********The new VLAD.compute  **********/ 
        
        vector<DMatch> matches;
        matcher->match(desc1_32f, matches); //desc1 contains descriptors for each image
        
        
        Mat responseHist(vocabulary.rows, desc1_32f.cols, CV_32FC1, Scalar::all(0.0));
       // float *dptr = (float*)responseHist.data;
        for( size_t i = 0; i < matches.size(); i++ )
        {
            int queryIdx = matches[i].queryIdx;
            int trainIdx = matches[i].trainIdx; // cluster index
            CV_Assert( queryIdx == (int)i );
            Mat residual;
            
            subtract(desc1_32f.row(matches[i].queryIdx),vocabulary.row(matches[i].trainIdx),residual,noArray(), CV_32F);
            add(responseHist.row(matches[i].trainIdx),residual,responseHist.row(matches[i].trainIdx),noArray(),responseHist.type());
            
            
            
        }
        responseHist /= norm(responseHist,NORM_L2, noArray());
        
        Mat responseVector(1, vocabulary.rows*desc1_32f.cols, CV_32FC1, Scalar::all(0.0));
        
        responseVector = responseHist.reshape(0,1);
        
        
		float predLabel=4;
		if(responseVector.rows!=0)
		{
			predLabel = svm.predict(responseVector, false);
		}
		cout<<"\n value:"<<predLabel;
		f2 << imgName << " " << predLabel << "\n"; 
	}	
	
	f2.close();
	return 0;
}	
