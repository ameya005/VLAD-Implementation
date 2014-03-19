//BOWKMeans Recogniser

#include <iostream>
#include <cv.h>
#include <highgui.h>
#include "opencv2/ml/ml.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/legacy/legacy.hpp"
#include <string>
#include <fstream>
#include <vector>
//#include "HarrisDetector.cpp"

using namespace cv;
using namespace std;


int main()
{
	Mat vocabulary;
	
	FileStorage fs1("vocabulary_surf_vlad.xml", FileStorage::READ);
	cout<<"Reading Vocabulary from file";
	
	fs1["vocabulary"]>>vocabulary;
	fs1.release();
	Mat img;
	Ptr<FeatureDetector> detector = FeatureDetector::create("SURF");
	
	Ptr<DescriptorExtractor> extractor = DescriptorExtractor::create("SURF");
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce");
	BOWImgDescriptorExtractor bowide(extractor,matcher);
	bowide.setVocabulary(vocabulary);
	vector<KeyPoint> keypoints;
	Mat desc,desc1;
	//vector<Mat> desc;
	//Setting up training data
	char opt='y';
	int i=0;
	char compName[50],response[50];
	ifstream specNameList;
	specNameList.open("dir.txt");
	while(!specNameList.eof())
	{
		//cout<<"\n Enter Species for training";
		//cin>>compName;
		specNameList >> compName;
		strcpy(response,compName);
		
		strcat(response,"SURFVLADResponse.xml");
		cout<<"\n Adding to "<<response;
		chdir(compName);
		
		string imgName;
		ifstream f1;
		f1.open("list.txt");
		
		FileStorage fs2(response, FileStorage::WRITE);
		//int flag =0;
		Mat desc;
		//Mat responseCopy;
		while(!f1.eof())
		{
			//Mat responseHist(1,30,CV_32FC1);
			f1>>imgName;
			cout<<"\n"<<imgName;
			img=imread(imgName);
			if(!img.data)
				continue;
			cvtColor(img,img,CV_BGR2GRAY);
			SURF surf;
			vector<Point2f> points;
			//harrisFeatures(img, points);
			//cout<<"\n POints size:"<<points.size();
			/*
		
			for(i=0;i<points.size();i++)
			{
				KeyPoint temp(points[i],10,-1,0,0,-1);
				keypoints.push_back(temp);
				//cout<<"\n Point "<<i<<" "<< keypoints[i].pt.x <<" "<<keypoints[i].pt.y; 
			}
			*/	
            surf(img,img,keypoints,desc1,false);
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
			
			int i;
			cout<<"\n ResponseHist:"<<responseVector.rows<<","<<responseVector.cols<<","<<responseVector.type()<<","<<responseVector.channels();
			//cout<<"\n"<<desc.cols;
			
			if(responseHist.rows!=0)
			{ 
				
				desc.push_back(responseVector);
				
			}
			//cout<<"\n "<<desc.rows<<" "<<desc.cols;
			
		}
		
		fs2<<"responseHist"<<desc;
		
		fs2.release();
		
		f1.close();
		//f2.close();
		
		//cout<<"\nAnother comp?(y/n)";
		//cin>>opt;
		chdir("..");
		
	}
    return 0;
}
		
		
	
	
	

