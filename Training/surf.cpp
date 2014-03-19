//
#include <cv.h>
#include <highgui.h>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include <string>
#include <fstream>
//#include "HarrisDetector.cpp"

using namespace std;
using namespace cv;

int main()
{
	Mat img;
	Mat threshImg;
    
	//adaptiveThreshold(img, threshImg);
	ifstream dir_list;
	dir_list.open("dir.txt");
	if(!dir_list.is_open())
	{
		cout<<"file list doesnot exist";
		exit(0);
	}
	ifstream f1;
	
	
	char dirName[20];
	string imgName;
	while(!dir_list.eof())
	{
		Mat desc_all;
		dir_list>>dirName;
		cout<<"\n Creating feature vectors for "<<dirName;
		chdir(dirName);
		f1.open("list.txt");
		if(f1.is_open()==0)
		{
			cout<<"\n Input File does not exist for" << dirName<<"!";
			exit(0);
		}
		while(!f1.eof())
		{
			vector<KeyPoint> keypoints;
		    Mat desc;
            f1>>imgName;
			cout<<"\n"<<imgName;
			img=imread(imgName);
			if(!img.data)
			{	
				cout<<"\n File not found!";		
				continue;
			}
			cvtColor(img,img,CV_BGR2GRAY);
			SURF surf;
			
			vector<Point2f> points;
			//harrisFeatures(img, points);
			cout<<"\n POints size:"<<points.size();
			int i;
			/*
			for(i=0;i<points.size();i++)
			{
				KeyPoint temp(points[i],10,-1,0,0,-1);
				keypoints.push_back(temp);
				cout<<"\n Point "<<i<<" "<< keypoints[i].pt.x <<" "<<keypoints[i].pt.y; 
			}
			*/	
			surf(img,img,keypoints,desc,false);
			desc_all.push_back(desc);

          
	
		}
		string featureVectorFile = string(dirName) + string("SURF.xml");
		cout<<"\n Writing Feature vectors into: " << featureVectorFile;
		chdir("../SIFT"); 
		FileStorage fs(featureVectorFile,FileStorage::WRITE);

		fs<<"desc_all"<<desc_all;
    	
		fs.release();	
		f1.close();
		chdir("..");
	}
	dir_list.close();
	
	return 1;
}
