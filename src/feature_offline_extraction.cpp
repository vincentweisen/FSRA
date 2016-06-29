/*
* Copyright (c) 2016 Vincent Ee -=- iceira.edu.tw>
*
* All rights reserved.
*
* $Id: feature_offline_extraction.cpp 2016-04-12 15:24:30Z  $
*
*/
/** \author Vincent Ee
*
* This algorithrm is prepare database use for ROS fsra package.

* First, we use SURF to extract multiple feature descriptor of the giving images.
* Next, given all the SURF descriptor, we train the dictionary by using BoW.
* Finally, all the descriptors extract to fix dimension feature by searching BoW dictionary.
* The extracted offline BoW feature would be use in fsra.cpp.

* Define : 

*         map_set : The number of map.
*         foldername : The folder path for map images.
*         frame_num : Iamges frame number.

* Input  : Images.
* Output : BoW dictionary, images BoW feature.

* Usage example : ./build/feature_offline_extraction
*/

#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#ifndef HAVE_OPENCV_NONFREE

int main(int, char**)
{
    printf("The sample requires nonfree module that is not available in your OpenCV distribution.\n");
    return -1;
}

#else

# include "opencv2/core/core.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/nonfree/features2d.hpp"

using namespace std;
using namespace cv;

int minHessian = 1000;

const int map_set = 2;
int frame_num[map_set] = {94, 84};
std::string foldername[map_set] = {"data/map_images_303-v1", "data/map_images_306-v3"};


int main( int argc, char** argv )
{

	string word;

	SurfFeatureDetector detector(minHessian,4,2,false);				// !! # Set extended = false (64 elements). if true (128 elements), BOWImgDescriptorExtractor.compute will be error!
																	// Since the FLAN can't match 128 elements
	SurfDescriptorExtractor extractor;
	Mat featuresUnclustered;

	for (int g=0; g<map_set; g++)
	{
		cout<<endl;
		cout<<"-----------  Set "<<g+1<<" Extraction  -----------"<<endl;
		cout<<endl;

		std::string surf_des_path = foldername[g]+"/surf_des.yml";
		FileStorage fs(surf_des_path, FileStorage::WRITE);
	
		for (int k=0; k<frame_num[g]; k++)
		{
			if (k<10)
				word="/frame000";
			else if (k<100)
				word="/frame00";
			else if (k<1000)
				word="/frame0";
			else
				word="/frame";

			std::ostringstream s;
			s<<k;

			std::string img_path = foldername[g]+word+s.str()+".jpg";
			Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE );

			//Detect the keypoints using SURF Detector

			std::vector<KeyPoint> surf_keypoints;
			detector.detect( img, surf_keypoints );
			cout<<img_path<<"    --    keypoints size : "<<surf_keypoints.size()<<endl;

			//Calculate descriptors (feature vectors)

			Mat surf_descriptors;
			extractor.compute( img, surf_keypoints, surf_descriptors );

			//put the all feature descriptors in a single Mat object (BoW use)
			featuresUnclustered.push_back(surf_descriptors);

			//Save Surf descriptor
			write(fs, "surf_des_"+s.str(), surf_descriptors);

			Mat img_surf_keypoints;
			drawKeypoints( img, surf_keypoints, img_surf_keypoints, Scalar::all(-1), DrawMatchesFlags::DEFAULT );
			imshow("surf_keypoints", img_surf_keypoints );
			waitKey(1);
		}

		cout<<endl;
		cout<<"-----------  Set "<<g+1<<" surf_des.yml Saved  -----------"<<endl;
		cout<<endl;

		fs.release();
	}


	//BOWKMeans Training

	cout<<endl;
	cout<<"----------- BoW dictionary training -----------"<<endl;
	cout<<endl;

	//the number of bags
	int dictionarySize = 200;
	//define Term Criteria
	TermCriteria tc(CV_TERMCRIT_ITER,100,0.001);
	//retries number
	int retries = 1;
	//necessary flags
	int flags = KMEANS_PP_CENTERS;
	//Create the BoW (or BoF) trainer
	BOWKMeansTrainer bowTrainer(dictionarySize,tc,retries,flags);
	//cluster the feature vectors
	Mat dictionary = bowTrainer.cluster(featuresUnclustered);

	//Extract Bow histogram

	//create a nearest neighbor matcher
	Ptr<DescriptorMatcher> matcher(new FlannBasedMatcher);
	//create SURF feature point extracter
	Ptr<FeatureDetector> ddetector(new SurfFeatureDetector(minHessian,4,2,false));
	//create SURF descriptor extractor
	Ptr<DescriptorExtractor> dextractor(new SurfDescriptorExtractor(minHessian,4,2,false));
	//create BoF (or BoW) descriptor extractor
	BOWImgDescriptorExtractor bowDE(dextractor,matcher);
	//Set the dictionary with the vocabulary we created in the first step
	bowDE.setVocabulary(dictionary);

	for (int g=0; g<map_set; g++)
	{
		cout<<endl;
		cout<<"-----------  Set "<<g+1<<" BoW histogram Extraction -----------"<<endl;
		cout<<endl;

		//store the vocabulary
		FileStorage fs1(foldername[g]+"/dictionary.yml", FileStorage::WRITE);
		fs1 << "vocabulary" << dictionary;
		fs1.release();

		//open the file to write the resultant descriptor
		FileStorage fs2(foldername[g]+"/bowsurf.yml", FileStorage::WRITE);	
	
		for (int k=0; k<frame_num[g]; k++)
		{
			if (k<10)
				word="/frame000";
			else if (k<100)
				word="/frame00";
			else if (k<1000)
				word="/frame0";
			else
				word="/frame";

			std::ostringstream s;
			s<<k;

			std::string img_path = foldername[g]+word+s.str()+".jpg";
			Mat img = imread(img_path, CV_LOAD_IMAGE_GRAYSCALE );

			//To store the keypoints that will be extracted by SURF
			vector<KeyPoint> keypoints;
			//Detect SURF keypoints (or feature points)
			ddetector->detect(img,keypoints);
			//To store the BoW (or BoF) representation of the image
			Mat bowDescriptor;
			//extract BoW (or BoF) descriptor from given image
			bowDE.compute(img,keypoints,bowDescriptor);

			cout<<img_path<<"    --    BoW size : "<<bowDescriptor.size()<<endl;
			//write the new BoF descriptor to the file
			write(fs2, "bow_surf_"+s.str(), bowDescriptor);
		}

		cout<<endl;
		cout<<"-----------  Set "<<g+1<<" bowsurf.yml Saved  -----------"<<endl;
		cout<<endl;

	fs2.release();
	}

	return 0;
}

#endif
