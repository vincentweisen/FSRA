/*
* Copyright (c) 2016 Vincent Ee -=- iceira.edu.tw>
*
* All rights reserved.
*
* $Id: fsraClass.cpp 2016-06-12 15:24:30Z  $
*
*/
/** \author Vincent Ee
*

* The FSRA class is an algorithm that can estimate sensor 6-DoF pose to locate in point-cloud map.
* We use scene recognition and ICP technique to solve the sensor pose.

* Define : 

*         map_set : The number of map.
*         foldername : The folder path for map images.
*         frame_num : Iamges frame number.

* Input  : Images, RGBD-Map, image-trajectory, BoW dictionary, images BoW feature.
* Output : Eigen::Affine3f pose, pcl::PointCloud<PointType>::Ptr map_cloud.

* Usage example : FSRA t(query_img, query_cloud);
*/


// C++ specific includes
#include <iostream>
#include <ctime>
#include <time.h>

// OpenCV specific includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/nonfree/features2d.hpp>

// PCL specific includes
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/ros/conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/console/parse.h>
#include <pcl/filters/crop_box.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/fpfh.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/common/transforms.h>


const int map_set = 2;
int frame_num[map_set] = {94, 84};
std::string foldername[map_set] = {"/home/vincentee/code/ros_hydro/src/fsra/map_images_303-v1", "/home/vincentee/code/ros_hydro/src/fsra/map_images_306-v3"};
int pre_group_ = 3;
int cmod_ = 0;

int minHessian = 1000;
cv::Mat dictionary;
cv::Ptr<cv::DescriptorMatcher> matcher(new cv::FlannBasedMatcher);
cv::Ptr<cv::FeatureDetector> detector(new cv::SurfFeatureDetector(minHessian,4,2,false));
cv::Ptr<cv::DescriptorExtractor> extractor(new cv::SurfDescriptorExtractor(minHessian,4,2,false));
cv::BOWImgDescriptorExtractor bowDE(extractor,matcher);

float voxel_leaf_ = 0.04;
float voxel_rad_ = 0.08;
float voxel_nei_ = 20;
typedef pcl::PointXYZRGB PointType;
typedef pcl::FPFHSignature33 CloudDesType;

Eigen::Affine3f pose;
pcl::PointCloud<PointType>::Ptr map_cloud(new pcl::PointCloud<PointType>);
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("Map Visualizer"));


void keyboardEventOccurred(const pcl::visualization::KeyboardEvent &event, void* viewer_void)
{
    if(event.keyDown ())
    {
		char key_ = event.getKeySym().c_str()[0];
		switch(key_)
		{
	    	case 'p':
			cmod_ = 1;
			break;
	    	case 'i':
			cmod_ = 0;
			break;
		}
	}
}

void downsampleFilter(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud)
{
	// Filter object.
	pcl::VoxelGrid<PointType> filter;
	filter.setInputCloud(in_cloud);
	// We set the size of every voxel to be 1x1x1cm
	// (only one point per every cubic centimeter will survive).
	filter.setLeafSize(voxel_leaf_, voxel_leaf_, voxel_leaf_);
 
	filter.filter(*in_cloud);

	// Remove outlier.
	pcl::RadiusOutlierRemoval<PointType> outlier;
	outlier.setInputCloud(in_cloud);
	// Every point must have 10 neighbors within 15cm, or it will be removed.
	outlier.setRadiusSearch(voxel_rad_);
	outlier.setMinNeighborsInRadius(voxel_nei_);

	outlier.filter(*out_cloud);
}

class FSRA
{
public:
	FSRA(cv::Mat img);
	FSRA(cv::Mat img, pcl::PointCloud<PointType>::Ptr cloud);
	Eigen::Affine3f estimateCoarsePose();
	Eigen::Affine3f estimateExactPose(Eigen::Affine3f coarsePose);
private:
	cv::Mat c_img;
	pcl::PointCloud<PointType>::Ptr c_cloud;
	int c_group; int c_retri_index;
	void distanceCalculate(float* distance, cv::Mat query_bowDescriptor, int dir_size_);
	int minIndex(float* array, int m);
	void retrieveIndexGroup(int &group, int &retri_index, int min_index);
	void showRetrieveImage();
	void readTrajectory(std::vector<long double> &timestamp, std::vector<Eigen::Affine3f> &map_trajectory);
	int searchTimestamp(std::vector<long double> timestamp);
	void visualizer();
	void sceneCloudPreprocess(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud);
	void mapCropping(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud, Eigen::Affine3f P);
	void normalEstimation(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, float rad);
	void fpfhEstimation(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, pcl::PointCloud<CloudDesType>::Ptr descriptors, float rad);
	void corrRejector(pcl::PointCloud<PointType>::Ptr cloud_1, pcl::PointCloud<PointType>::Ptr cloud_2, pcl::CorrespondencesPtr correspondences, pcl::CorrespondencesPtr corr_filtered);
	void globalICP( pcl::PointCloud<PointType>::Ptr cloud_1, pcl::PointCloud<PointType>::Ptr cloud_2, pcl::PointCloud<PointType>::Ptr cloud_out, Eigen::Matrix4f guess_transform, Eigen::Matrix4f &icp_transform);
};

FSRA::FSRA(cv::Mat img)
{
	c_img = img;
	pose = estimateCoarsePose();
}

FSRA::FSRA(cv::Mat img, pcl::PointCloud<PointType>::Ptr cloud)
{
	c_img = img;
	c_cloud = cloud;
	c_group = pre_group_;

	if (cmod_ != 2)
		pose = estimateCoarsePose();
	if (cmod_ == 1)
		pose = estimateExactPose(pose);

	visualizer();
	pre_group_ = c_group;
}

Eigen::Affine3f FSRA::estimateCoarsePose()
{
	ROS_INFO("---------- Coarse Pose Estimation ----------");

	cv::vector<cv::KeyPoint> keypoints;
	detector->detect(c_img,keypoints);
	cv::Mat query_bowDescriptor;
	bowDE.compute(c_img,keypoints,query_bowDescriptor);

	int dir_size_ = query_bowDescriptor.cols;

	int total_frame_num = 0;
	for (int i=0; i<map_set; i++)
		total_frame_num += frame_num[i];

	float distance[total_frame_num];
	distanceCalculate(distance, query_bowDescriptor, dir_size_);
	int min_index = minIndex(distance, total_frame_num);

	retrieveIndexGroup(c_group, c_retri_index, min_index);
	cout<<"Retrieved (group, index) : ("<<c_group<<", "<<c_retri_index<<")"<<endl;

	showRetrieveImage();

	std::vector<long double> timestamp;
	std::vector<Eigen::Affine3f> map_trajectory;
	readTrajectory(timestamp, map_trajectory);

	int coarse_pose_index = searchTimestamp(timestamp);

	return map_trajectory[coarse_pose_index];
}

Eigen::Affine3f FSRA::estimateExactPose(Eigen::Affine3f coarsePose)
{
	ROS_INFO("---------- Exact Pose Estimation ----------");

	cout<<"Reading cloud ..."<<endl;

	// Transform to the same axis with rgbdslam-map
	pcl::PointCloud<PointType>::Ptr scene_cloud(new pcl::PointCloud<PointType>);
	sceneCloudPreprocess(c_cloud, scene_cloud);
	//pcl::io::loadPCDFile<PointType> ("/home/vincentee/code/asus_nav/data/map_images_303-v1/scene/scene_4p.pcd", *scene_cloud);

	// Map cropping
	cout<<"Map cropping ..."<<endl;
	pcl::PointCloud<PointType>::Ptr crop_cloud(new pcl::PointCloud<PointType>);
	mapCropping(map_cloud, crop_cloud, coarsePose);
/*
	// Normal estimation
	cout<<"Normal estimation ..."<<endl;
	pcl::PointCloud<pcl::Normal>::Ptr scene_normals(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>::Ptr crop_normals(new pcl::PointCloud<pcl::Normal>);
	normalEstimation(scene_cloud, scene_normals, 0.5);
	normalEstimation(crop_cloud, crop_normals, 0.5);

	// FPFH estimation
	cout<<"FPFH estimation ..."<<endl;
	pcl::PointCloud<CloudDesType>::Ptr scene_fpfh_descriptors(new pcl::PointCloud<CloudDesType>());
	pcl::PointCloud<CloudDesType>::Ptr crop_fpfh_descriptors(new pcl::PointCloud<CloudDesType>());
	fpfhEstimation(scene_cloud, scene_normals, scene_fpfh_descriptors, 0.4);
	fpfhEstimation(crop_cloud, crop_normals, crop_fpfh_descriptors, 0.4);

	// Correspondences estimation
	cout<<"Correspondences estimation ..."<<endl;
	pcl::CorrespondencesPtr correspondences (new pcl::Correspondences);
	pcl::registration::CorrespondenceEstimation<CloudDesType, CloudDesType> cest;
	cest.setInputSource (scene_fpfh_descriptors);
	cest.setInputTarget (crop_fpfh_descriptors);
	cest.determineCorrespondences (*correspondences);

	// Correspondences rejection
	cout<<"Correspondences rejection ..."<<endl;
	pcl::CorrespondencesPtr corr_filtered (new pcl::Correspondences);
	corrRejector(scene_cloud, crop_cloud, correspondences, corr_filtered);

	// Align transform estimation
	cout<<"Align transform estimation ..."<<endl;
	Eigen::Matrix4f corr_transform;
	pcl::registration::TransformationEstimationSVD<PointType, PointType> trans_est;
	trans_est.estimateRigidTransformation (*scene_cloud, *crop_cloud, *corr_filtered, corr_transform);
*/
	// Global map icp estimation
	cout<<"Global map icp estimation ..."<<endl;
	Eigen::Matrix4f icp_transform;
	pcl::PointCloud<PointType>::Ptr transform_cloud(new pcl::PointCloud<PointType>);
	globalICP(scene_cloud, map_cloud, transform_cloud, coarsePose.matrix(), icp_transform);

	Eigen::Affine3f exactPose;
	exactPose.matrix() = icp_transform;

	// Alignment visualization
	cout<<"Alignment visualization ..."<<endl;
//	viewer->removePointCloud ("global_map");
	viewer->removeCoordinateSystem ();
//	pcl::visualization::PointCloudColorHandlerRGBField<PointType> cropped_color_handler (crop_cloud);
//	viewer->addPointCloud (crop_cloud, cropped_color_handler, "cropped");
//	pcl::visualization::PointCloudColorHandlerRGBField<PointType> scene_color_handler (scene_cloud);
//	viewer->addPointCloud (scene_cloud, scene_color_handler, "scene");
	pcl::visualization::PointCloudColorHandlerCustom<PointType> transform_color_handler (transform_cloud, 0, 255, 0);
	viewer->addPointCloud (transform_cloud, transform_color_handler, "transform");
//	viewer->addCoordinateSystem (0.8, coarsePose);
	viewer->addCoordinateSystem (0.8, exactPose);

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce ();
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}

	cmod_ = 2;
//	viewer->removePointCloud ("cropped");
//	viewer->removePointCloud ("scene");
	viewer->removePointCloud ("transform");

	return exactPose;
}

void FSRA::distanceCalculate(float* distance, cv::Mat query_bowDescriptor, int dir_size_)
{
	int cas = 0;

	for (int g=0; g<map_set; g++)
	{
		std::string des_path = foldername[g]+"/bowsurf.yml";
		cv::FileStorage fs(des_path, cv::FileStorage::READ);

		for (int k=0; k<frame_num[g]; k++)
		{
			std::ostringstream s;
			s<<k;

			cv::Mat bowDescriptor;
			fs["bow_surf_"+s.str()] >> bowDescriptor;

			float d;
			distance[cas+k] = 0;

			for (int i=0; i<dir_size_; i++)
			{
				d = query_bowDescriptor.at<float>(0, i) - bowDescriptor.at<float>(0, i);
				distance[cas+k] += pow(d,2.0);
			}
		}
		cas += frame_num[g];
		fs.release();
	}
}

int FSRA::minIndex(float* array, int m)
{
	int length = m;
	int nor_bit = 1000000;

    int min = 0;
	int p = array[0]*nor_bit;

     for(int i=1; i<length; i++)
     {
		int a = array[i]*nor_bit;
        if(a < p)
		{
			p = array[i]*nor_bit;
			min = i;
		}
     }

	//std::cout<<"Index = "<<min<<std::endl;
	return min;
}

void FSRA::retrieveIndexGroup(int &group, int &retri_index, int min_index)
{
	int sum = 0;

	for (int g=0; g<map_set; g++)
	{
		sum += frame_num[g];
		int check = min_index - sum;
		if (check < 0)
		{
			c_retri_index = check + frame_num[g];
			c_group = g;
			break;
		}
	}
}

void FSRA::showRetrieveImage()
{
	std::string word;

	if (c_retri_index<10)
			word="/frame000";
		else if (c_retri_index<100)
			word="/frame00";
		else if (c_retri_index<1000)
			word="/frame0";
		else
			word="/frame";

	std::ostringstream s;
	s<<c_retri_index;
	std::string retrived_path = foldername[c_group]+word+s.str()+".jpg";
	cv::Mat retrieved_img = cv::imread(retrived_path, CV_LOAD_IMAGE_GRAYSCALE );

	cv::Size sz = c_img.size();
	cv::Mat im(sz.height+sz.height, sz.width, CV_8UC1);
	cv::Mat left(im, cv::Rect(0, 0, sz.width, sz.height));
	c_img.copyTo(left);
	cv::Mat right(im, cv::Rect(0, sz.height, sz.width, sz.height));
	retrieved_img.copyTo(right);
	imshow("Query - Retrieve", im);

	cv::waitKey(1);
}

void FSRA::readTrajectory(std::vector<long double> &timestamp, std::vector<Eigen::Affine3f> &map_trajectory)
{
	std::string trajectory_path = foldername[c_group]+"/trajectory_estimate.txt";
	std::ifstream file(trajectory_path.c_str());

	if(file.is_open())
	{
		std::string line;
		getline(file, line);
		while(!file.eof())
		{
			long double time;
			Eigen::Vector3f vec3;
			Eigen::Vector4f vec4;
			Eigen::Affine3f aff;

			file >> time;
			timestamp.push_back(time);

			for(int i = 0; i < 3; ++i)
			{
				file >> vec3(i);
			}
			aff.translation() = vec3;

			for(int i = 0; i < 4; ++i)
			{
				file >> vec4(i);
			}
			Eigen::Matrix3f mat3 = Eigen::Quaternionf(vec4(3), vec4(0), vec4(1), vec4(2)).toRotationMatrix();
			Eigen::Matrix4f mat4 = Eigen::Matrix4f::Identity();
			mat4.block(0,0,3,3) = mat3;
			mat4(0,3) = vec3(0);
			mat4(1,3) = vec3(1);
			mat4(2,3) = vec3(2);
			aff.matrix() = mat4;
			map_trajectory.push_back(aff);
		}
	}
	else
	{
		cout<<trajectory_path<<" --> reading fail !!!"<<endl;
		ros::shutdown();
	}
	file.close();
}

int FSRA::searchTimestamp(std::vector<long double> timestamp)
{
	int nor_bit = 1000000;

	double diff = timestamp[timestamp.size()-1] - timestamp[0];
	int d = int(diff*nor_bit);
	int p = int(d*(double(c_retri_index)/double(frame_num[c_group])));
	long int check = (long int)(p) + (long int)(timestamp[0]*nor_bit);
	int coarse_pose_index = 0;

	for (int i=1; i<timestamp.size(); i++)
	{
		long int a = (long int)(timestamp[i-1]*nor_bit);
		long int b = (long int)(timestamp[i]*nor_bit);

		if((check>a) && (check<=b))
		{
			coarse_pose_index = i;
			break;
		}
	}

	//cout<<"coarse_pose_index : "<<coarse_pose_index<<endl;
	return coarse_pose_index;
}

void FSRA::visualizer()
{
	if ((pre_group_ != c_group) || (pre_group_ == 3))
	{
		if (pre_group_ != 3)
		{
			cout<<"Point-cloud map reload ..."<<endl;
			viewer->removePointCloud ("global_map");
			viewer->removeCoordinateSystem ();
		}

		pcl::io::loadPCDFile<PointType> (foldername[c_group]+"/map.pcd", *map_cloud);
		pcl::visualization::PointCloudColorHandlerRGBField<PointType> color_handler (map_cloud);
		viewer->addPointCloud (map_cloud, color_handler, "global_map");
		viewer->addCoordinateSystem (1.2);
		viewer->addCoordinateSystem (0.8, pose);
		viewer->registerKeyboardCallback(keyboardEventOccurred, (void*)&viewer);
	}
	else
	{
		cout<<"Coarse pose update ..."<<endl;
		viewer->removeCoordinateSystem ();
		viewer->addCoordinateSystem (0.8, pose);
	}

	viewer->spinOnce();
	boost::this_thread::sleep (boost::posix_time::microseconds (100000));
}


void FSRA::mapCropping(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud, Eigen::Affine3f P)
{
	pcl::CropBox<PointType> cropBoxFilter; 
	Eigen::Vector4f minPoint;
	minPoint[0]= 0.5;  // define minimum point x
	minPoint[1]=-3.0;  // define minimum point y
	minPoint[2]=-10.0;  // define minimum point z
	Eigen::Vector4f maxPoint;
	maxPoint[0]=5.0;  // define max point x
	maxPoint[1]=3.0;  // define max point y
	maxPoint[2]=10.0;  // define max point z

	Eigen::Matrix4f M = P.matrix();
	Eigen::Vector3f crop_rot;
	crop_rot[0] = atan2( M(2,1) , M(2,2) );													//Roll
	crop_rot[1] = atan2( -M(2,0) , ( sqrt( pow(M(2,1),2.0) + pow(M(2,2),2.0) ) ) );			//Pitch
	crop_rot[2] = atan2( M(1,0) , M(0,0) );													//Yaw

	cropBoxFilter.setInputCloud (in_cloud);
	cropBoxFilter.setMin(minPoint);
	cropBoxFilter.setMax(maxPoint);
//	cropBoxFilter.setTransform(coarsePose);					// This class function have some bug. Thus, use setRotation()
	cropBoxFilter.setRotation(crop_rot);
	cropBoxFilter.setTranslation(P.translation());
	cropBoxFilter.filter(*out_cloud);
}

void FSRA::sceneCloudPreprocess(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud)
{
	float theta = M_PI/2;
	Eigen::Affine3f transform = Eigen::Affine3f::Identity();
	transform.rotate (Eigen::AngleAxisf (-theta, Eigen::Vector3f::UnitX()));
	transform.rotate (Eigen::AngleAxisf (theta, Eigen::Vector3f::UnitY()));

	pcl::transformPointCloud (*in_cloud, *out_cloud, transform);

	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*out_cloud, *out_cloud, indices);
//	downsampleFilter(out_cloud, out_cloud);

	pcl::CropBox<PointType> cropBoxFilter; 
	Eigen::Vector4f minPoint;
	minPoint[0]= 0.5;  // define minimum point x
	minPoint[1]=-3.0;  // define minimum point y
	minPoint[2]=-10.0;  // define minimum point z
	Eigen::Vector4f maxPoint;
	maxPoint[0]=5.0;  // define max point x
	maxPoint[1]=3.0;  // define max point y
	maxPoint[2]=10.0;  // define max point z
	cropBoxFilter.setInputCloud (out_cloud);
	cropBoxFilter.setMin(minPoint);
	cropBoxFilter.setMax(maxPoint);
	cropBoxFilter.filter(*out_cloud);
}
void FSRA::normalEstimation(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<pcl::Normal>::Ptr normals, float rad)
{
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
	pcl::NormalEstimation<PointType, pcl::Normal> normalEstimation;

	normalEstimation.setInputCloud(in_cloud);
	normalEstimation.setRadiusSearch(rad);
	normalEstimation.setSearchMethod(kdtree);
	normalEstimation.compute(*normals);
}

void FSRA::fpfhEstimation(pcl::PointCloud<PointType>::Ptr in_cloud, 
						pcl::PointCloud<pcl::Normal>::Ptr normals, 
						pcl::PointCloud<CloudDesType>::Ptr descriptors,
						float rad)
{
	pcl::FPFHEstimation<PointType, pcl::Normal, CloudDesType> fpfh;
	pcl::search::KdTree<PointType>::Ptr kdtree(new pcl::search::KdTree<PointType>);
	fpfh.setInputCloud(in_cloud);
	fpfh.setInputNormals(normals);
	fpfh.setSearchMethod(kdtree);
	fpfh.setRadiusSearch(rad);
	fpfh.compute(*descriptors);
}

void FSRA::corrRejector(pcl::PointCloud<PointType>::Ptr cloud_1, 
						pcl::PointCloud<PointType>::Ptr cloud_2, 
						pcl::CorrespondencesPtr correspondences, 
						pcl::CorrespondencesPtr corr_filtered)
{
	pcl::registration::CorrespondenceRejectorSampleConsensus<PointType> rejector;
	rejector.setInputSource (cloud_1);
	rejector.setInputTarget (cloud_2);
	rejector.setInlierThreshold (0.5);				//0.5
	rejector.setMaximumIterations (1000);			//1000
	rejector.setRefineModel (false);
	rejector.setInputCorrespondences (correspondences);;
	rejector.getCorrespondences (*corr_filtered);
	std::cout<<"Number correspondences : "<<corr_filtered->size()<<endl;
}

void FSRA::globalICP( pcl::PointCloud<PointType>::Ptr cloud_1, 
					pcl::PointCloud<PointType>::Ptr cloud_2, 
					pcl::PointCloud<PointType>::Ptr cloud_out, 
					Eigen::Matrix4f guess_transform, 
					Eigen::Matrix4f &icp_transform)
{
	pcl::IterativeClosestPoint<PointType, PointType> icp;
	icp.setInputSource(cloud_1);
	icp.setInputTarget(cloud_2);
	icp.setMaximumIterations (100);				//100
	icp.setTransformationEpsilon (1e-5);			//1e-4
	icp.setEuclideanFitnessEpsilon (1e-5);			//1e-4
	icp.setMaxCorrespondenceDistance(0.2);			//0.2
	icp.align(*cloud_out, guess_transform);
	icp_transform = icp.getFinalTransformation ();
	std::cout<<"FitnessScore : "<<icp.getFitnessScore()<<std::endl;
	std::cout<<icp_transform<<std::endl;
}
