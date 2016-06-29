/*
 * * Copyright (c) 2015 Vincent Ee -=- iceira.edu.tw>
 * *
 * * All rights reserved.
 * *
 * * $Id: preprocess.cpp 2015-12-02 21:24:30Z  $
 * *
 * */
/** \author Vincent Ee
 * *
 * * This algorithrm is for point cloud downsampling, remove ceiling and floor, also visualize before using map_retrieval.cpp (3D indoor fast localization).
 * * Define : 
 * *         argc=2 : Only visualize.
 * *         argc=3 : downsampling and visualize.
* *          height_thress_min : The minimum hieght of map cropping.
* *          height_thress_max : The maximum hieght of map cropping.
 *
 * * Input  : RGBDSLAM point cloud map.
 * * Output : Downsampled point cloud map.
 *
 * * Usage : ./build/preprocess data/rgbdmap1.pcd
 * *		 ./build/preprocess data/rgbdmap1.pcd data/global_map1.pcd
 * */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/common/projection_matrix.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/radius_outlier_removal.h>
#include <pcl/filters/crop_box.h>

using namespace std;

float height_thress_min = -0.55;
float height_thress_max = 3.0;
typedef pcl::PointXYZRGB PointType;

boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis (pcl::PointCloud<PointType>::ConstPtr cloud)
{
  	// --------------------------------------------
  	// -----Open 3D viewer and add point cloud-----
  	// --------------------------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
	viewer->setBackgroundColor (0, 0, 0);
	pcl::visualization::PointCloudColorHandlerRGBField<PointType> rgb(cloud);
	viewer->addPointCloud<PointType> (cloud, rgb, "scene");
	viewer->addCoordinateSystem (0.2,0,0,0);

	return (viewer);
}

void downsample(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud)
{
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*in_cloud,*in_cloud, indices);
	
	// Filter object.
	pcl::VoxelGrid<PointType> filter;
	filter.setInputCloud(in_cloud);
	// We set the size of every voxel to be 1x1x1cm
	// (only one point per every cubic centimeter will survive).
	filter.setLeafSize(0.04f, 0.04f, 0.04f);		//0.04f
 
	filter.filter(*in_cloud);
	
	// Remove outlier.
	pcl::RadiusOutlierRemoval<PointType> outlier;
	outlier.setInputCloud(in_cloud);
	// Every point must have 10 neighbors within 15cm, or it will be removed.
	outlier.setRadiusSearch(0.1);				//0.08
	outlier.setMinNeighborsInRadius(20);		//20

	outlier.filter(*out_cloud);
	
	out_cloud = in_cloud;
}

void PassThroughFilter(pcl::PointCloud<PointType>::Ptr in_cloud, pcl::PointCloud<PointType>::Ptr out_cloud)
{
	pcl::CropBox<PointType> cropBoxFilter;

	Eigen::Vector4f minPoint;
	minPoint[0] = 0.5;  // define minimum point x
	minPoint[1] = -3.0;  // define minimum point y
	minPoint[2] = -10.0;  // define minimum point z

	Eigen::Vector4f maxPoint;
	maxPoint[0] = 5.0;  // define max point x
	maxPoint[1] = 3.0;  // define max point y
	maxPoint[2] = 10.0;  // define max point z

	cropBoxFilter.setInputCloud (in_cloud);
	cropBoxFilter.setMin(minPoint);
	cropBoxFilter.setMax(maxPoint);
	cropBoxFilter.filter(*out_cloud);
}

int main(int argc, char **argv)
{
	//Read the input model
	pcl::PointCloud<PointType>::Ptr model_read(new pcl::PointCloud<PointType>);

	if (pcl::io::loadPCDFile<PointType>(argv[1], *model_read) != 0)
	{
		return -1;
	}

	cout<<"Original size : "<<model_read->points.size()<<endl;

	if (argc==3)
	{
		
		downsample(model_read, model_read);
		PassThroughFilter(model_read, model_read);

		cout<<"Filter size : "<<model_read->points.size()<<endl;
		
		pcl::io::savePCDFileASCII(argv[2], *model_read);
	}


	/////////////////////////PCL Visualization//////////////////////////
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
	viewer = simpleVis(model_read);

	while (!viewer->wasStopped ())
	{
		viewer->spinOnce (100);
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}

	return 0;
}

