/*
* Copyright (c) 2016 Vincent Ee -=- iceira.edu.tw>
*
* All rights reserved.
*
* $Id: fsra.cpp 2016-06-12 15:24:30Z  $
*
*/
/** \author Vincent Ee
*

* This node implemented FSRA to localize the sensor pose in the given RGBD-MAP.

* Subscribe :sensor_msgs::ImageConstPtr, sensor_msgs::PointCloud2ConstPtr.
* Publish : geometry_msgs::PoseStamped, sensor_msgs::PointCloud2.

* Usage example : rosrun fsra fsra
*/


// FSRA class include
#include "fsra.h"

// ROS specific include
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_broadcaster.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <sensor_msgs/Image.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

// Declare point cloud map publisher
ros::Publisher pub_cloud_map;
sensor_msgs::PointCloud2 output_map;
// Declare sensor pose publisher
ros::Publisher pub_pose;
geometry_msgs::PoseStamped output_pose;

void callback (const sensor_msgs::ImageConstPtr& image, const sensor_msgs::PointCloud2ConstPtr& pointcloud)
{
	// CV data conversion

	cv_bridge::CvImagePtr cv_msg = cv_bridge::toCvCopy(image, "bgr8");
	cv::Mat query_img;
	cv::cvtColor(cv_msg->image,query_img,CV_BGR2GRAY);
	//query_img = cv::imread("/home/vincentee/code/asus_nav/data/map_images_303-v1/scene/scene_4.jpg", CV_LOAD_IMAGE_GRAYSCALE );

	// PCL data conversion

	pcl::PCLPointCloud2 pc2;
	pcl_conversions::toPCL(*pointcloud, pc2);
	pcl::PointCloud<PointType>::Ptr query_cloud(new pcl::PointCloud<PointType>);
	pcl::fromROSMsg (pc2, *query_cloud);

	// Execute FSRA

	FSRA t(query_img, query_cloud);

	// Pose display

	cout<<pose.matrix()<<endl;

	// ~~~~~~~~~~ Output publising ~~~~~~~~~~~~~

	// Map

	pcl::toROSMsg (*map_cloud, output_map);
	output_map.header.frame_id = "/map";
	output_map.header.stamp = ros::Time::now ();
	pub_cloud_map.publish (output_map);

	// Pose

	Eigen::Matrix4f mat4 = pose.matrix();
	output_pose.pose.position.x = mat4(0,3);
	output_pose.pose.position.y = mat4(1,3);
	output_pose.pose.position.z = mat4(2,3);
	Eigen::Matrix3f mat_rot = pose.rotation();
	Eigen::Quaternionf q(mat_rot);
	output_pose.pose.orientation.w = q.w();
	output_pose.pose.orientation.x = q.x();
	output_pose.pose.orientation.y = q.y();
	output_pose.pose.orientation.z = q.z();
	output_pose.header.frame_id = "/map";
	output_pose.header.stamp = ros::Time::now ();
	pub_pose.publish (output_pose);
}

int main (int argc, char** argv)
{

	// Set the dictionary with the vocabulary

	cv::FileStorage fs(foldername[0]+"/dictionary.yml", cv::FileStorage::READ);
	fs["vocabulary"] >> dictionary;
	fs.release();
	bowDE.setVocabulary(dictionary);

	// Initialize ROS

	ros::init (argc, argv, "fsra");
	ros::NodeHandle nh;
	ros::Rate loop_rate(1000);

	ROS_INFO("---------- Start FSRA-Localization ----------");

	// Subscribe RGB-image & Point-Cloud

	message_filters::Subscriber<sensor_msgs::Image> image_sub(nh, "/camera/rgb/image_color", 1);
	message_filters::Subscriber<sensor_msgs::PointCloud2> cloud_sub(nh, "/camera/depth_registered/points", 1);

	typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, sensor_msgs::PointCloud2> MySyncPolicy;

	message_filters::Synchronizer<MySyncPolicy> sync(MySyncPolicy(10), image_sub, cloud_sub);

	sync.registerCallback(boost::bind(&callback, _1, _2));

	//Create a ROS publisher for the point cloud map output

	pub_cloud_map = nh.advertise<sensor_msgs::PointCloud2> ("cloud_map", 10);

	//Create a ROS publisher for the sensor pose output

	pub_pose = nh.advertise<geometry_msgs::PoseStamped> ("pose", 10);

	while(ros::ok())
	{
		ros::spinOnce();
		loop_rate.sleep();
	}

	return 0;
}
