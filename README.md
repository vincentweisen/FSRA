# fsra

//////////////////////////////////////////////////////////////////////////

//////////////// Training Stage (地圖與影像軌跡建立) ///////////////

//////////////////////////////////////////////////////////////////////////

啓動kinect：

    roslaunch openni_launch openni.launch
    
同時執行：

	roslaunch rgbdslam rgbdslam.launch
	
	rosrun image_view extract_images image:=/camera/rgb/image_color _filename_format:=map_images_306-v2/frame%04i.jpg  _sec_per_frame:=1.0
	
建完同時暫停

存取地圖 + 存取軌跡

地圖preprocess：

	./build/preprocess data/map_images_306-v2/map_306-v2.pcd data/map_images_306-v2/map.pcd
Surf-BoF training:

在/code/asus_nav文件夾更改src/fsra_src/feature_offline_extraction.cpp的frame_num和foldername

編譯：

	make
	
執行

	./build/feature_offline_extraction
	
///////////////////////////////////////////////////////////////////

//////////////// Testing Stage (視覺定位) ////////////////////

///////////////////////////////////////////////////////////////////

FSRA執行：

	在ROS的FSRA文件夾更改vim src/fsra_v2.cpp的 frame_num和 foldername

編譯：

	catkin_make --pkg fsra

執行：

	rosrun fsra fsra_v2

Usage：

	rosrun fsra fsra_v2  --> Real-time coarse pose estimation

	視窗按'p'  -->  Point-cloud Alignment

	關閉視窗 -->   Exact pose estimation 

	視窗按'i'  -->  恢復 coarse pose estimation
