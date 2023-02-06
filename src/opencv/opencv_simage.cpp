#include "../simage/simage_platform.hpp"

#include <opencv2/opencv.hpp>



namespace simage
{
	static void stuff()
	{
		cv::Mat frame;

		cv::VideoCapture cam(0);

		cam >> frame;
	}
}