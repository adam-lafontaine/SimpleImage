#include "tests_include.hpp"

//#include <opencv2/opencv.hpp>
//
//
//static cv::VideoCapture g_cam;


void camera_test(img::View const& out)
{
 	img::CameraUSB camera;

	if (!img::open_camera(camera))
	{
		return;
	}

	auto dst = img::sub_view(out, make_range(camera.image_width, camera.image_height));

	if (!img::grab_image(camera, dst))
	{
		img::close_all_cameras();
		return;
	}

	img::close_all_cameras();
}