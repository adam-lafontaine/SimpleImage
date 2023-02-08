#include "tests_include.hpp"


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


//void camera_continuous_test(img::View const& out)
//{
//	img::CameraUSB camera;
//
//	if (!img::open_camera(camera))
//	{
//		return;
//	}
//
//	auto dst = img::sub_view(out, make_range(camera.image_width, camera.image_height));
//
//	auto frame_count = 0;
//
//	auto const grab_condition = [&frame_count]() { return frame_count < 120; };
//
//	auto const grab_cb = [&](img::ViewBGR const& src)
//	{
//		img::map(src, dst);
//		++frame_count;
//	};
//
//	img::grab_continuous(camera, grab_cb, grab_condition);
//
//	img::close_all_cameras();
//}