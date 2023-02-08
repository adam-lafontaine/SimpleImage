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
		
	}

	img::close_all_cameras();
}


void camera_callback_test(img::View const& out)
{
	img::CameraUSB camera;

	if (!img::open_camera(camera))
	{
		return;
	}

	auto width = camera.image_width;
	auto height = camera.image_height;

	auto dst = img::sub_view(out, make_range(width, height));

	img::Buffer32 buffer;
	mb::create_buffer(buffer, width * height * 6);

	auto rgb = img::make_view_3(width, height, buffer);
	auto gray = img::make_view_1(width, height, buffer);
	auto grad = img::make_view_2(width, height, buffer);
	auto dst_x = img::select_channel(grad, img::XY::X);
	auto dst_y = img::select_channel(grad, img::XY::Y);

	auto const to_avg_abs = [](r32 x, r32 y) { return ((x < 0.0f ? -x : x) + (y < 0.0f ? -y : y)) / 2; };

	auto const grab_cb = [&](img::ViewBGR const& src)
	{
		img::map_bgr_rgb(src, rgb);
		img::transform_gray(rgb, gray);
		img::gradients_xy(gray, grad);
		img::transform(grad, gray, to_avg_abs);
		img::map_rgb(gray, dst);
	};

	if (!img::grab_image(camera, grab_cb))
	{
		
	}

	mb::destroy_buffer(buffer);
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