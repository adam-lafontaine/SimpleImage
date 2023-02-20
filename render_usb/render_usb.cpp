#include "proc_def.hpp"
#include "../src/simage/simage.hpp"

#include <cmath>


// memory
img::Buffer32 buffer;
img::View3r32 view3a;
img::View2r32 view2a;
img::View1r32 view1a;
img::View3r32 view3b;


// tests
img::View3r32 view_rgb;
img::View1r32 view_gray;
img::View2r32 view_grad;
img::View3r32 view_blur;
img::View1r32 view_red;
img::View1r32 view_green;
img::View1r32 view_blue;


static r32 to_hypot(r32 grad_x, r32 grad_y)
{
	return std::hypotf(grad_x, grad_y);
}


void close_camera_procs()
{
	mb::destroy_buffer(buffer);
}


bool init_camera_procs(img::CameraUSB const& camera)
{
	auto width = camera.frame_roi.width;
	auto height = camera.frame_roi.height;

	auto n_channels = 9;

	if (!mb::create_buffer(buffer, width * height * n_channels))
	{
		return false;
	}

	view3a = img::make_view_3(width, height, buffer);
	view2a = img::make_view_2(width, height, buffer);
	view1a = img::make_view_1(width, height, buffer);
	view3b = img::make_view_3(width, height, buffer);

	view_rgb = view3a;
	view_gray = view1a;
	view_grad = view2a;
	view_blur = view3b;
	view_red = img::select_channel(view_rgb, img::RGB::R);
	view_green = img::select_channel(view_rgb, img::RGB::G);
	view_blue = img::select_channel(view_rgb, img::RGB::B);

	return true;
}


void show_camera(img::View const& src, img::View const& dst)
{
	img::copy(src, dst);
}


void show_blur(img::View const& src, img::View const& dst)
{
	img::map_rgb(src, view_rgb);
	img::blur(view_rgb, view_blur);
	img::map_rgb(view_blur, dst);
}


void show_gray(img::View const& src, img::View const& dst)
{
	img::map_rgb(src, view_rgb);
	img::transform_gray(view_rgb, view_gray);
	img::map_rgb(view_gray, dst);
}


void show_gradients(img::View const& src, img::View const& dst)
{
	img::map_rgb(src, view_rgb);
	img::transform_gray(view_rgb, view_gray);
	img::gradients_xy(view_gray, view_grad);
	img::transform(view_grad, view_gray, to_hypot);
	img::map_rgb(view_gray, dst);
}


void show_gradients_red(img::View const& src, img::View const& dst)
{
	img::map_rgb(src, view_rgb);
	img::transform_gray(view_rgb, view_gray);
	img::gradients_xy(view_gray, view_grad);

	img::transform(view_grad, view_red, to_hypot);
	img::fill(view_green, 0);
	img::fill(view_blue, 0);

	img::map_rgb(view_rgb, dst);
}


void show_gradients_green(img::View const& src, img::View const& dst)
{
	img::map_rgb(src, view_rgb);
	img::transform_gray(view_rgb, view_gray);
	img::gradients_xy(view_gray, view_grad);

	img::transform(view_grad, view_green, to_hypot);
	img::fill(view_red, 0);
	img::fill(view_blue, 0);

	img::map_rgb(view_rgb, dst);
}


void show_gradients_blue(img::View const& src, img::View const& dst)
{
	img::map_rgb(src, view_rgb);
	img::transform_gray(view_rgb, view_gray);
	img::gradients_xy(view_gray, view_grad);

	img::transform(view_grad, view_blue, to_hypot);
	img::fill(view_green, 0);
	img::fill(view_red, 0);

	img::map_rgb(view_rgb, dst);
}