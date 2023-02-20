#include "proc_def.hpp"
#include "../src/simage/simage.hpp"

#include <cmath>


// memory
img::Buffer32 buffer;
img::View3r32 view3a;
img::View2r32 view2a;
img::View1r32 view1a;


// tests
img::View3r32 view_rgb;
img::View1r32 view_gray;
img::View2r32 view_grad;


static r32 to_hypot(r32 grad_x, r32 grad_y)
{
	return std::hypotf(grad_x, grad_y);
}


void close_camera_procs()
{
	mb::destroy_buffer(buffer);
}


bool init_camera_procs(img::CameraUSB& camera)
{
	if (!img::open_camera(camera))
	{
		return false;
	}

	auto width = camera.image_width;
	auto height = camera.image_height;

	auto n_channels = 6;

	mb::create_buffer(buffer, width * height * n_channels);

	view3a = img::make_view_3(width, height, buffer);
	view2a = img::make_view_2(width, height, buffer);
	view1a = img::make_view_1(width, height, buffer);

	view_rgb = view3a;
	view_gray = view1a;
	view_grad = view2a;

	return true;
}


void show_camera(img::View const& src, img::View const& dst)
{
	img::copy(src, dst);
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