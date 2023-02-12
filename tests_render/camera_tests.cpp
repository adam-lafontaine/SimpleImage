#include "tests_include.hpp"

#include <array>
#include <algorithm>
#include <cmath>


constexpr u32 BIN_SPACE = 1;
constexpr u32 HIST_SPACE = 5;
constexpr auto N_BINS = 64;


class HistParams
{
public:
	u32 n_bins;
	u32 bin_width;
	u32 bin_space;
	u32 hist_height;
	u32 hist_space;
};


static void fill_to_top(img::View1r32 const& view, r32 value, u8 color)
{
	assert(value >= 0.0f);
	assert(value <= 1.0f);

	int y_begin = (int)(view.height * (1.0f - value));

	auto r = make_range(view);

	if (y_begin < 0)
	{
		r.y_begin = 0;
	}
	else if ((u32)y_begin >= view.height)
	{
		return;
	}
	else
	{
		r.y_begin = (u32)y_begin;
	}

	img::fill(img::sub_view(view, r), color);
}


static void draw_histogram(const r32* values, img::View1r32 const& dst, HistParams const& props)
{
	u32 space_px = props.bin_space;
	auto width = props.bin_width;

	auto max = *std::max_element(values, values + props.n_bins);

	img::fill(dst, 128);

	auto r = make_range(width, dst.height);

	for (u32 i = 0; i < props.n_bins; ++i)
	{
		auto val = max == 0.0f ? 0.0f : values[i] / max;

		fill_to_top(img::sub_view(dst, r), val, 0);

		r.x_begin += (width + space_px);
		r.x_end += (width + space_px);
	}
}


static void draw(img::Histogram12r32& hists, img::View1r32 const& dst, HistParams const& props)
{
	img::fill(dst, 255);

	u32 space_px = props.hist_space;
	auto height = props.hist_height;

	auto r = make_range(dst.width, height);
	r.x_begin = space_px;
	r.x_end -= space_px;

	for (u32 i = 0; i < 12; ++i)
	{
		r.y_begin += space_px;
		r.y_end += space_px;

		draw_histogram(hists.list[i], img::sub_view(dst, r), props);

		r.y_begin += height;
		r.y_end += height;
	}
}


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

	auto const to_hypot = [](r32 grad_x, r32 grad_y) { return std::hypotf(grad_x, grad_y); };

	auto const grab_cb = [&](img::View const& src)
	{
		img::map_rgb(src, rgb);
		img::transform_gray(rgb, gray);
		img::gradients_xy(gray, grad);
		img::transform(grad, gray, to_hypot);
		img::map_rgb(gray, dst);
	};

	Image grab_image;
	img::create_image(grab_image, dst.width, dst.height);

	if (!img::grab_image(camera, grab_cb, img::make_view(grab_image)))
	{
		
	}

	mb::destroy_buffer(buffer);
	img::close_all_cameras();
}


void camera_histogram_test(img::View const& out)
{
	img::CameraUSB camera;

	if (!img::open_camera(camera))
	{
		return;
	}

	auto width = out.width;
	auto height = out.height;

	HistParams params{};
	params.n_bins = N_BINS;
	params.bin_space = BIN_SPACE;
	params.hist_space = HIST_SPACE;
	params.bin_width = (width + BIN_SPACE - 2 * HIST_SPACE) / N_BINS - BIN_SPACE;
	params.hist_height = (height - HIST_SPACE) / 12 - HIST_SPACE;

	img::Buffer32 buffer;
	mb::create_buffer(buffer, width * height);

	auto hist_view = img::make_view_1(width, height, buffer);

	img::Histogram12r32 hists;
	hists.n_bins = N_BINS;

	auto const grab_cb = [&](img::View const& src)
	{
		img::make_histograms(src, hists);
		draw(hists, hist_view, params);
		img::map_rgb(hist_view, out);
	};

	Image grab_image;
	img::create_image(grab_image, width, height);

	if (!img::grab_image(camera, grab_cb, img::make_view(grab_image)))
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