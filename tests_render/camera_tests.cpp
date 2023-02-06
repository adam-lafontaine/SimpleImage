#include "tests_include.hpp"

#include <opencv2/opencv.hpp>


void camera_test(img::View const& out)
{
	cv::Mat frame;

	cv::VideoCapture cam(0);

	cam >> frame;

	auto total_bytes = frame.dataend - frame.datastart;
	auto total_pixels = frame.cols * frame.rows;

	auto bytes_per_pixel = (r32)total_bytes / total_pixels;

	img::ImageBGR image;
	image.width = (u32)frame.cols;
	image.height = (u32)frame.rows;
	image.data_ = (img::BGR*)frame.data;

	auto view = img::make_view(image);

	auto dst = img::sub_view(out, make_range(view.width, view.height));

  	img::map(view, dst);

	frame.release();
}