#pragma once

#include "../../simage/simage.hpp"

#include <filesystem>


namespace fs = std::filesystem;
namespace img = simage;

using path_t = fs::path;

#ifdef _WIN32

// set this directory for your system
constexpr auto ROOT_DIR = "C:\\D_Data\\Repos\\SimpleImage\\test_app";

#else

// set this directory for your system
constexpr auto ROOT_DIR = "/home/adam/Repos/SimpleImage/test_app";

#endif // _WIN32


constexpr auto IMAGE_IN_DIR = "in_files/";

const auto ROOT_PATH = path_t(ROOT_DIR);
const auto IMAGE_IN_PATH = ROOT_PATH / IMAGE_IN_DIR;

const auto CORVETTE_PATH = IMAGE_IN_PATH / "corvette.png";
const auto CADILLAC_PATH = IMAGE_IN_PATH / "cadillac.png";
const auto WEED_PATH = IMAGE_IN_PATH / "weed.png";
const auto CHESS_PATH = IMAGE_IN_PATH / "chess_board.bmp";


inline bool directory_files_test()
{
    auto title = "directory_files_test";
	printf("\n%s:\n", title);

	auto const test_dir = [](path_t const& dir)
	{
		auto result = fs::is_directory(dir);
		auto msg = result ? "PASS" : "FAIL";
		printf("%s: %s\n", dir.string().c_str(), msg);

		return result;
	};

	auto result =
		test_dir(ROOT_PATH) &&
		test_dir(IMAGE_IN_PATH);

	auto const test_file = [](path_t const& file)
	{
		auto result = fs::exists(file);
		auto msg = result ? "PASS" : "FAIL";
		printf("%s: %s\n", file.string().c_str(), msg);

		return result;
	};

	result =
		result &&
		test_file(CORVETTE_PATH) &&
		test_file(CADILLAC_PATH) &&
		test_file(WEED_PATH) &&
		test_file(CHESS_PATH);

	return result;
}


void fill_test(img::View const& out);

void fill_gray_test(img::View const& out);

void copy_test(img::View const& out);

void copy_gray_test(img::View const& out);

void resize_image_test(img::View const& out);

void resize_gray_image_test(img::View const& out);

void split_channels_red_test(img::View const& out);

void split_channels_green_test(img::View const& out);

void split_channels_blue_test(img::View const& out);

void alpha_blend_test(img::View const& out);

void transform_test(img::View const& out);

void transform_gray_test(img::View const& out);

void threshold_min_test(img::View const& out);

void threshold_min_max_test(img::View const& out);

void binarize_test(img::View const& out);

void binarize_rgb_test(img::View const& out);

void blur_test(img::View const& out);

void gradients_test(img::View const& out);

void gradients_xy_test(img::View const& out);

void rotate_test(img::View const& out);

void rotate_gray_test(img::View const& out);

void centroid_test(img::View const& out);

void skeleton_test(img::View const& out);