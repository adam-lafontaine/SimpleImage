#include "../../src/simage/simage.hpp"

#include <filesystem>


namespace img = simage;
namespace fs = std::filesystem;


using Image = img::Image;
using ImageView = img::View;
using GrayImage = img::ImageGray;
using GrayView = img::ViewGray;
using Pixel = img::Pixel;

using path_t = fs::path;

#ifdef _WIN32

// set this directory for your system
//constexpr auto ROOT_DIR = "../../../tests/";
constexpr auto ROOT_DIR = "C:\\D_Data\\Repos\\SimpleImage\\examples\\tests";

#else

// set this directory for your system
constexpr auto ROOT_DIR = "../";

#endif // _WIN32


constexpr auto TEST_IMAGE_DIR = "TestImages/";
constexpr auto IMAGE_IN_DIR = "in_files/";
constexpr auto IMAGE_OUT_DIR = "out_files/";

const auto ROOT_PATH = path_t(ROOT_DIR);
const auto TEST_IMAGE_PATH = ROOT_PATH / TEST_IMAGE_DIR;
const auto IMAGE_IN_PATH = TEST_IMAGE_PATH / IMAGE_IN_DIR;
const auto IMAGE_OUT_PATH = TEST_IMAGE_PATH / IMAGE_OUT_DIR;

const auto CORVETTE_PATH = IMAGE_IN_PATH / "corvette.png";
const auto CADILLAC_PATH = IMAGE_IN_PATH / "cadillac.png";
const auto WEED_PATH = IMAGE_IN_PATH / "weed.png";
const auto CHESS_PATH = IMAGE_IN_PATH / "chess_board.bmp";


inline void empty_dir(path_t const& dir)
{
	fs::create_directories(dir);

	for (auto const& entry : fs::directory_iterator(dir))
	{
		fs::remove_all(entry);
	}
}


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
		test_dir(TEST_IMAGE_PATH) &&
		test_dir(IMAGE_IN_PATH) &&
		test_dir(IMAGE_OUT_PATH);

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


bool execute_tests();

bool memory_buffer_tests();

bool stb_simage_tests();

bool create_image_tests();

bool make_view_tests();

bool map_tests();

bool map_rgb_tests();

bool map_rgb_hsv_tests();

bool map_rgb_lch_tests();

bool map_rgb_yuv_tests();

bool sub_view_tests();

bool fill_tests();

bool copy_tests();

bool histogram_tests();

bool gradients_tests();

bool blur_tests();
