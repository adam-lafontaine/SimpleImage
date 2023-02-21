#pragma once
#include "tests_def.hpp"
#include "../src/simage/simage.hpp"


#include <filesystem>

namespace fs = std::filesystem;
namespace img = simage;


using Image = img::Image;
using ImageView = img::View;
using GrayImage = img::ImageGray;
using GrayView = img::ViewGray;
using Pixel = img::Pixel;

using path_t = fs::path;

#ifdef _WIN32

// set this directory for your system
//constexpr auto ROOT_DIR = "../../../tests/";
constexpr auto ROOT_DIR = "C:\\D_Data\\Repos\\SimpleImage\\tests";

#else

// set this directory for your system
constexpr auto ROOT_DIR = "/home/adam/Repos/SimpleImage/tests/";

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

