#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "SimpleImage Interleaved Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;


bool memory_buffer_tests();


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


static std::vector<std::function<void(img::View const&)>> tests = 
{
	fill_test,
	fill_gray_test,
	copy_test,
	copy_gray_test,
	resize_image_test,
	resize_gray_image_test,
	split_channels_red_test,
	split_channels_green_test,
	split_channels_blue_test,
	alpha_blend_test,
	transform_test,
	transform_gray_test,
	threshold_min_test,
	threshold_min_max_test,
	binarize_test,
	binarize_rgb_test,
	blur_test,
	gradients_test,
	gradients_xy_test,
	rotate_test,
	rotate_gray_test,
	centroid_test,
	skeleton_test
};


static void run_next_test(Input const& input, app::AppState& app_state)
{
	static int test_id = -1;
	static int frame_count = 0;

	frame_count++;
	if (frame_count < FRAMES_PER_TEST)
	{
		return;
	}

	frame_count = 0;
	test_id++;	

	if (test_id >= tests.size())
	{
		app_state.stop();
		return;
	}

	tests[test_id](app_state.screen_buffer[0]);
}


static bool run_preliminary_tests()
{
	return directory_files_test() &&
		memory_buffer_tests() &&
		true;
}


int main()
{
	if (!run_preliminary_tests())
	{
		return EXIT_FAILURE;
	}

    app::WindowSettings window_settings{};
	window_settings.app_title = APP_TITLE;
	window_settings.version = APP_VERSION;
	window_settings.screen_width = 1280;
	window_settings.screen_height = 720;

	app::AppState app_state;

    if (!render_init(window_settings, app_state))
	{
		return EXIT_FAILURE;
	}

	app_state.start();

    render_run(app_state, [&](auto const& input) { run_next_test(input, app_state); });

    return EXIT_SUCCESS;
}