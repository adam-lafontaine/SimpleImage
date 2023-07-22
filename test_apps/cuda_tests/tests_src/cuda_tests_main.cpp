#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "SimpleImage CUDA Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;


bool device_buffer_tests();
bool unified_buffer_tests();

void copy_device_test(img::View const& out);
void copy_device_gray_test(img::View const& out);
void copy_device_sub_view_test(img::View const& out);
void copy_device_sub_view_gray_test(img::View const& out);
void rgb_gray_test(img::View const& out);
void alpha_blend_test(img::View const& out);
void threshold_min_test(img::View const& out);
void threshold_min_max_test(img::View const& out);
void blur_gray_test(img::View const& out);
void blur_rgb_test(img::View const& out);
void gradients_test(img::View const& out);
void gradients_xy_test(img::View const& out);
void rotate_rgb_test(img::View const& out);
void rotate_gray_test(img::View const& out);


static std::vector<std::function<void(img::View const&)>> tests = 
{
	copy_device_test,
	copy_device_gray_test,
	copy_device_sub_view_test,
	copy_device_sub_view_gray_test,
	rgb_gray_test,
	alpha_blend_test,
	threshold_min_test,
	threshold_min_max_test,
	blur_gray_test,
	blur_rgb_test,
	gradients_test,
	gradients_xy_test,
	rotate_rgb_test,
	rotate_gray_test,
};


static bool run_preliminary_tests()
{
	return directory_files_test() &&
		device_buffer_tests() &&
		unified_buffer_tests() &&
		true;
}


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