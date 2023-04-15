#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "SimpleImage Planar Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;


bool hsv_conversion_test();
bool yuv_conversion_test();
bool lch_conversion_test();

void map_rgba_tests(img::View const& out);
void map_rgb_tests(img::View const& out);
void map_gray_tests(img::View const& out);
void fill_rgba_test(img::View const& out);
void fill_rgb_test(img::View const& out);
void fill_gray_test(img::View const& out);
void hsv_draw_test(img::View const& out);
void yuv_draw_test(img::View const& out);
void lch_draw_test(img::View const& out);
void transform_test(img::View const& out);
void transform_gray_test(img::View const& out);
void threshold_test(img::View const& out);
void binarize_test(img::View const& out);
void alpha_blend_test(img::View const& out);
void rotate_rgb_test(img::View const& out);
void rotate_gray_test(img::View const& out);
void blur_gray_test(img::View const& out);
void blur_rgb_test(img::View const& out);
void gradients_tests(img::View const& out);
void gradients_xy_tests(img::View const& out);


static std::vector<std::function<void(img::View const&)>> tests = 
{
	map_rgba_tests,
	map_rgb_tests,
	map_gray_tests,
	fill_rgba_test,
	fill_rgb_test,
	fill_gray_test,
	hsv_draw_test,
	yuv_draw_test,
	lch_draw_test,
	transform_test,
	transform_gray_test,
	threshold_test,
	binarize_test,
	alpha_blend_test,
	rotate_rgb_test,
	rotate_gray_test,
	blur_gray_test,
	blur_rgb_test,
	gradients_tests,
	gradients_xy_tests,
};


static bool run_preliminary_tests()
{
	return directory_files_test() &&
		hsv_conversion_test() &&
		yuv_conversion_test() &&
		lch_conversion_test() &&
		true;
}


static void run_selected_test(Input const& input, app::AppState& app_state)
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
		app_state.signal_stop = true;
		return;
	}

	tests[test_id](app_state.screen_pixels);
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

    render_run(app_state, [&](auto const& input) { run_selected_test(input, app_state); });

    return EXIT_SUCCESS;
}