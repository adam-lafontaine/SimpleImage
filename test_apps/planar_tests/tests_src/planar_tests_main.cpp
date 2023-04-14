#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "SimpleImage Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;


bool hsv_conversion_test();

void fill_rgba_test(img::View const& out);
void fill_rgb_test(img::View const& out);
void fill_gray_test(img::View const& out);
void hsv_draw_test(img::View const& out);


static std::vector<std::function<void(img::View const&)>> tests = 
{
	fill_rgba_test,
	fill_rgb_test,
	fill_gray_test,
	hsv_draw_test,
};


static bool run_preliminary_tests()
{
	return directory_files_test() &&
		//hsv_conversion_test() &&
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