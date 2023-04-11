#include "../app/app.hpp"

constexpr auto APP_TITLE = "SimpleImage Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;


static std::vector<std::function<void(img::View const&)>> tests = 
{

};


static void run_selected_test(Input const& input, app::AppState& app_state)
{
	static int test_id = -1;
	static int frame_count = 0;

	if (!input.keyboard.space_key.pressed)
	{
		return;
	}

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