#include "../../app/app.hpp"
#include "../../tests_include.hpp"

constexpr auto APP_TITLE = "USB Camera Test App";
constexpr auto APP_VERSION = "1.0";

// display each test result for ~0.5 seconds
constexpr int FRAMES_PER_TEST = 30;



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

    render_run(app_state, [&](auto const& input) { /*run_next_test(input, app_state);*/ });

    return EXIT_SUCCESS;
}