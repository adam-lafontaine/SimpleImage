#include "../app/app.hpp"

constexpr auto APP_TITLE = "SimpleImage Test App";
constexpr auto APP_VERSION = "1.0";


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

    render_run(app_state, [&](auto const& input) {  });

    return EXIT_SUCCESS;
}