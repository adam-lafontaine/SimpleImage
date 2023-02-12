#include "../src/util/execute.hpp"
#include "../src/app/app.hpp"
#include "tests_def.hpp"

//#define LEAK_CHECK
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
#include "../src/util/win32_leak_check.h"
#endif

#include <chrono>


constexpr auto APP_TITLE = "SimpleImage SDL2 Tests";
constexpr auto APP_VERSION = "1.0";


static std::vector<std::function<void(img::View const&)>> tests =
{
	fill_platform_view_test,
	//copy_image_test,
	//resize_image_test,
	histogram_image_test,
	camera_test,
	camera_callback_test,
	//camera_histogram_test,
	//camera_continuous_test,
};


static void run_selected_test(Input const& input, app::AppState& app_state)
{
	static int test_id = -1;	

	if (!input.keyboard.space_key.pressed)
	{
		return;
	}

	test_id++;

	if (test_id >= tests.size())
	{
		test_id = 0;
	}

	tests[test_id](app_state.screen_pixels);

	/*while (!app_state.signal_stop)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(10));

		if (!input.keyboard.space_key.pressed)
		{
  			continue;
		}

		test_id++;

		if (test_id >= funcs.size())
		{
			test_id = 0;
		}

		funcs[test_id](app_state.screen_pixels);
	}*/
}


int main()
{
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
	int dbgFlags = _CrtSetDbgFlag(_CRTDBG_REPORT_FLAG);
	dbgFlags |= _CRTDBG_CHECK_ALWAYS_DF;   // check block integrity
	dbgFlags |= _CRTDBG_DELAY_FREE_MEM_DF; // don't recycle memory
	dbgFlags |= _CRTDBG_LEAK_CHECK_DF;     // leak report on exit
	_CrtSetDbgFlag(dbgFlags);
#endif

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

	/*Input user_input;

	execute({
		[&]() { run_selected_test(user_input, app_state); },
		[&]() { render_run(app_state, [&](auto const& input) { user_input = input; }); }
	});*/

	render_run(app_state, [&](auto const& input) { run_selected_test(input, app_state); });

	return EXIT_SUCCESS;
}