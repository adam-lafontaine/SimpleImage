#include "../src/util/execute.hpp"
#include "../src/app/app.hpp"
#include "tests_def.hpp"


//#define LEAK_CHECK
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
#include "../src/util/win32_leak_check.h"
#endif



constexpr auto APP_TITLE = "SimpleImage SDL2 Tests";
constexpr auto APP_VERSION = "1.0";




static void run_tests(img::View const& screen_out)
{
	fill_platform_view_test(screen_out);
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
	window_settings.screen_width = 800;
	window_settings.screen_height = 600;

	app::AppSettings app_settings;

	if (!render_init(window_settings, app_settings))
	{
		return EXIT_FAILURE;
	}

	std::array<std::function<void()>, 2> f_list = 
	{
		[&]() { render_run(app_settings); },
		[&]() { run_tests(app_settings.screen_pixels); }
	};

	execute_parallel(f_list);

	return EXIT_SUCCESS;
}