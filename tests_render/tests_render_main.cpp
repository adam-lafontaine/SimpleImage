#include "../src/util/execute.hpp"
#include "../src/app/app.hpp"
#include "tests_def.hpp"


//#define LEAK_CHECK
#if defined(_WIN32) && defined(_DEBUG) && defined(LEAK_CHECK)
#include "../src/util/win32_leak_check.h"
#endif



constexpr auto APP_TITLE = "SimpleImage SDL2 Tests";
constexpr auto APP_VERSION = "1.0";



static void do_nothing()
{

}


static void run_selected_test(Input const& input, app::AppState& app_state)
{
	static int test_id = -1;

	static std::vector<std::function<void(img::View const&)>> funcs = 
	{
		//fill_platform_view_test,
		//copy_image_test,
		//resize_image_test,
		histogram_image_test,
	};

	if (!input.keyboard.space_key.pressed)
	{
		return;
	}

	test_id++;

	if (test_id >= funcs.size())
	{
		test_id = 0;
	}

	auto& screen_out = app_state.screen_pixels;

	funcs[test_id](screen_out);
}


static void process_input(Input const& input, app::AppState& app_state)
{
	execute({
		[&]() { run_selected_test(input, app_state); },
		do_nothing
	});
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
	window_settings.screen_width = 900;
	window_settings.screen_height = 700;

	app::AppState app_state;

	if (!render_init(window_settings, app_state))
	{
		return EXIT_FAILURE;
	}

	render_run(app_state, [&](auto const& input) { process_input(input, app_state); });

	return EXIT_SUCCESS;
}