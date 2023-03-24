#include "../../src/util/execute.hpp"
#include "../../src/app/app.hpp"
#include "../render_usb/proc_def.hpp"

constexpr auto APP_TITLE = "Render USB";
constexpr auto APP_VERSION = "1.0";

static std::vector<std::function<void(img::View const&, img::View const&)>> proc_list =
{
	show_camera,
	show_blur,
	show_gray,
	show_gradients,
	show_gradients_red,
	show_gradients_green,
	show_gradients_blue,
};


static void run_selected_proc(Input const& input, img::CameraUSB const& camera, img::View const& dst)
{
	static int proc_id = -1;

	if (input.keyboard.space_key.pressed)
	{
		proc_id++;

		if (proc_id >= proc_list.size())
		{
			proc_id = 0;
		}
	}

	if (proc_id < 0 || proc_id >= proc_list.size())
	{
		return;
	}

	img::grab_rgb(camera, [&](img::View const& src) { proc_list[proc_id](src, dst); });
}


static void adjust_screen_views(img::CameraUSB& camera, img::View& app_screen)
{
	if (camera.frame_width == app_screen.width && camera.frame_height == app_screen.height)
	{
		return;
	}	

	// change camera roi if it is larger than the screen
	auto roi_camera = make_range(camera.frame_width, camera.frame_height);

	if (camera.frame_width > app_screen.width)
	{
		roi_camera.x_begin = (camera.frame_width - app_screen.width) / 2;
		roi_camera.x_end = roi_camera.x_begin + app_screen.width;
	}

	if (camera.frame_height > app_screen.height)
	{
		roi_camera.y_begin = (camera.frame_height - app_screen.height) / 2;
		roi_camera.y_end = roi_camera.y_begin + app_screen.height;
	}

	img::set_roi(camera, roi_camera);

	// screen view that fits in camera roi
	u32 x_adj_screen = 0;
	u32 y_adj_screen = 0;
	
	if (camera.frame_width < app_screen.width)
	{
		x_adj_screen = (app_screen.width - camera.frame_width) / 2;
	}
	
	if (camera.frame_height < app_screen.height)
	{
		y_adj_screen = (app_screen.height - camera.frame_height) / 2;
	}
	
	auto roi_screen = make_range(camera.frame_width, camera.frame_height);

	roi_screen.x_begin += x_adj_screen;
	roi_screen.x_end += x_adj_screen;
	roi_screen.y_begin += y_adj_screen;
	roi_screen.y_end += y_adj_screen;	

	app_screen = img::sub_view(app_screen, roi_screen);
}


int main()
{
	img::CameraUSB camera;
	if (!img::open_camera(camera))
	{
		return EXIT_FAILURE;
	}

	app::WindowSettings window_settings{};
	window_settings.app_title = APP_TITLE;
	window_settings.version = APP_VERSION;
	window_settings.screen_width = camera.frame_width;
	window_settings.screen_height = camera.frame_height;

	app::AppState app_state;

	if (!render_init(window_settings, app_state))
	{
		img::close_camera(camera);
		return EXIT_FAILURE;
	}	

	auto full_screen = app_state.screen_pixels;

	auto out_view = full_screen;

	adjust_screen_views(camera, out_view);
	
	if (!init_camera_procs(camera))
	{
		return EXIT_FAILURE;
	}

	render_run(app_state, [&](auto const& input) { run_selected_proc(input, camera, out_view); });

	img::close_camera(camera);
	close_camera_procs();

	return EXIT_SUCCESS;
}