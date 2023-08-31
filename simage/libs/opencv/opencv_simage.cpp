#include "../../simage.hpp"

#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>

#include <array>

namespace img = simage;


constexpr int N_CAMERAS = 1;


class DeviceCV
{
public:
	cv::VideoCapture capture;

	cv::Mat bgr_frame;
	img::ViewBGR bgr_view;

	img::Image cb_image;
	img::View cb_view;
	img::ViewGray cb_gray_view;
};


static DeviceCV g_devices[N_CAMERAS];


static void close_all_cameras()
{
	for (int i = 0; i < N_CAMERAS; ++i)
	{
		auto& device = g_devices[i];

		device.capture.release();
		device.bgr_frame.release();

		img::destroy_image(device.cb_image);
	}
}


static bool grab_and_convert_frame_bgr(DeviceCV& device)
{
	auto& cap = device.capture;

	if (!cap.grab())
	{
		return false;
	}

	auto& frame = device.bgr_frame;

	if (!cap.retrieve(frame))
	{
		return false;
	}

	device.bgr_view.data = (img::BGRu8*)frame.data;

	return true;
}


template <class ViewSRC, class ViewDST>
static void write_frame_sub_view_rgba(img::CameraUSB const& camera, ViewSRC const& src, ViewDST const& dst)
{
	auto width = std::min(camera.frame_width, dst.width);
	auto height = std::min(camera.frame_height, dst.height);
	auto r = make_range(width, height);

	img::map_rgba(img::sub_view(src, r), img::sub_view(dst, r));
}


template <class ViewSRC, class ViewDST>
static void write_frame_sub_view_gray(img::CameraUSB const& camera, ViewSRC const& src, ViewDST const& dst)
{
	auto width = std::min(camera.frame_width, dst.width);
	auto height = std::min(camera.frame_height, dst.height);
	auto r = make_range(width, height);

	img::map_gray(img::sub_view(src, r), img::sub_view(dst, r));
}


namespace simage
{
	static bool camera_is_initialized(CameraUSB const& camera)
	{
		return camera.is_open
			&& camera.device_id >= 0
			&& camera.device_id < N_CAMERAS;
	}


	bool open_camera(CameraUSB& camera)
	{
		auto& device = g_devices[0]; // support one device only
		auto& cap = device.capture;

		cap = cv::VideoCapture(1);
		if (!cap.isOpened())
		{
			cap = cv::VideoCapture(0);
			if (!cap.isOpened())
			{
				return false;
			}
		}

		camera.device_id = 0; // support one device only
		camera.frame_width = (u32)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		camera.frame_height = (u32)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		camera.max_fps = (u32)cap.get(cv::CAP_PROP_FPS);

		if (!create_image(device.cb_image, camera.frame_width, camera.frame_height))
		{
			cap.release();
			return false;
		}
		
		device.bgr_view.width = camera.frame_width;
		device.bgr_view.height = camera.frame_height;
		device.bgr_view.data = (BGRu8*)(12345);

		device.cb_view = make_view(device.cb_image);
		device.cb_gray_view.width = camera.frame_width;
		device.cb_gray_view.height = camera.frame_height;
		device.cb_gray_view.data = (u8*)device.cb_image.data_;

		camera.is_open = true;

		assert(camera.frame_width);
		assert(camera.frame_height);
		assert(camera.max_fps);

		return true;
	}


	void close_camera(CameraUSB& camera)
	{
		camera.is_open = false;

		if (camera.device_id < 0 || camera.device_id >= N_CAMERAS)
		{
			return;
		}

		close_all_cameras();
	}


	bool grab_rgb(CameraUSB const& camera, View const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		if (camera.frame_width == dst.width && camera.frame_height == dst.height)
		{
			map_rgba(device.bgr_view, dst);
		}
		else
		{
			write_frame_sub_view_rgba(camera, device.bgr_view, dst);
		}

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, SubView const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		write_frame_sub_view_rgba(camera, device.bgr_view, dst);

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, Range2Du32 const& roi, View const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		write_frame_sub_view_rgba(camera, sub_view(device.bgr_view, roi), dst);

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, Range2Du32 const& roi, SubView const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		write_frame_sub_view_rgba(camera, sub_view(device.bgr_view, roi), dst);

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, view_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_devices[camera.device_id];

		if (!grab_rgb(camera, device.cb_view))
		{
			return false;
		}

		grab_cb(device.cb_view);

		return true;
	}


	bool grab_rgb_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_rgba(device.bgr_view, device.cb_view);
				grab_cb(device.cb_view);
			}
		}

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, Range2Du32 const& roi, view_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_devices[camera.device_id];

		if (!grab_rgb(camera, roi, device.cb_view))
		{
			return false;
		}

		grab_cb(device.cb_view);

		return true;
	}


	bool grab_rgb_continuous(CameraUSB const& camera, Range2Du32 const& roi, view_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_rgba(sub_view(device.bgr_view, roi), device.cb_view);
				grab_cb(device.cb_view);
			}
		}

		return true;
	}


	bool grab_gray(CameraUSB const& camera, ViewGray const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		if (camera.frame_width == dst.width && camera.frame_height == dst.height)
		{
			map_gray(device.bgr_view, dst);
		}
		else
		{
			write_frame_sub_view_gray(camera, device.bgr_view, dst);
		}

		return true;
	}


	bool grab_gray(CameraUSB const& camera, SubViewGray const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		write_frame_sub_view_gray(camera, device.bgr_view, dst);

		return true;
	}


	bool grab_gray(CameraUSB const& camera, Range2Du32 const& roi, ViewGray const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		write_frame_sub_view_gray(camera, sub_view(device.bgr_view, roi), dst);

		return true;
	}


	bool grab_gray(CameraUSB const& camera, Range2Du32 const& roi, SubViewGray const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		if (!grab_and_convert_frame_bgr(device))
		{
			return false;
		}

		write_frame_sub_view_gray(camera, sub_view(device.bgr_view, roi), dst);

		return true;
	}


	bool grab_gray(CameraUSB const& camera, view_gray_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_devices[camera.device_id];

		if (!grab_gray(camera, device.cb_gray_view))
		{
			return false;
		}

		grab_cb(device.cb_gray_view);

		return true;
	}


	bool grab_gray_continuous(CameraUSB const& camera, view_gray_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_gray(device.bgr_view, device.cb_gray_view);
				grab_cb(device.cb_gray_view);
			}
		}

		return true;
	}


	bool grab_gray(CameraUSB const& camera, Range2Du32 const& roi, view_gray_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_devices[camera.device_id];

		if (!grab_gray(camera, roi, device.cb_gray_view))
		{
			return false;
		}

		grab_cb(device.cb_gray_view);

		return true;


		return true;
	}


	bool grab_gray_continuous(CameraUSB const& camera, Range2Du32 const& roi, view_gray_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_gray(sub_view(device.bgr_view, roi), device.cb_gray_view);
				grab_cb(device.cb_gray_view);
			}
		}

		return true;
	}
}