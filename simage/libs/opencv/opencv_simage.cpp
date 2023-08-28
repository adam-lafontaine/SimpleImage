#include "../../simage.hpp"
#include "../../src/util/execute.hpp"

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

	img::ImageBGR bgr_image;
};


static DeviceCV g_devices[N_CAMERAS];


static void close_all_cameras()
{
	for (int i = 0; i < N_CAMERAS; ++i)
	{
		g_devices[i].capture.release();

		g_devices[i].bgr_frame.release();
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

	device.bgr_image.data_ = (img::BGRu8*)frame.data;

	return true;
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
		auto& device = g_devices[0];
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

		camera.device_id = 0;
		camera.frame_width = (u32)cap.get(cv::CAP_PROP_FRAME_WIDTH);
		camera.frame_height = (u32)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
		camera.max_fps = (u32)cap.get(cv::CAP_PROP_FPS);

		if (!create_image(camera.cb_image, camera.frame_width, camera.frame_height))
		{
			cap.release();
			return false;
		}

		ImageBGR bgr;
		device.bgr_image.width = camera.frame_width;
		device.bgr_image.height = camera.frame_height;
		device.bgr_image.data_ = (BGRu8*)(12345);

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
		
		destroy_image(camera.cb_image);

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
			map_rgba(img::make_view(device.bgr_image), dst);
		}
		else
		{
			auto width = std::min(camera.frame_width, dst.width);
			auto height = std::min(camera.frame_height, dst.height);
			auto r = make_range(width, height);

			map_rgba(img::sub_view(device.bgr_image, r), img::sub_view(dst, r));
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

		auto width = std::min(camera.frame_width, dst.width);
		auto height = std::min(camera.frame_height, dst.height);
		auto r = make_range(width, height);

		map_rgba(img::sub_view(device.bgr_image, r), img::sub_view(dst, r));

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

		auto device_view = sub_view(device.bgr_image, roi);

		auto width = std::min(device_view.width, dst.width);
		auto height = std::min(device_view.height, dst.height);
		auto r = make_range(width, height);

		map_rgba(img::sub_view(device_view, r), img::sub_view(dst, r));

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

		auto device_view = sub_view(device.bgr_image, roi);

		auto width = std::min(device_view.width, dst.width);
		auto height = std::min(device_view.height, dst.height);
		auto r = make_range(width, height);

		map_rgba(img::sub_view(device_view, r), img::sub_view(dst, r));

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, view_callback const& grab_cb)
	{
		assert(verify(camera));

		auto view = img::make_view(camera.cb_image);

		if (!grab_rgb(camera, view))
		{
			return false;
		}

		grab_cb(view);

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

		auto camera_view = img::make_view(camera.cb_image);

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_rgba(img::make_view(device.bgr_image), camera_view);
				grab_cb(camera_view);
			}
		}

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, Range2Du32 const& roi, view_callback const& grab_cb)
	{
		assert(verify(camera));

		img::View camera_view{};
		camera_view.data = camera.cb_image.data_;
		camera_view.width = roi.x_end - roi.x_begin;
		camera_view.height = roi.y_end - roi.y_end;

		if (!grab_rgb(camera, roi, camera_view))
		{
			return false;
		}

		grab_cb(camera_view);

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

		img::View camera_view{};
		camera_view.data = camera.cb_image.data_;
		camera_view.width = roi.x_end - roi.x_begin;
		camera_view.height = roi.y_end - roi.y_end;

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_rgba(sub_view(device.bgr_image, roi), camera_view);
				grab_cb(camera_view);
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
			map_gray(img::make_view(device.bgr_image), dst);
		}
		else
		{
			auto width = std::min(camera.frame_width, dst.width);
			auto height = std::min(camera.frame_height, dst.height);
			auto r = make_range(width, height);

			map_gray(img::sub_view(device.bgr_image, r), img::sub_view(dst, r));
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

		auto width = std::min(camera.frame_width, dst.width);
		auto height = std::min(camera.frame_height, dst.height);
		auto r = make_range(width, height);

		map_gray(img::sub_view(device.bgr_image, r), img::sub_view(dst, r));

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

		auto device_view = sub_view(device.bgr_image, roi);

		auto width = std::min(device_view.width, dst.width);
		auto height = std::min(device_view.height, dst.height);
		auto r = make_range(width, height);

		map_gray(img::sub_view(device_view, r), img::sub_view(dst, r));

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

		auto device_view = sub_view(device.bgr_image, roi);

		auto width = std::min(device_view.width, dst.width);
		auto height = std::min(device_view.height, dst.height);
		auto r = make_range(width, height);

		map_gray(img::sub_view(device_view, r), img::sub_view(dst, r));

		return true;
	}


	bool grab_gray(CameraUSB const& camera, view_gray_callback const& grab_cb)
	{
		assert(verify(camera));

		img::ViewGray camera_view{};
		camera_view.data = (u8*)camera.cb_image.data_;
		camera_view.width = camera.frame_width;
		camera_view.height = camera.frame_height;

		if (!grab_gray(camera, camera_view))
		{
			return false;
		}

		grab_cb(camera_view);

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

		img::ViewGray camera_view{};
		camera_view.data = (u8*)camera.cb_image.data_;
		camera_view.width = camera.frame_width;
		camera_view.height = camera.frame_height;

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_gray(img::make_view(device.bgr_image), camera_view);
				grab_cb(camera_view);
			}
		}

		return true;
	}


	bool grab_gray(CameraUSB const& camera, Range2Du32 const& roi, view_gray_callback const& grab_cb)
	{
		assert(verify(camera));

		img::ViewGray camera_view{};
		camera_view.data = (u8*)camera.cb_image.data_;
		camera_view.width = roi.x_end - roi.x_begin;
		camera_view.height = roi.y_end - roi.y_end;

		if (!grab_gray(camera, roi, camera_view))
		{
			return false;
		}

		grab_cb(camera_view);

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

		img::ViewGray camera_view{};
		camera_view.data = (u8*)camera.cb_image.data_;
		camera_view.width = roi.x_end - roi.x_begin;
		camera_view.height = roi.y_end - roi.y_end;

		while (grab_condition())
		{
			if (grab_and_convert_frame_bgr(device))
			{
				map_gray(sub_view(device.bgr_image, roi), camera_view);
				grab_cb(camera_view);
			}
		}

		return true;
	}
}