#include "../../simage.hpp"
#include "../../src/util/color_space.hpp"
#include "w32.h"

#ifndef NDEBUG
#include <cstdio>
#endif


namespace img = simage;


typedef bool(convert_rgba_callback_t)(w32::Frame& frame, img::View const& dst);
typedef bool(convert_gray_callback_t)(w32::Frame& frame, img::ViewGray const& dst);


namespace convert
{
    static bool rgba_error(w32::Frame& frame, img::View const& dst)
    {
        return false;
    }


    static bool gray_error(w32::Frame& frame, img::ViewGray const& dst)
    {
        return false;
    }


    static bool yuyv_to_rgba(w32::Frame& frame, img::View const& dst)
    {
        assert((size_t)frame.size_bytes == sizeof(img::YUV2u8) * dst.width * dst.height);

        img::ImageYUV image;
        image.width = dst.width;
        image.height = dst.height;
        image.data_ = (img::YUV2u8*)frame.data;

        auto src = img::make_view(image);

        img::map_yuv_rgba(src, dst);

        return true;
    }


    static bool yuyv_to_gray(w32::Frame& frame, img::ViewGray const& dst)
    {
        assert((size_t)frame.size_bytes == sizeof(img::YUV2u8) * dst.width * dst.height);

        img::ImageYUV image;
        image.width = dst.width;
        image.height = dst.height;
        image.data_ = (img::YUV2u8*)frame.data;

        auto src = img::make_view(image);

        img::map_gray(src, dst);

        return true;
    }


    static bool uyuv_to_rgba(w32::Frame& frame, img::View const& dst)
    {
        assert((size_t)frame.size_bytes == sizeof(img::UVY2u8) * dst.width * dst.height);

        img::ImageUVY image;
        image.width = dst.width;
        image.height = dst.height;
        image.data_ = (img::UVY2u8*)frame.data;

        auto src = img::make_view(image);

        img::map_yuv_rgba(src, dst);

        return true;
    }


    static bool uyuv_to_gray(w32::Frame& frame, img::ViewGray const& dst)
    {
        assert((size_t)frame.size_bytes == sizeof(img::UVY2u8) * dst.width * dst.height);

        img::ImageUVY image;
        image.width = dst.width;
        image.height = dst.height;
        image.data_ = (img::UVY2u8*)frame.data;

        auto src = img::make_view(image);

        img::map_gray(src, dst);

        return true;
    }


    static bool rgb_to_rgba(w32::Frame& frame, img::View const& dst)
    {
        assert((size_t)frame.size_bytes == sizeof(img::RGBu8) * dst.width * dst.height);

        img::ImageRGB image;
        image.width = dst.width;
        image.height = dst.height;
        image.data_ = (img::RGBu8*)frame.data;

        auto src = img::make_view(image);

        img::map_rgba(src, dst);

        return true;
    }


    static bool rgb_to_gray(w32::Frame& frame, img::ViewGray const& dst)
    {
        assert((size_t)frame.size_bytes == sizeof(img::RGBu8) * dst.width * dst.height);

        img::ImageRGB image;
        image.width = dst.width;
        image.height = dst.height;
        image.data_ = (img::RGBu8*)frame.data;

        auto src = img::make_view(image);

        img::map_gray(src, dst);

        return true;
    }
}


class DeviceW32
{
public:
    w32::Device_p p_device = nullptr;    

    w32::MediaSource_p p_source = nullptr;
    w32::SourceReader_p p_reader = nullptr;

    w32::Sample_p p_sample = nullptr;

    convert_rgba_callback_t* convert_rgba = convert::rgba_error;
    convert_gray_callback_t* convert_gray = convert::gray_error;

    img::Image rgba_frame;
    
    img::View rgba_view;
	img::ViewGray gray_view;

    int device_id = -1;
    
    int frame_width = -1;
    int frame_height = -1;
    int frame_stride = -1;
    int fps = -1;

    bool is_connected = false;
};


class DeviceListW32
{
public:
    w32::Device_p* device_list = nullptr;

    std::vector<DeviceW32> devices;

    bool is_connected = false;
};


static DeviceListW32 g_device_list;


static void print_error(const char* msg)
{
#ifndef NDEBUG

    printf("%s\n", msg);

#endif
}


static void print_device_list(DeviceListW32 const& list)
{
#ifndef NDEBUG

    WCHAR buffer[250] = { 0 };
    auto dst = (LPWSTR)buffer;
    u32 len = 0;
    HRESULT hr = S_OK;

    for (auto& device : list.devices)
    {
        memset(buffer, 0, sizeof(buffer));

        hr = device.p_device->GetAllocatedString(MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME, &dst, &len);
        wprintf(L"%s\n", dst);
    }

#endif
}


static void set_frame_formats(DeviceW32& device, w32::PixelFormat pixel_format)
{
    using PF = w32::PixelFormat;

    switch(pixel_format)
    {
        case PF::RGB24:
            device.convert_rgba = convert::rgb_to_rgba;
            device.convert_gray = convert::rgb_to_gray;
            break;
        case PF::YUYV:
            device.convert_rgba = convert::yuyv_to_rgba;
            device.convert_gray = convert::yuyv_to_gray;
            break;

        case PF::UYVY:
            device.convert_rgba = convert::uyuv_to_rgba;
            device.convert_gray = convert::uyuv_to_gray;
            break;        

        default: return;
    }
}


static void disconnect_device(DeviceW32& device)
{
    if (!device.is_connected)
    {
        return;
    }

    w32::release(device.p_source);
    w32::release(device.p_reader);
    //w32::release(device.p_sample);

    device.frame_width = -1;
    device.frame_height = -1;
    device.frame_stride = -1;
    device.fps = -1;

    device.convert_rgba = convert::rgba_error;
    device.convert_gray = convert::gray_error;
        
    device.is_connected = false;
}


static bool connect_device(DeviceW32& device)
{
    if (!w32::activate(device.p_device, device.p_source, device.p_reader))
    {
        print_error("Error w32::activate()");
        return false;
    }

    auto result = w32::get_frame_format(device.p_reader);
    if (!result.success)
    {
        print_error("Error w32::get_frame_format()");
        return false;
    }

    auto& format = result.data;

    device.frame_width = (int)format.width;
    device.frame_height = (int)format.height;
    device.frame_stride = (int)format.stride;
    device.fps = (int)format.fps;

    set_frame_formats(device, format.pixel_format);

    device.is_connected = true;

    return true;
}


static DeviceW32* get_default_device(DeviceListW32& list)
{
    auto& devices = list.devices;
    if (devices.empty())
    {
        return nullptr;
    }

    // return camera with highest index that will connect
    for (int id = (int)devices.size() - 1; id >= 0; --id)
    {
        if (connect_device(devices[id]))
        {
            return devices.data() + id;
        }
    }

    return nullptr;
}


static bool enumerate_devices(DeviceListW32& list)
{
    if (!w32::init())
    {
        return false;
    }

    HRESULT hr = S_OK;

    IMFAttributes* p_attr = nullptr;

    hr = MFCreateAttributes(&p_attr, 1);
    if (FAILED(hr))
    {
        return false;
    }

    hr = p_attr->SetGUID(
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE,
        MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE_VIDCAP_GUID
    );

    if (FAILED(hr))
    {
        w32::release(p_attr);
        return false;
    }

    u32 n_devices = 0;

    hr = MFEnumDeviceSources(p_attr, &list.device_list, &n_devices);
    w32::release(p_attr);

    if (FAILED(hr) || !n_devices)
    {
        return false;
    }

    for (u32 i = 0; i < n_devices; ++i)
    {
        DeviceW32 device;

        device.p_device = list.device_list[i];
        device.device_id = i;

        list.devices.push_back(std::move(device));
    }

    return true;
}


static void close_devices(DeviceListW32& list)
{
    for (auto& device : list.devices)
    {
        img::destroy_image(device.rgba_frame);
        disconnect_device(device);
        w32::release(device.p_device);
    }

    CoTaskMemFree(list.device_list);
    w32::shutdown();
}


static bool grab_and_convert_frame_rgba(DeviceW32& device)
{
    auto result = w32::read_frame(device.p_reader, device.p_sample);
    if (!result.success)
    {
        print_error("Error: w32::read_frame()");
        return false;
    }

    auto& frame = result.data;

    if (!device.convert_rgba(frame, device.rgba_view))
    {
        print_error("Error: convert_rgba()");
        return false;
    }

    w32::release(frame);

    return true;
}


static bool grab_and_convert_frame_gray(DeviceW32& device)
{
    auto result = w32::read_frame(device.p_reader, device.p_sample);
    if (!result.success)
    {
        print_error("Error: w32::read_frame()");
        return false;
    }

    auto& frame = result.data;

    if (!device.convert_gray(frame, device.gray_view))
    {
        print_error("Error: convert_gray()");
        return false;
    }

    w32::release(frame);

    return true;
}


template <class ViewSRC, class ViewDST>
static void write_frame_sub_view_rgba(img::CameraUSB const& camera, ViewSRC const& src, ViewDST const& dst)
{
	auto width = std::min(camera.frame_width, dst.width);
	auto height = std::min(camera.frame_height, dst.height);
	auto r = make_range(width, height);

	img::copy(img::sub_view(src, r), img::sub_view(dst, r));
}


template <class ViewSRC, class ViewDST>
static void write_frame_sub_view_gray(img::CameraUSB const& camera, ViewSRC const& src, ViewDST const& dst)
{
	auto width = std::min(camera.frame_width, dst.width);
	auto height = std::min(camera.frame_height, dst.height);
	auto r = make_range(width, height);

	img::copy(img::sub_view(src, r), img::sub_view(dst, r));
}


static bool camera_is_initialized(img::CameraUSB const& camera)
{
    return camera.is_open        
        && camera.device_id >= 0
        && camera.device_id < (int)g_device_list.devices.size();
}


namespace simage
{
    bool open_camera(CameraUSB& camera)
    {
        if (!enumerate_devices(g_device_list))
        {
            print_error("Error enumerate_devices()");
            return false;
        }

        print_device_list(g_device_list);

        auto result = get_default_device(g_device_list);
        if (!result)
        {
            print_error("No connected devices available");
            return false;
        }        

        auto& device = *result;

        auto const fail = [&]()
        {
            close_devices(g_device_list);
            destroy_image(device.rgba_frame);
            return false;
        };

        auto width = device.frame_width;
        auto height = device.frame_height;

        camera.device_id = device.device_id;
        camera.max_fps = device.fps;
        camera.frame_width = width;
        camera.frame_height = height;  

        if (!create_image(device.rgba_frame, width, height))
        {
            return fail();
        }

        device.rgba_view = make_view(device.rgba_frame);

        img::ImageGray gray_frame;
        gray_frame.width = width;
        gray_frame.height = height;
        gray_frame.data_ = (u8*)device.rgba_frame.data_;

        device.gray_view = img::make_view(gray_frame);

        camera.is_open = true;

        return true;
    }


    void close_camera(CameraUSB& camera)
    {
        camera.device_id = -1;
        camera.max_fps = -1;
        camera.frame_width = -1;
        camera.frame_height = -1;
        
        camera.is_open = false;

        close_devices(g_device_list);
    }


    bool grab_rgb(CameraUSB const& camera, View const& dst)
    {
        assert(verify(camera));
        
        if (!camera_is_initialized(camera))
        {
            return false;
        }
        
        auto& device = g_device_list.devices[camera.device_id];

        if (!grab_and_convert_frame_rgba(device))
        {
            return false;
        }

        if (camera.frame_width == dst.width && camera.frame_height == dst.height)
		{
			copy(device.rgba_view, dst);
		}
		else
		{
			write_frame_sub_view_rgba(camera, device.rgba_view, dst);
		}

        return true;
    }


    bool grab_rgb(CameraUSB const& camera, Range2Du32 const& roi, View const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_and_convert_frame_rgba(device))
		{
			return false;
		}

		write_frame_sub_view_rgba(camera, sub_view(device.rgba_view, roi), dst);

		return true;
	}


	bool grab_rgb(CameraUSB const& camera, view_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_rgb(camera, device.rgba_view))
		{
			return false;
		}

		grab_cb(device.rgba_view);

		return true;
	}


    bool grab_rgb_continuous(CameraUSB const& camera, view_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_device_list.devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_rgba(device))
			{
				grab_cb(device.rgba_view);
			}
		}

		return true;
	}


    bool grab_rgb(CameraUSB const& camera, Range2Du32 const& roi, view_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_rgb(camera, roi, device.rgba_view))
		{
			return false;
		}

		grab_cb(device.rgba_view);

		return true;
	}


    bool grab_rgb_continuous(CameraUSB const& camera, Range2Du32 const& roi, view_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_device_list.devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_rgba(device))
			{
				grab_cb(device.rgba_view);
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

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_and_convert_frame_gray(device))
		{
			return false;
		}

		if (camera.frame_width == dst.width && camera.frame_height == dst.height)
		{
			copy(device.gray_view, dst);
		}
		else
		{
			write_frame_sub_view_gray(camera, device.gray_view, dst);
		}

		return true;
	}


    bool grab_gray(CameraUSB const& camera, Range2Du32 const& roi, ViewGray const& dst)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_and_convert_frame_gray(device))
		{
			return false;
		}

		write_frame_sub_view_gray(camera, sub_view(device.gray_view, roi), dst);

		return true;
	}


    bool grab_gray(CameraUSB const& camera, view_gray_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_gray(camera, device.gray_view))
		{
			return false;
		}

		grab_cb(device.gray_view);

		return true;
	}


    bool grab_gray_continuous(CameraUSB const& camera, view_gray_callback const& grab_cb, bool_f const& grab_condition)
	{
		assert(verify(camera));

		if (!camera_is_initialized(camera))
		{
			return false;
		}

		auto& device = g_device_list.devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_gray(device))
			{
				grab_cb(device.gray_view);
			}
		}

		return true;
	}


    bool grab_gray(CameraUSB const& camera, Range2Du32 const& roi, view_gray_callback const& grab_cb)
	{
		assert(verify(camera));

		auto& device = g_device_list.devices[camera.device_id];

		if (!grab_gray(camera, roi, device.gray_view))
		{
			return false;
		}

		grab_cb(device.gray_view);

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

		auto& device = g_device_list.devices[camera.device_id];

		while (grab_condition())
		{
			if (grab_and_convert_frame_gray(device))
			{
				grab_cb(device.gray_view);
			}
		}

		return true;
	}
}