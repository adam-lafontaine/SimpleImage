#include "../../simage.hpp"
#include "../../src/util/color_space.hpp"
#include "w32.h"

#ifndef NDEBUG
#include <cstdio>
#endif


namespace img = simage;


class DeviceW32
{
public:
    w32::Device_p p_device = nullptr;    

    w32::MediaSource_p p_source = nullptr;
    w32::SourceReader_p p_reader = nullptr;

    w32::Sample_p p_sample = nullptr;

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


static void disconnect_device(DeviceW32& device)
{
    if (!device.is_connected)
    {
        return;
    }

    w32::release(device.p_source);
    w32::release(device.p_reader);
    w32::release(device.p_sample);

    device.frame_width = -1;
    device.frame_height = -1;
    device.is_connected = false;
}