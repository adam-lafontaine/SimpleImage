#include "../../simage.hpp"
#include "../../src/util/color_space.hpp"

#include <Windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <Mfreadwrite.h>
#include <Shlwapi.h>
#include <vector>

#ifndef NDEBUG
#include <cstdio>
#endif

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "Shlwapi.lib")

namespace img = simage;


class DeviceW32
{
public:
    IMFActivate* p_device = nullptr;
    int device_id = -1;

    IMFMediaSource* p_source = nullptr;
    IMFSourceReader* p_reader = nullptr;
    IMFSample* p_sample = nullptr;
    
    int frame_width = -1;
    int frame_height = -1;

    bool is_connected = false;
};


class DeviceListW32
{
public:
    IMFActivate** device_list = nullptr;

    std::vector<DeviceW32> devices;

    bool is_connected = false;
};


static DeviceListW32 g_device_list;


template <class T>
static void imf_release(T*& ptr)
{
    if (ptr)
    {
        ptr->Release();
        ptr = nullptr;
    }
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


static void disconnect_device(DeviceW32& device)
{
    if (!device.is_connected)
    {
        return;
    }

    imf_release(device.p_source);
    imf_release(device.p_reader);
    imf_release(device.p_sample);

    device.frame_width = -1;
    device.frame_height = -1;
    device.is_connected = false;
}