#include <Windows.h>
#include <mfapi.h>
#include <mfidl.h>
#include <Mfreadwrite.h>
#include <Shlwapi.h>
#include <vector>

#pragma comment(lib, "mf.lib")
#pragma comment(lib, "mfplat.lib")
#pragma comment(lib, "mfreadwrite.lib")
#pragma comment(lib, "mfuuid.lib")
#pragma comment(lib, "Shlwapi.lib")


/* types */

namespace w32
{
    using Device_p = IMFActivate*;
    using MediaSource_p = IMFMediaSource*;
    using SourceReader_p = IMFSourceReader*;
    using Buffer_p = IMFMediaBuffer*;
    using Sample_p = IMFSample*;


    class Frame
    {
    public:
        DWORD stream_index = 0;
        DWORD flags = 0;
        LONGLONG timestamp = 0;

        Buffer_p buffer = nullptr;

        BYTE* data = nullptr;
        DWORD size_bytes = 0;

        bool is_locked = false;
    };


    class FrameFormat
    {
    public:
        UINT32 width = 0;
        UINT32 height = 0;
        UINT32 stride = 0;
        UINT32 fps = 0;
        UINT32 pixel_size = 0;
    };


    template <typename T>
    class DataResult
    {
    public:
        T data;
        bool success = false;
    };


    union Bytes8
    {
        UINT64 val64 = 0;

        struct
        {
            UINT32 lo32;
            UINT32 hi32;
        };
    };


    template <class T>
    static void release(T*& ptr)
    {
        if (ptr)
        {
            ptr->Release();
            ptr = nullptr;
        }
    }
}


namespace w32
{
    static bool init()
    {
        HRESULT hr = MFStartup(MF_VERSION);

        return hr == S_OK;
    }


    static void shutdown()
    {
        MFShutdown();
    }
}


/* read_frame */

namespace w32
{
    static DataResult<Frame> read_frame(SourceReader_p reader, Sample_p& sample)
    {
        DataResult<Frame> result{};
        auto& frame = result.data;

        HRESULT hr = reader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 
            0, 
            &frame.stream_index, 
            &frame.flags, 
            &frame.timestamp, 
            &sample);

        if (FAILED(hr))
        {
            return result;
        }

        hr = sample->ConvertToContiguousBuffer(&frame.buffer);
        if (FAILED(hr))
        {
            return result;
        }

        hr = frame.buffer->Lock(&frame.data, NULL, &frame.size_bytes);
        if (FAILED(hr))
        {
            release(frame.buffer);
            return result;
        }

        frame.is_locked = true;

        result.success = true;
        return result;
    }


    static void release(Frame& frame)
    {
        if (frame.is_locked)
        {
            frame.buffer->Unlock();
            frame.is_locked = false;
        }

        release(frame.buffer);
    }
}


/* ? */

namespace w32
{
    static bool activate(Device_p device, MediaSource_p& source, SourceReader_p& reader)
    {
        HRESULT hr = device->ActivateObject(__uuidof(IMFMediaSource), (void**)&source);
        if (FAILED(hr))
        {
            return false;
        }

        hr = MFCreateSourceReaderFromMediaSource(source, NULL, &reader);
        if (FAILED(hr))
        {
            release(source);
            return false;
        }

        hr = reader->SetStreamSelection(MF_SOURCE_READER_FIRST_VIDEO_STREAM, TRUE);
        if (FAILED(hr))
        {
            release(source);
            release(reader);
            return false;
        }

        Frame frame{};
        Sample_p sample = nullptr;

        hr = reader->ReadSample(
            (DWORD)MF_SOURCE_READER_FIRST_VIDEO_STREAM, 
            0, 
            &frame.stream_index, 
            &frame.flags, 
            &frame.timestamp, 
            &sample);

        if (FAILED(hr))
        {
            release(source);
            release(reader);
            return false;
        }

        return true;
    }


    static DataResult<FrameFormat> get_frame_format(SourceReader_p reader)
    {
        DataResult<FrameFormat> result{};
        auto& format = result.data;

        IMFMediaType* media_type = nullptr;

        HRESULT hr = reader->GetCurrentMediaType(MF_SOURCE_READER_FIRST_VIDEO_STREAM, &media_type);
        if (FAILED(hr))
        {
            return result;
        }

        GUID major_type;
        hr = media_type->GetGUID(MF_MT_MAJOR_TYPE, &major_type);
        if (FAILED(hr) || major_type != MFMediaType_Video)
        {
            release(media_type);
            return result;
        }

        hr = MFGetAttributeSize(media_type, MF_MT_FRAME_SIZE, &format.width, &format.height);
        if (FAILED(hr) || !format.width || ! format.height)
        {
            release(media_type);
            return result;
        }

        hr = media_type->GetUINT32(MF_MT_DEFAULT_STRIDE, &format.stride);
        if (FAILED(hr) || !format.stride)
        {
            release(media_type);
            return result;
        }

        Bytes8 fps;

        hr = media_type->GetUINT64(MF_MT_FRAME_RATE, &fps.val64);

        if (FAILED(hr) || !fps.hi32 || !fps.lo32)
        {
            release(media_type);
            return result;
        }

        format.fps = fps.hi32 / fps.lo32;
        format.pixel_size = format.stride / format.width;

        release(media_type);
        result.success = true;
        return result;
    } 
}