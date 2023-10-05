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


namespace w32
{
    template <class T>
    static void release(T*& ptr)
    {
        if (ptr)
        {
            ptr->Release();
            ptr = nullptr;
        }
    }


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


    class FrameResult
    {
    public:
        Frame frame;
        bool success = false;
    };


    static FrameResult read_frame(SourceReader_p reader, Sample_p& sample)
    {
        FrameResult result{};
        auto& frame = result.frame;

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
        }

        release(frame.buffer);
    }
}


