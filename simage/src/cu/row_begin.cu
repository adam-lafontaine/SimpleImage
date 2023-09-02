/* row_begin */

namespace gpuf
{
    template <typename T>
    GPU_FUNCTION
    inline T* row_begin(DeviceMatrix2D<T> const& view, u32 y)
    {
        return view.data + (u64)(y * view.width);
    }


    template <typename T>
    GPU_FUNCTION
	inline T* xy_at(DeviceMatrix2D<T> const& view, u32 x, u32 y)
    {
        return gpuf::row_begin(view, y) + x;
    }


    template <typename T>
    GPU_FUNCTION
	inline T* xy_at(DeviceMatrix2D<T> const& view, Point2Du32 const& pt)
    {
        return gpuf::row_begin(view, pt.y) + pt.x;
    }


    GPU_FUNCTION
    inline u8* channel_xy_at(DeviceView const& view, u32 x, u32 y, u32 ch)
    {
        auto p = gpuf::xy_at(view, x, y);
        return p->channels + ch;
    }
}