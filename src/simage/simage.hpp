#pragma once

#include "simage_platform.hpp"
#include "../util/memory_buffer.hpp"

#include <array>
#include <functional>

namespace mb = memory_buffer;


namespace simage
{
	enum class RGB : int
	{
		R = 0, G = 1, B = 2
	};


	enum class RGBA : int
	{
		R = 0, G = 1, B = 2, A = 3
	};


	enum class HSV : int
	{
		H = 0, S = 1, V = 2
	};


	enum class LCH : int
	{
		L = 0, C = 1, H = 2
	};


	enum class YUV : int
	{
		Y = 0, U = 1, V = 2
	};


	enum class GA : int
	{
		G = 0, A = 1
	};


	enum class XY : int
	{
		X = 0, Y = 1
	};


	template <typename T>
	constexpr inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}

}


/* view */

namespace simage
{
	template <typename T, size_t N>
	class ViewCh2D
	{
	public:

		u32 image_width = 0;

		T* channel_data[N] = {};		

		u32 width = 0;
		u32 height = 0;

		union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};
	};


	template <size_t N>
	using ViewCh2Dr32 = ViewCh2D<r32, N>;


	using View1r32 = MatrixView<r32>;

    using View4r32 = ViewCh2Dr32<4>;
	using View3r32 = ViewCh2Dr32<3>;
	using View2r32 = ViewCh2Dr32<2>;


	template <typename T, size_t N>
	class SomeArray
	{
	public:
		T data[N];
	};


	template <size_t N>
	using SomeArrayr32 = SomeArray<r32,N>;
}


/* make_view */

namespace simage
{
	using Buffer32 = MemoryBuffer<r32>;

	using ViewRGBAr32 = View4r32;
	using ViewRGBr32 = View3r32;
	using ViewHSVr32 = View3r32;
	using ViewLCHr32 = View3r32;


	View1r32 make_view_1(u32 width, u32 height, Buffer32& buffer);

	View2r32 make_view_2(u32 width, u32 height, Buffer32& buffer);

	View3r32 make_view_3(u32 width, u32 height, Buffer32& buffer);

	View4r32 make_view_4(u32 width, u32 height, Buffer32& buffer);
}


/* map */

namespace simage
{
	void map(ViewGray const& src, View1r32 const& dst);

	void map(View1r32 const& src, ViewGray const& dst);

	void map(ViewYUV const& src, View1r32 const& dst);
}


/* map_rgb */

namespace simage
{	
	void map_rgba(View const& src, ViewRGBAr32 const& dst);

	void map_rgba(ViewRGBAr32 const& src, View const& dst);
	
	void map_rgb(View const& src, ViewRGBr32 const& dst);

	void map_rgb(ViewRGBr32 const& src, View const& dst);

	void map_rgb(View1r32 const& src, View const& dst);
}


/* map_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVr32 const& dst);

	void map_hsv_rgb(ViewHSVr32 const& src, View const& dst);


	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst);	

	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst);
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHr32 const& dst);

	void map_lch_rgb(ViewLCHr32 const& src, View const& dst);

	void map_rgb_lch(ViewRGBr32 const& src, ViewLCHr32 const& dst);

	void map_lch_rgb(ViewLCHr32 const& src, ViewRGBr32 const& dst);
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst);	
}


/* map_bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBr32 const& dst);
}


/* sub_view */

namespace simage
{
	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range);

	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range);

	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range);

	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	View1r32 select_channel(ViewRGBAr32 const& view, RGBA channel);

	View1r32 select_channel(ViewRGBr32 const& view, RGB channel);

	View1r32 select_channel(ViewHSVr32 const& view, HSV channel);

	View1r32 select_channel(View2r32 const& view, GA channel);

	View1r32 select_channel(View2r32 const& view, XY channel);


	ViewRGBr32 select_rgb(ViewRGBAr32 const& view);
}


/* fill */

namespace simage
{
	void fill(View4r32 const& view, Pixel color);

	void fill(View3r32 const& view, Pixel color);

	void fill(View1r32 const& view, u8 gray);
}


/* transform */

namespace simage
{
	void transform(View1r32 const& src, View1r32 const& dst, std::function<r32(r32)> const& func);

	void transform(View2r32 const& src, View1r32 const& dst, std::function<r32(r32, r32)> const& func);

	void transform(View3r32 const& src, View1r32 const& dst, std::function<r32(r32, r32, r32)> const& func);

	
	inline void transform_gray(ViewRGBr32 const& src, View1r32 const& dst)
	{
		return transform(src, dst, [](r32 red, r32 green, r32 blue) { return 0.299f * red + 0.587f * green + 0.114f * blue; });
	}
}


/* shrink */

namespace simage
{
	void shrink(View1r32 const& src, View1r32 const& dst);

	void shrink(View3r32 const& src, View3r32 const& dst);

	void shrink(ViewGray const& src, View1r32 const& dst);

	void shrink(View const& src, ViewRGBr32 const& dst);
}


/* gradients */

namespace simage
{
	void gradients_xy(View1r32 const& src, View2r32 const& xy_dst);
}


/* blur */

namespace simage
{
	void blur(View1r32 const& src, View1r32 const& dst);

	void blur(View3r32 const& src, View3r32 const& dst);
}

#ifdef SIMAGE_CUDA

#include "../cuda/device.hpp"


/* view */

namespace simage
{
    template <typename T, size_t N>
	class CudaViewCh2D
	{
	public:

		u32 image_width = 0;

		T* channel_data[N] = {};		

		u32 width = 0;
		u32 height = 0;

		union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};
	};


    template <typename T>
    class CudaView
    {
    public:
        u32 image_width = 0;

        T* image_data = nullptr;

        u32 width = 0;
		u32 height = 0;

		union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};
    };


    using CudaView4r32 = CudaViewCh2D<r32, 4>;
	using CudaView3r32 = CudaViewCh2D<r32, 3>;
	using CudaView2r32 = CudaViewCh2D<r32, 2>;
    using CudaView1r32 = CudaView<r32>;

    using CudaView4r16 = CudaViewCh2D<cuda::r16, 4>;
	using CudaView3r16 = CudaViewCh2D<cuda::r16, 3>;
	using CudaView2r16 = CudaViewCh2D<cuda::r16, 2>;
    using CudaView1r16 = CudaView<cuda::r16>;

    using CudaViewRGBr32 = CudaView3r32;
    using CudaViewRGBAr32 = CudaView4r32;
}


namespace simage
{
    
}


#endif // SIMAGE_CUDA