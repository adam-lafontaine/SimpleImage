/* row begin */

namespace simage
{
	template <typename T>
	static inline T* row_begin(Matrix2D<T> const& image, u32 y)
	{
		return image.data_ + (u64)(y * image.width);
	}


	template <typename T>
	static inline T* row_begin(MatrixView<T> const& view, u32 y)
	{
		return view.matrix_data_ + (u64)((view.y_begin + y) * view.matrix_width + view.x_begin);
	}


	template <typename T>
	static inline T* xy_at(MatrixView<T> const& view, u32 x, u32 y)
	{
		return row_begin(view, y) + x;
	}


    template <typename T>
	static T* row_offset_begin(MatrixView<T> const& view, u32 y, int y_offset)
	{
		assert(verify(view));

		int y_eff = y + y_offset;

		auto offset = (view.y_begin + y_eff) * view.matrix_width + view.x_begin;

		auto ptr = view.matrix_data_ + (u64)(offset);
		assert(ptr);

		return ptr;
	}


    template <typename T, size_t N>
	static inline u64 row_offset(ChannelView<T, N> const& view, u32 y)
	{
		return ((u64)view.y_begin + y) * view.channel_width_ + view.x_begin;
	}


	static RGBf32p rgb_row_begin(ViewRGBf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		RGBf32p rgb{};

		rgb.R = view.channel_data_[id_cast(RGB::R)] + offset;
		rgb.G = view.channel_data_[id_cast(RGB::G)] + offset;
		rgb.B = view.channel_data_[id_cast(RGB::B)] + offset;

		return rgb;
	}


	static RGBAf32p rgba_row_begin(ViewRGBAf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		RGBAf32p rgba{};

		rgba.R = view.channel_data_[id_cast(RGBA::R)] + offset;
		rgba.G = view.channel_data_[id_cast(RGBA::G)] + offset;
		rgba.B = view.channel_data_[id_cast(RGBA::B)] + offset;
		rgba.A = view.channel_data_[id_cast(RGBA::A)] + offset;

		return rgba;
	}


	static HSVf32p hsv_row_begin(ViewHSVf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		HSVf32p hsv{};

		hsv.H = view.channel_data_[id_cast(HSV::H)] + offset;
		hsv.S = view.channel_data_[id_cast(HSV::S)] + offset;
		hsv.V = view.channel_data_[id_cast(HSV::V)] + offset;

		return hsv;
	}


	static YUVf32p yuv_row_begin(ViewYUVf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		YUVf32p yuv{};

		yuv.Y = view.channel_data_[id_cast(YUV::Y)] + offset;
		yuv.U = view.channel_data_[id_cast(YUV::U)] + offset;
		yuv.V = view.channel_data_[id_cast(YUV::V)] + offset;

		return yuv;
	}


	static LCHf32p lch_row_begin(ViewLCHf32 const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		LCHf32p lch{};

		lch.L = view.channel_data_[id_cast(LCH::L)] + offset;
		lch.C = view.channel_data_[id_cast(LCH::C)] + offset;
		lch.H = view.channel_data_[id_cast(LCH::H)] + offset;

		return lch;
	}


	template <typename T, size_t N>
	static std::array<T*, N> view_row_begin(ChannelView<T, N> const& view, u32 y)
	{
		assert(verify(view));
		assert(y < view.height);

		auto offset = row_offset(view, y);

		std::array<T*, N> rows = { 0 };

		for (u32 ch = 0; ch < N; ++ch)
		{
			rows[ch] = view.channel_data_[ch] + offset;
		}

		return rows;
	}


	template <typename T, size_t N>
	static T* channel_row_begin(ChannelView<T, N> const& view, u32 y, u32 ch)
	{
		assert(verify(view));

		assert(y < view.height);

		auto offset = row_offset(view, y);

		return view.channel_data_[ch] + offset;
	}
}