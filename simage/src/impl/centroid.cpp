/* centroid */

namespace simage
{
	Point2Du32 centroid(ViewGray const& src)
	{
		PROFILE_BLOCK(PL::CentroidView)
		
		f64 total = 0.0;
		f64 x_total = 0.0;
		f64 y_total = 0.0;

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u64 val = s[x] ? 1 : 0;
				total += val;
				x_total += x * val;
				y_total += y * val;
			}
		}

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = (u32)(x_total / total);
			pt.y = (u32)(y_total / total);
		}

		return pt;
	}


	Point2Du32 centroid(ViewGray const& src, u8_to_bool_f const& func)
	{
		PROFILE_BLOCK(PL::CentroidViewGray)

		f64 total = 0.0;
		f64 x_total = 0.0;
		f64 y_total = 0.0;

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; ++x)
			{
				u64 val = func(s[x]) ? 1 : 0;
				total += val;
				x_total += x * val;
				y_total += y * val;
			}
		}

		Point2Du32 pt{};

		if (total == 0)
		{
			pt.x = src.width / 2;
			pt.y = src.height / 2;
		}
		else
		{
			pt.x = (u32)(x_total / total);
			pt.y = (u32)(y_total / total);
		}

		return pt;
	}


	
}
