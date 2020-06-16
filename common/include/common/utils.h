#pragma once

namespace common
{
inline bool isnan(double value)
{
	return value != value;
}
}  // ns common