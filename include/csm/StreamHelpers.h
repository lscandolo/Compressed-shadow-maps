#pragma once

#include "Sampler.h"

inline std::istream & operator>>(std::istream & str, SamplerConfig::SamplerType& s) {
	unsigned int type = 0;
	if (str >> type)
		s = static_cast<SamplerConfig::SamplerType>(s);
	return str;
}


inline std::istream & operator>>(std::istream & str, SamplerConfig& c) {
	
	str >> c.type;
	str >> c.sampleQty;
	str >> c.sampleResolution;
	str >> c.tileResolution;

	return str;
}

inline std::ostream & operator<<(std::ostream & str, SamplerConfig& c) {
	
	str << c.type;
	str << c.sampleQty;
	str << c.sampleResolution;
	str << c.tileResolution;

	return str;
}

static
inline std::wstring s2ws(const std::string& str)
{
	typedef std::codecvt_utf8<wchar_t> convert_typeX;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.from_bytes(str);
}

static
inline std::string ws2s(const std::wstring& wstr)
{
	typedef std::codecvt_utf8<wchar_t> convert_typeX;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.to_bytes(wstr);
}