#pragma once

#include <cstddef>
#include <string>

namespace tools
{
class MappedFile
{
   public:
	MappedFile(const std::string& fileName);
	~MappedFile();

	const char* begin() const;
	const char* end() const;

   private:
	bool open();
	void close();

   private:
	int handle_;
	size_t size_;
	char* mapping_ = nullptr;
	std::string fileName_;
};
}  // namespace tools