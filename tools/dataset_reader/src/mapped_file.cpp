#include "dataset_reader/mapped_file.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include <stdexcept>

namespace tools
{
MappedFile::MappedFile(const std::string& fileName)
	: handle_(-1), size_(0), mapping_(nullptr), fileName_(fileName)
{
	if (!open())
	{
		throw std::runtime_error("Failed to read file with file name:" +
								 fileName_);
	}
}

MappedFile::~MappedFile()
{
	close();
}

const char* MappedFile::begin() const
{
	return mapping_;
}
const char* MappedFile::end() const
{
	return mapping_ + size_;
}

bool MappedFile::open()
{
	close();

	int h = ::open(fileName_.data(), O_RDONLY);
	if (h < 0)
	{
		return false;
	}

	lseek(h, 0, SEEK_END);
	size_ = lseek(h, 0, SEEK_CUR);

	auto m = mmap(nullptr, size_, PROT_READ, MAP_SHARED, h, 0);
	if (m == MAP_FAILED)
	{
		::close(h);
		return false;
	}

	handle_ = h;
	mapping_ = static_cast<char*>(m);
	return true;
}

void MappedFile::close()
{
	if (handle_ >= 0)
	{
		munmap(mapping_, size_);
		::close(handle_);
		handle_ = -1;
	}
}
}  // namespace tools
