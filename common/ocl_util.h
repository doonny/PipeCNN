// OpenCL Utility Functions
#ifndef _OCL_UTIL_H
#define _OCL_UTIL_H

#include <stdlib.h>
#include <string.h>
#include <string>

#include <algorithm>
#include <stdarg.h>

#ifdef _WIN32 // Windows
#include <windows.h>
#else         // Linux
#include <stdio.h>
#include <unistd.h>
#endif

#include "CL/opencl.h"

namespace ocl_util {

// Debuging and Error functions
void printError(cl_int error);
void _checkError(int line, 
				const char *file, 
				cl_int error, 
				const char *msg, ...); // does not return
#define checkError(status, ...) _checkError(__LINE__, __FILE__, status, __VA_ARGS__)

// Smart pointers.
// scoped_array: assumes pointer was allocated with operator new[]; destroys with operator delete[]
// Also supports allocation/reset with a number, which is the number of
// elements of type T.
template<typename T>
class scoped_array {
public:
  typedef scoped_array<T> this_type;

  scoped_array() : m_ptr(NULL) {}
  scoped_array(T *ptr) : m_ptr(NULL) { reset(ptr); }
  explicit scoped_array(size_t n) : m_ptr(NULL) { reset(n); }
  ~scoped_array() { reset(); }

  T *get() const { return m_ptr; }
  operator T *() const { return m_ptr; }
  T *operator ->() const { return m_ptr; }
  T &operator *() const { return *m_ptr; }
  T &operator [](int index) const { return m_ptr[index]; }

  this_type &operator =(T *ptr) { reset(ptr); return *this; }

  void reset(T *ptr = NULL) { delete[] m_ptr; m_ptr = ptr; }
  void reset(size_t n) { reset(new T[n]); }
  T *release() { T *ptr = m_ptr; m_ptr = NULL; return ptr; }

private:
  T *m_ptr;

  // noncopyable
  scoped_array(const this_type &);
  this_type &operator =(const this_type &);
};

// Find a platform that contains the search string in its name (case-insensitive match)
// Returns NULL if no match is found
cl_platform_id findPlatform(const char *platform_name_search);

// Returns the name of the platform
std::string getPlatformName(cl_platform_id pid);

// Returns the name of the device.
std::string getDeviceName(cl_device_id did);

// Returns an array of device ids for the given platform and the
// device type.
// Return value must be freed with delete[].
cl_device_id *getDevices(cl_platform_id pid, cl_device_type dev_type, cl_uint *num_devices);

// Display all the device informations
void displayDeviceInfo(cl_device_id did);

// Load the opencl source or binary files
size_t load_file_to_memory(const char *filename, char **result); 

// Create a OpenCL program from a source or binary file.
// The program is created for all given devices associated with the context. The same
// binary is used for all devices.
cl_program createProgramFromFile(cl_context context, const char *kernelbinary, const cl_device_id *devices, unsigned num_devices);

// Funtion to generate random floating-point numbers
float rand_float();

// Memory operations for DMA
void *alignedMalloc(size_t size, size_t alignment);
void alignedFree(void * ptr);

} // namespace ocl_utils

#endif

