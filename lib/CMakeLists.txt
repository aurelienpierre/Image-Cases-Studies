project("ImageUtils")

set(CMAKE_CXX_STANDARD 11)  # enable C++11 standard
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++1y -Wall")
add_library (ImageUtils MODULE ${SOURCE_FILES})