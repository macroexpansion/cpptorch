#ifndef NOCONTOUR_HPP
#define NOCONTOUR_HPP

#include <exception>

struct NoContourException : public std::exception {
    const char * what () const throw () {
        return "No contour found";
    }
};

#endif