#ifndef MATRIXVIEWHPP
#define MATRIXVIEWHPP

#include <cassert>

/*
* This class gives a 2D-matrix view on a buffer
* The 2D-matrix is row major, ie its rows are 
* stored contiguously in the buffer
*/
template<typename T>
class matrixView{
private:
    T* buf_ = nullptr;
    size_t rows_ = 0;
    size_t cols_ = 0;

public:
    matrixView() = default;

    matrixView(T* buf, size_t rows, size_t cols):buf_(buf),rows_(rows),cols_(cols){}

    T*     data() { return buf_; }
    const T* data() const { return buf_; }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    size_t elems() const {return rows_ * cols_; }

    T& operator() (size_t i, size_t j) { 
        assert(i < rows_ && j < cols_);

        return buf_[cols_*i + j]; 
    }

    const T& operator() (size_t i, size_t j) const {
        assert(i < rows_ && j < cols_);
        return buf_[cols_ * i + j];
    }

};

#endif