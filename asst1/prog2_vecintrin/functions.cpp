#include <stdio.h>
#include <algorithm>
#include <math.h>
#include "CMU418intrin.h"
#include "logger.h"
using namespace std;


void absSerial(float* values, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	if (x < 0) {
	    output[i] = -x;
	} else {
	    output[i] = x;
	}
    }
}

// implementation of absolute value using 15418 instrinsics
void absVector(float* values, float* output, int N) {
    __cmu418_vec_float x;
    __cmu418_vec_float result;
    __cmu418_vec_float zero = _cmu418_vset_float(0.f);
    __cmu418_mask maskAll, maskIsNegative, maskIsNotNegative;

    //  Note: Take a careful look at this loop indexing.  This example
    //  code is not guaranteed to work when (N % VECTOR_WIDTH) != 0.
    //  Why is that the case?
    for (int i=0; i<N; i+=VECTOR_WIDTH) {

	// All ones
	maskAll = _cmu418_init_ones();

	// All zeros
	maskIsNegative = _cmu418_init_ones(0);

	// Load vector of values from contiguous memory addresses
	_cmu418_vload_float(x, values+i, maskAll);               // x = values[i];

	// Set mask according to predicate
	// 对maskAll中启用的每个通道，如果 x[i] < zero[i]，则maskIsNegative的对应位设为 1，否则设为 0。
	_cmu418_vlt_float(maskIsNegative, x, zero, maskAll);     // if (x < 0) {

	// Execute instruction using mask ("if" clause)
	_cmu418_vsub_float(result, zero, x, maskIsNegative);      //   output[i] = -x;

	// Inverse maskIsNegative to generate "else" mask
	maskIsNotNegative = _cmu418_mask_not(maskIsNegative);     // } else {

	// Execute instruction ("else" clause)
	_cmu418_vload_float(result, values+i, maskIsNotNegative); //   output[i] = x; }

	// Write results back to memory
	_cmu418_vstore_float(output+i, result, maskAll);
    }
}

// Accepts an array of values and an array of exponents
// For each element, compute values[i]^exponents[i] and clamp value to
// 4.18.  Store result in outputs.
// Uses iterative squaring, so that total iterations is proportional
// to the log_2 of the exponent
void clampedExpSerial(float* values, int* exponents, float* output, int N) {
    for (int i=0; i<N; i++) {
	float x = values[i];
	float result = 1.f;
	int y = exponents[i];
	float xpower = x;
	while (y > 0) {
	    if (y & 0x1)
		result *= xpower;
	    xpower = xpower * xpower;
	    y >>= 1;
	}
	if (result > 4.18f) {
	    result = 4.18f;
	}
	output[i] = result;
    }
}

void clampedExpVector(float* values, int* exponents, float* output, int N) {
    // Implement your vectorized version of clampedExpSerial here
	__cmu418_vec_float x;
	__cmu418_vec_int y;
    __cmu418_vec_float result;
	
	__cmu418_vec_int zero = _cmu418_vset_int(0);
	__cmu418_vec_int one = _cmu418_vset_int(1);
	__cmu418_vec_float limit = _cmu418_vset_float(4.18f);

    for(int i=0;i<N;i+=VECTOR_WIDTH){
		__cmu418_mask maskAll=i+VECTOR_WIDTH>N?_cmu418_init_ones(N-i):_cmu418_init_ones();
		_cmu418_vload_float(x,values+i,maskAll);
		_cmu418_vload_int(y,exponents+i,maskAll);
		_cmu418_vset_float(result,1.f,maskAll);

		__cmu418_mask activeMask=_cmu418_init_ones(0);
		_cmu418_vgt_int(activeMask, y, zero, maskAll);

		while(_cmu418_cntbits(activeMask)>0){
			//注意_cmu418_vbitand_int的第一个参数必须是__cmu418_vec_int引用类型，不能是__cmu418_mask引用类型，所以这里需要转换
			__cmu418_vec_int oddVec= _cmu418_vset_int(0);
			__cmu418_mask oddMask=_cmu418_init_ones(0);
			_cmu418_vbitand_int(oddVec, y, one, activeMask);
			_cmu418_veq_int(oddMask, oddVec, one, activeMask);
			_cmu418_vmult_float(result, result, x, oddMask);

			_cmu418_vmult_float(x, x, x, activeMask);
			_cmu418_vshiftright_int(y, y, one, activeMask);

			activeMask=_cmu418_init_ones(0);
			_cmu418_vgt_int(activeMask, y, zero, maskAll);

		}

		__cmu418_mask gtMask=_cmu418_init_ones(0);
		_cmu418_vgt_float(gtMask, result, limit, maskAll);
		_cmu418_vset_float(result, 4.18f, gtMask);

		_cmu418_vstore_float(output + i, result, maskAll);
	}
}


float arraySumSerial(float* values, int N) {
    float sum = 0;
    for (int i=0; i<N; i++) {
	sum += values[i];
    }

    return sum;
}

// Assume N % VECTOR_WIDTH == 0
// Assume VECTOR_WIDTH is a power of 2
float arraySumVector(float* values, int N) {
    // Implement your vectorized version here
	__cmu418_vec_float sumVec=_cmu418_vset_float(0.f);
	__cmu418_mask maskAll=_cmu418_init_ones();
	for(int i=0;i<N;i+=VECTOR_WIDTH){
		__cmu418_vec_float vec;
		_cmu418_vload_float(vec,values+i,maskAll);
		_cmu418_vadd_float(sumVec,sumVec,vec,maskAll);
	}
	__cmu418_vec_float temp;
	__cmu418_vec_float result;
	for(int i=0;i<log2(VECTOR_WIDTH);++i){
		_cmu418_hadd_float(temp, sumVec);
		_cmu418_interleave_float(result, temp);
		sumVec=result;
	}
	return result.value[0];

}
