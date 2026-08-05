#include <smmintrin.h> // For _mm_stream_load_si128
#include <emmintrin.h> // For _mm_mul_ps
#include <assert.h>
#include <stdint.h>

extern void saxpySerial(int N,
			float scale,
			float X[],
			float Y[],
			float result[]);


void saxpyStreaming(int N,
                    float scale,
                    float X[],
                    float Y[],
                    float result[])
{
    // Replace this code with ones that make use of the streaming instructions
    // saxpySerial(N, scale, X, Y, result);
    __m128 scale_vec = _mm_set1_ps(scale);
    int i;
    for(i=0;i+8<N;i+=8){
        __m128 x_vec = _mm_loadu_ps(&X[i]);
        __m128 y_vec = _mm_loadu_ps(&Y[i]);
        __m128 r_vec = _mm_add_ps(_mm_mul_ps(scale_vec, x_vec), y_vec);
        _mm_storeu_ps(&result[i], r_vec);
    }
    for (;i<N;++i) {
        result[i]=scale*X[i]+Y[i];
    }
    _mm_sfence();
}

