#ifndef Esnesnesn_fechoMath_h
#define Esnesnesn_fechoMath_h

#include <Accelerate/Accelerate.h>
using namespace std;

namespace Fecho {
    
    //utilities, and overloaded functions for choose float or double routines for various math tasks
    
    class Math {
    public:
        
        //utilities
        
        template<typename T>
        static inline void rowMajorToColMajor(T *x, const int cols, const int rows, T *y) {
            int cmx, cmy;
            cmx=cmy=0;
            for(int i=0; i < cols * rows; i++) {
                y[i] = x[(cmy*cols) + cmx];
                //don't use %, it's expensive
                cmy++;
                if(cmy == rows) {
                    cmy=0;
                    cmx++;
                }
            }
        }

        template<typename T>
        static inline void colMajorToRowMajor(T *x, const int cols, const int rows, T *y) {
            int cmx, cmy;
            cmx=cmy=0;
            for(int i=0; i < cols * rows; i++) {
                y[i] = x[(cmy*rows) + cmx];
                //don't use %, it's expensive
                cmy++;
                if(cmy == cols) {
                    cmy=0;
                    cmx++;
                }
            }
        }

        //from CLAPACK
        static inline void geev(char *jobvl, char *jobvr, __CLPK_integer *n, __CLPK_real *a, 
                         __CLPK_integer *lda, __CLPK_real *wr, __CLPK_real *wi, __CLPK_real *vl, __CLPK_integer *ldvl, __CLPK_real *vr, __CLPK_integer *ldvr, __CLPK_real *work, __CLPK_integer *lwork, __CLPK_integer *info) {
            sgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);            
        }
        
        static inline void geev(char *jobvl, char *jobvr, __CLPK_integer *n, __CLPK_doublereal *
                         a, __CLPK_integer *lda, __CLPK_doublereal *wr, __CLPK_doublereal *wi, __CLPK_doublereal *vl, __CLPK_integer *ldvl, __CLPK_doublereal *vr, __CLPK_integer *ldvr, __CLPK_doublereal *work, __CLPK_integer *lwork, __CLPK_integer *info) {
            dgeev_(jobvl, jobvr, n, a, lda, wr, wi, vl, ldvl, vr, ldvr, work, lwork, info);
            
        }
        
        static inline void gelss(__CLPK_integer *m, __CLPK_integer *n, __CLPK_integer *nrhs, __CLPK_real *a, 
                                 __CLPK_integer *lda, __CLPK_real *b, __CLPK_integer *ldb, __CLPK_real *s, __CLPK_real *rcond, __CLPK_integer *
                                 rank, __CLPK_real *work, __CLPK_integer *lwork, __CLPK_integer *info) {
            sgelss_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, info);
        }
        
        static inline void gelss(__CLPK_integer *m, __CLPK_integer *n, __CLPK_integer *nrhs, 
                                 __CLPK_doublereal *a, __CLPK_integer *lda, __CLPK_doublereal *b, __CLPK_integer *ldb, __CLPK_doublereal *
                                 s, __CLPK_doublereal *rcond, __CLPK_integer *rank, __CLPK_doublereal *work, __CLPK_integer *lwork, 
                                 __CLPK_integer *info) {
            dgelss_(m, n, nrhs, a, lda, b, ldb, s, rcond, rank, work, lwork, info);
        }
        
        static inline void gels(char *trans, __CLPK_integer *m, __CLPK_integer *n, __CLPK_integer *
                                nrhs, __CLPK_real *a, __CLPK_integer *lda, __CLPK_real *b, __CLPK_integer *ldb, __CLPK_real *work, 
                                __CLPK_integer *lwork, __CLPK_integer *info) {
            sgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
        }
        static inline void gels(char *trans, __CLPK_integer *m, __CLPK_integer *n, __CLPK_integer *
                                nrhs, __CLPK_doublereal *a, __CLPK_integer *lda, __CLPK_doublereal *b, __CLPK_integer *ldb, 
                                __CLPK_doublereal *work, __CLPK_integer *lwork, __CLPK_integer *info) {
            dgels_(trans, m, n, nrhs, a, lda, b, ldb, work, lwork, info);
        }
        
        
        
        //FROM VDSP
        
        static inline void mattrans(float *__vDSP_a,
                                  vDSP_Stride __vDSP_aStride,
                                  float *__vDSP_c,
                                  vDSP_Stride __vDSP_cStride,
                                  vDSP_Length __vDSP_M,
                                  vDSP_Length __vDSP_N) {
            vDSP_mtrans(__vDSP_a, __vDSP_aStride, __vDSP_c, __vDSP_cStride, __vDSP_M, __vDSP_N);
        }
        static inline void mattrans(double *__vDSP_a,
                                  vDSP_Stride __vDSP_aStride,
                                  double *__vDSP_c,
                                  vDSP_Stride __vDSP_cStride,
                                  vDSP_Length __vDSP_M,
                                  vDSP_Length __vDSP_N) {
            vDSP_mtransD(__vDSP_a, __vDSP_aStride, __vDSP_c, __vDSP_cStride, __vDSP_M, __vDSP_N);
        }
        
        static inline void matmul(float *       __vDSP_a,
                                     vDSP_Stride   __vDSP_aStride,
                                     float *       __vDSP_b,
                                     vDSP_Stride   __vDSP_bStride,
                                     float *       __vDSP_c,
                                     vDSP_Stride   __vDSP_cStride,
                                     vDSP_Length   __vDSP_M,
                                     vDSP_Length   __vDSP_N,
                                     vDSP_Length   __vDSP_P) {
            vDSP_mmul(__vDSP_a, __vDSP_aStride, __vDSP_b, __vDSP_bStride, __vDSP_c, __vDSP_cStride, __vDSP_M, __vDSP_N, __vDSP_P);
        }
        
        static inline void matmul(double *      __vDSP_a,
                                          vDSP_Stride   __vDSP_aStride,
                                          double *      __vDSP_b,
                                          vDSP_Stride   __vDSP_bStride,
                                          double *      __vDSP_c,
                                          vDSP_Stride   __vDSP_cStride,
                                          vDSP_Length   __vDSP_M,
                                          vDSP_Length   __vDSP_N,
                                          vDSP_Length   __vDSP_P) {
            vDSP_mmulD(__vDSP_a, __vDSP_aStride, __vDSP_b, __vDSP_bStride, __vDSP_c, __vDSP_cStride, __vDSP_M, __vDSP_N, __vDSP_P);
            
        }
        
        static inline void vecadd(
                  const float   __vDSP_input1[],
                  vDSP_Stride   __vDSP_stride1,
                  const float   __vDSP_input2[],
                  vDSP_Stride   __vDSP_stride2,
                  float         __vDSP_result[],
                  vDSP_Stride   __vDSP_strideResult,
                                vDSP_Length   __vDSP_size) {
            vDSP_vadd(__vDSP_input1, __vDSP_stride1, __vDSP_input2, __vDSP_stride2, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }
        
        static inline void vecadd(const double   __vDSP_input1[],
                                      vDSP_Stride    __vDSP_stride1,
                                      const double   __vDSP_input2[],
                                      vDSP_Stride    __vDSP_stride2,
                                      double         __vDSP_result[],
                                      vDSP_Stride    __vDSP_strideResult,
                                vDSP_Length    __vDSP_size) {
            vDSP_vaddD(__vDSP_input1, __vDSP_stride1, __vDSP_input2, __vDSP_stride2, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }
        
        static inline void vecsadd(float *       __vDSP_A,
                                 vDSP_Stride   __vDSP_I,
                                 float *       __vDSP_B,
                                 float *       __vDSP_C,
                                 vDSP_Stride   __vDSP_K,
                                 vDSP_Length   __vDSP_N
                                 ){
            vDSP_vsadd(__vDSP_A, __vDSP_I, __vDSP_B, __vDSP_C, __vDSP_K, __vDSP_N);
        }
        
        static inline void vecsadd(double *       __vDSP_A,
                                 vDSP_Stride   __vDSP_I,
                                 double *       __vDSP_B,
                                 double *       __vDSP_C,
                                 vDSP_Stride   __vDSP_K,
                                 vDSP_Length   __vDSP_N
                                 ){
            vDSP_vsaddD(__vDSP_A, __vDSP_I, __vDSP_B, __vDSP_C, __vDSP_K, __vDSP_N);
        }
        
        static inline void vecsmul(const float    __vDSP_input1[],
                                 vDSP_Stride    __vDSP_stride1,
                                 const float *  __vDSP_input2,
                                 float          __vDSP_result[],
                                 vDSP_Stride    __vDSP_strideResult,
                                 vDSP_Length    __vDSP_size) {
            vDSP_vsmul(__vDSP_input1, __vDSP_stride1, __vDSP_input2, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }
        static inline void vecsmul(const double    __vDSP_input1[],
                                 vDSP_Stride     __vDSP_stride1,
                                 const double *  __vDSP_input2,
                                 double          __vDSP_result[],
                                 vDSP_Stride     __vDSP_strideResult,
                                 vDSP_Length     __vDSP_size) {
            vDSP_vsmulD(__vDSP_input1, __vDSP_stride1, __vDSP_input2, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }
        
        static inline void vecsma(const float *  __vDSP_A,
                                vDSP_Stride    __vDSP_I,
                                const float *  __vDSP_B,
                                const float *  __vDSP_C,
                                vDSP_Stride    __vDSP_K,
                                float *        __vDSP_D,
                                vDSP_Stride    __vDSP_L,
                                vDSP_Length    __vDSP_N) {
            vDSP_vsma(__vDSP_A, __vDSP_I, __vDSP_B, __vDSP_C, __vDSP_K, __vDSP_D, __vDSP_L, __vDSP_N);
        }

        static inline void vecsma(const double *  __vDSP_A,
                                vDSP_Stride     __vDSP_I,
                                const double *  __vDSP_B,
                                const double *  __vDSP_C,
                                vDSP_Stride     __vDSP_K,
                                double *        __vDSP_D,
                                vDSP_Stride     __vDSP_L,
                                vDSP_Length     __vDSP_N) {
            vDSP_vsmaD(__vDSP_A, __vDSP_I, __vDSP_B, __vDSP_C, __vDSP_K, __vDSP_D, __vDSP_L, __vDSP_N);
        }
        
        static inline void vecsub(const float   __vDSP_input1[],
                                vDSP_Stride   __vDSP_stride1,
                                const float   __vDSP_input2[],
                                vDSP_Stride   __vDSP_stride2,
                                float         __vDSP_result[],
                                vDSP_Stride   __vDSP_strideResult,
                                vDSP_Length   __vDSP_size) {
            vDSP_vsub(__vDSP_input1, __vDSP_stride1, __vDSP_input2, __vDSP_stride2, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }

        static inline void vecsub(const double   __vDSP_input1[],
                                vDSP_Stride    __vDSP_stride1,
                                const double   __vDSP_input2[],
                                vDSP_Stride    __vDSP_stride2,
                                double         __vDSP_result[],
                                vDSP_Stride    __vDSP_strideResult,
                                vDSP_Length    __vDSP_size) {
            vDSP_vsubD(__vDSP_input1, __vDSP_stride1, __vDSP_input2, __vDSP_stride2, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }
        
        static inline void vecsq(const float   __vDSP_input[],
                               vDSP_Stride   __vDSP_strideInput,
                               float         __vDSP_result[],
                               vDSP_Stride   __vDSP_strideResult,
                               vDSP_Length   __vDSP_size) {
            vDSP_vsq(__vDSP_input, __vDSP_strideInput, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }

        static inline void vecsq(const double   __vDSP_input[],
                               vDSP_Stride    __vDSP_strideInput,
                               double         __vDSP_result[],
                               vDSP_Stride    __vDSP_strideResult,
                               vDSP_Length    __vDSP_size) {
            vDSP_vsqD(__vDSP_input, __vDSP_strideInput, __vDSP_result, __vDSP_strideResult, __vDSP_size);
        }
        
        static inline void vecmean(float *       __vDSP_A,
                                 vDSP_Stride   __vDSP_I,
                                 float *       __vDSP_C,
                                 vDSP_Length   __vDSP_N) {
            vDSP_meanv(__vDSP_A, __vDSP_I, __vDSP_C, __vDSP_N);
        }

        static inline void vecmean(double *      __vDSP_A,
                                 vDSP_Stride   __vDSP_I,
                                 double *      __vDSP_C,
                                 vDSP_Length   __vDSP_N) {
            vDSP_meanvD(__vDSP_A, __vDSP_I, __vDSP_C, __vDSP_N);
        }
        
        static inline void vecclr(float *       __vDSP_C,
                                vDSP_Stride   __vDSP_K,
                                vDSP_Length   __vDSP_N) {
            vDSP_vclr(__vDSP_C, __vDSP_K, __vDSP_N);
        }
        static inline void vecclr(double *       __vDSP_C,
                                vDSP_Stride   __vDSP_K,
                                vDSP_Length   __vDSP_N) {
            vDSP_vclrD(__vDSP_C, __vDSP_K, __vDSP_N);
        }

        
        //FROM vforce
        static inline void vforcetanh(float *x, const float *y, const int *z) {
            vvtanhf(x, y, z);
        }
        static inline void vforcetanh(double *x, const double *y, const int *z) {
            vvtanh(x, y, z);
        }

        static inline void vforceatanh(float *x, const float *y, const int *z) {
            vvatanhf(x, y, z);
        }
        static inline void vforceatanh(double *x, const double *y, const int *z) {
            vvatanh(x, y, z);
        }

        static inline void vforceexp(float *x, const float *y, const int *z) {
            vvexpf(x, y, z);
        }
        static inline void vforceexp(double *x, const double *y, const int *z) {
            vvexp(x, y, z);
        }

        static inline void vforcerec(float *x, const float *y, const int *z) {
            vvrecf(x, y, z);
        }
        static inline void vforcerec(double *x, const double *y, const int *z) {
            vvrec(x, y, z);
        }

        static inline void vforcelog(float *x, const float *y, const int *z) {
            vvlogf(x, y, z);
        }
        static inline void vforcelog(double *x, const double *y, const int *z) {
            vvlog(x, y, z);
        }

        

    };
    

};

#endif
