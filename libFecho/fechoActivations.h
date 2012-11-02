#ifndef Esnesnesn_fechoActivations_h
#define Esnesnesn_fechoActivations_h
#include "fechoMath.h"

namespace Fecho {
    
    template <typename T>
    class ActivationFunctionBase {
    public:
        ActivationFunctionBase() {}
        virtual void process(T *x, const int size) {};
        virtual void invProcess(T* x, const int size) {};
    };
    
    template <typename T>
    class ActivationFunctionLinear : public ActivationFunctionBase<T> {
    public:
        inline void process(T *x, const int size) {};
        inline void invProcess(T* x, const int size) {};
    };
    
    
    template <typename T>
    class ActivationFunctionTanh : public ActivationFunctionBase<T> {
    public:
        inline void process(T *x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = tanh(x[i]);
#else
            Math::vforcetanh(x, x, &size);
#endif
        };
        inline void invProcess(T* x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = atanh(x[i]);
#else
            Math::vforceatanh(x, x, &size);
#endif
        };
    };
    

    template <typename T>
    class ActivationFunctionSigmoid : public ActivationFunctionBase<T> {
    public:
        inline void process(T *x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = 1.0 / (1.0 + exp(x[i]));
#else
            Math::vforceexp(x, x, &size);
            T one = 1.0f;
            Math::vsadd(x, 1, &one, x, 1, size);
            Math::vforcerec(x, x, &size);
#endif
        }
        inline void invProcess(T* x, const int size) {
#ifdef TARGET_OS_IPHONE
            for(int i=0; i < size; i++) x[i] = log(1.0/x[i] - 1.0);
#else
            Math::vforcerec(x, x, &size);
            T minusone = -1.0f;
            Math::vsadd(x, 1, &minusone, x, 1, size);
            Math::vforcelog(x, x, &size);
#endif
        }
    };
}

#endif
