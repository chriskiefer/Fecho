#ifndef Esnesnesn_fecho_h
#define Esnesnesn_fecho_h

#include <iostream>
#include <sys/types.h>
#include "dSFMTRand.h"
#include <vector>
#include <Accelerate/Accelerate.h>
#include <stdexcept>
#include "fechoActivations.h"
#include "fechoMath.h"
#include "fechoUtil.h"

using namespace std;

namespace Fecho {
    template<typename T>
    void debugArray(const string &msg, T *n, const int count) {
        cout << msg << ":";
        for(int i=0; i < count; i++) {
            cout << n[i] << ",";
        }
        cout << endl;
    }

    template<typename T> 
    class Reservoir {
    public:
        Reservoir(const uint inputSize, const uint reservoirSize, ActivationFunctionBase<T> *_act) {
            x.resize(reservoirSize, 0);
            inWeights.resize(reservoirSize * inputSize, 0);
            resWeights.resize(reservoirSize * reservoirSize, 0);
            inputs.resize(inputSize);
            nRes = reservoirSize;
            nIns = inputSize;
            act = _act;
            inScale = 1.0;
            inShift = 0.0;
            noise=0;
        }
        
        inline Reservoir& setInScale(T val) {inScale = val; return *this;}
        inline Reservoir& setInShift(T val) {inShift = val; return *this;}
        inline Reservoir& setNoise(T val) {noise = val; return *this;}
        
        inline vector<T> &getActivations() {return x;}
        inline vector<T> &getRes() {return resWeights;}
        inline vector<T> &getIns() {return inWeights;}
        inline vector<T> &getInputs() {return inputs;}
        void setInputs(T *newInputs) {
            memcpy(&inputs[0], newInputs, nIns * sizeof(T));            
        }
        inline const uint getNRes() {return nRes;}
        inline const uint getNIns() {return nIns;}
        void dump() {
            cout << "Inputs: " << nIns << "\tReservoir Nodes: " << nRes;
            cout << "\nInput weights: \n";
            for(int i=0; i < nRes * nIns; i++) cout << inWeights[i] << ",";
            cout << "\nReservoirweights: \n";
            for(int i=0; i < nRes * nRes; i++) cout << resWeights[i] << ",";
        }
        inline ActivationFunctionBase<T>* getActivationFunction() {return act;}
        void resetStates() {
            std::fill(x.begin(), x.end(), 0);
        }
        inline T* getInShift(){return &inShift;}
        inline T* getInScale(){return &inScale;}
        inline T getNoise(){return noise;}
    protected:
    private:
        Reservoir() {}
        vector<T> x; //activations
        vector<T> inWeights; //input weights 
        vector<T> resWeights; //reservoir weights
        vector<T> inputs; //input values
        uint nRes, nIns;
        ActivationFunctionBase<T> *act;        
        T inScale, inShift;
        T noise;
    };

    template<typename T>
    class ReadOut {
    public:
        //all matrices in row major format
        ReadOut(Reservoir<T> &_res, const uint _size, ActivationFunctionBase<T> *_act) : size(_size) {
            res = &_res;
            weightsRes.resize(res->getNRes() * size, 0);
            weightsIn.resize(res->getNIns() * size, 0);
            fbWeights.resize(res->getNRes() * size);
            outputs.resize(size, 0);
            outputsFromIns.resize(size, 0);
            act = _act;
            mapInsToOuts = true;
        }
        
        inline void update() {
            //multiply output weights by [input;x]
            //separate into two multiplications - res weights and inputs, then sum
            Math::mmul(&weightsRes[0], 1, &res->getActivations()[0], 1, &outputs[0], 1, size, 1, res->getNRes());
//            debugArray<T>("upres", &weightsRes[0], weightsRes.size());
//            debugArray<T>("act", &res->getActivations()[0], res->getNRes());
            if(res->getNIns() > 0 && mapInsToOuts) {
                Math::mmul(&weightsIn[0], 1, &res->getInputs()[0], 1, &outputsFromIns[0], 1, size, 1, res->getNIns());
                Math::vadd(&outputs[0], 1, &outputsFromIns[0], 1, &outputs[0], 1, size);
            }
            act->process(&outputs[0], size);
//            debugArray<T>("op", &outputs[0], outputs.size());
            
        }
        /**
         Force the output units to specific values. For when training with feedback weights.
         */
        inline void teacherForce(T* forcedValues) {
            //copy forced outputs to actual outputs
            memcpy(&outputs[0], forcedValues, sizeof(T) * size);
            act->process(&outputs[0], size);
        }
        
//        inline T* getOutputWeights() {return outs;}
        inline vector<T> &getOutputs() {return outputs;}
        inline const uint getSize() {return size;}
        inline Reservoir<T> &getRes() {return res;}
        inline ActivationFunctionBase<T>* getActivationFunction() {return act;}
        inline void setMapInsToOuts(bool val) {mapInsToOuts = val;}
        inline bool insAreMappedToOuts() {return mapInsToOuts;}
        inline vector<T> &getFbWeights() {return fbWeights;}        

        //input is (nRes * nIns) x nOuts matrix
        void setOutputWeights(T *newWeights) {
            //split into two matrices for res and inputs
            memcpy(&weightsRes[0], newWeights, weightsRes.size() * sizeof(T));
            if (mapInsToOuts)
                memcpy(&weightsIn[0], newWeights + weightsRes.size(), weightsIn.size() * sizeof(T));
        }
    protected:
        uint size;
        Reservoir<T> *res;
        vector<T> weightsRes;
        vector<T> weightsIn;
        vector<T> outputs, outputsFromIns;
        vector<T> fbWeights;
        ActivationFunctionBase<T> *act;
        bool mapInsToOuts;
    };
    
    template <typename T>
    class Initialiser {
    public:
        class EVException: public std::runtime_error { public: EVException(): std::runtime_error("Exception: Failed to compute the eigenvalues of the reservoir\n") {} };
        
        Initialiser() : resLow(-1), resHigh(1), resConnectivity(0.1), alpha(0.9), inLow(-1), inHigh(1), inConnectivity(0.1),
        fbLow(-1), fbHigh(1.0), fbConnectivity(1.0)
        {
            updateResRange();
            updateInRange();
            updateFbRange();
        }
        
        inline void updateResRange() {resRange = resHigh - resLow;}
        inline void updateInRange() {inRange = inHigh - inLow;}
        inline void updateFbRange() {fbRange = fbHigh - fbLow;}
        inline Initialiser& setSpectralRadius(T val) {alpha=val; return *this;}
        inline Initialiser& setResConnectivity(T val) {resConnectivity=val; return *this;}
        inline Initialiser& setResRangeLow(T val) {resLow=val; updateResRange(); return *this;}
        inline Initialiser& setResRangeHigh(T val) {resHigh=val; updateResRange(); return *this;}
        inline Initialiser& setInConnectivity(T val) {inConnectivity=val; return *this;}
        inline Initialiser& setInRangeLow(T val) {inLow=val; updateInRange(); return *this;}
        inline Initialiser& setInRangeHigh(T val) {inHigh=val; updateInRange(); return *this;}
        inline Initialiser& setFbConnectivity(T val) {fbConnectivity=val; return *this;}
        inline Initialiser& setFbRangeLow(T val) {fbLow=val; updateFbRange(); return *this;}
        inline Initialiser& setFbRangeHigh(T val) {fbHigh=val; updateFbRange(); return *this;}
        
        void init(Reservoir<T> &net, ReadOut<T> &ro) {
            
            //choose non-zero connections
            int resWeightCount = net.getNRes() * net.getNRes();
            vector<int> indexes(resWeightCount,0);
            for(int i=0; i < indexes.size(); i++) indexes[i] = i;
            random_shuffle(indexes.begin(), indexes.end());
            vector<T> &res = net.getRes();
            for(int i=0; i < resWeightCount * resConnectivity; i++) {
                res[indexes[i]] = resLow + (resRange * rand.randUF());
            }
            
            int inWeightsCount = net.getNIns() * net.getNRes();
            vector<int> inIndexes(inWeightsCount,0);
            for(int i=0; i < inIndexes.size(); i++) inIndexes[i] = i;
            random_shuffle(inIndexes.begin(), inIndexes.end());
            vector<T> &ins = net.getIns();
            for(int i=0; i < inWeightsCount * inConnectivity; i++) {
                ins[inIndexes[i]] = inLow + (inRange * rand.randUF());
            }
            
            int fbWeightsCount = net.getNRes();
            vector<int> fbIndexes(fbWeightsCount,0);
            for(int i=0; i < fbIndexes.size(); i++) fbIndexes[i] = i;
            random_shuffle(fbIndexes.begin(), fbIndexes.end());
            vector<T> &fbWeights = ro.getFbWeights();
            for(int i=0; i < fbWeightsCount * fbConnectivity; i++) {
                fbWeights[fbIndexes[i]] = fbLow + (fbRange * rand.randUF());
            }
            
            //scale weights
            //find the eigenvalues of the reservoir
            char nchar = 'N';
            __CLPK_integer n = net.getNRes();
            vector<T> wr(n), wi(n);
            vector<T> rescopy(resWeightCount);
            memcpy(&rescopy[0], &res[0], resWeightCount * sizeof(T));
            __CLPK_integer lwork = -1;
            T wkopt;
            __CLPK_integer info;
            Math::geev(&nchar, &nchar, &n, &rescopy[0], &n, &wr[0], &wi[0], NULL, &n, NULL, &n, &wkopt, &lwork, &info);
            lwork = (int)wkopt;
            vector<T> work(lwork);
            Math::geev(&nchar, &nchar, &n, &rescopy[0], &n, &wr[0], &wi[0], NULL, &n, NULL, &n, &work[0], &lwork, &info);
            if (info > 0) {
                throw(EVException());
            }else{
                //get max eigenvalue
                for(int i=0; i < n; i++) {
                    wr[i] = sqrt(pow(wr[i],2) + pow(wi[i],2));
                }
                T maxEigenvalue = *max_element(wr.begin(), wr.end());
                
                //do the scaling
                T scaleFactor = alpha / maxEigenvalue;
                for(int i=0; i < resWeightCount * resConnectivity; i++) {
                    res[indexes[i]] *= scaleFactor;
                }
            }
            
            
        }
    private:
        dSFMTRand rand;        
        T resRange, resLow, resHigh;
        T inRange, inLow, inHigh;
        T fbRange, fbLow, fbHigh;
        T resConnectivity, inConnectivity, fbConnectivity;
        T alpha;
    };
    
    template <typename T>
    class Simulator {
    public:
        Simulator(Reservoir<T> &_net, ReadOut<T> &_ro) {
            net = &_net;
            ro = &_ro;
            WinxU.resize(net->getNRes());
            WxX.resize(net->getNRes());
            WxY.resize(net->getNRes());
            res = &net->getRes()[0];
            x = &net->getActivations()[0];
            ins = &net->getIns()[0];
            inputsSS.resize(net->getNIns());
            noiseBuffer.resize(net->getNRes()); //used for double->float conversions
            noiseVect.resize(net->getNRes());
            twiceNoise = net->getNoise() * 2.0;
        }
       
        virtual inline void simulate(T *inputs) {
            simulateOneEpoch(inputs);
            ro->update();
        }
        
        inline void randArray(vector<T> &vals);
        
        inline void simulateOneEpoch(T *inputs) {
            net->setInputs(inputs);
            
            if (net->getNoise() > 0) {
                for(int i=0; i < noiseVect.size(); i++) 
                    noiseVect[i] = rand.randUF();
                T shift = -0.5;
                //shift range -0.5 < x < 0.5
                Math::vsadd(&noiseVect[0], 1, &shift, &noiseVect[0], 1, noiseVect.size());
                //scale and add to x
                Math::vsma(&noiseVect[0], 1, &twiceNoise, x, 1, x, 1, net->getNRes());
            }
            
            //res weights * x
            Math::mmul(res, 1, x, 1, &WxX[0], 1, net->getNRes(), 1, net->getNRes());
            
            if (net->getNIns() > 0) {
                //scale and shift inputs
                Math::vsmul(inputs, 1, net->getInScale(), &inputsSS[0], 1, net->getNIns());
                Math::vsadd(&inputsSS[0], 1, net->getInShift(), &inputsSS[0], 1, net->getNIns());
                
                //ins * inputs
                Math::mmul(ins, 1, inputs, 1, &WinxU[0], 1, net->getNRes(), 1, net->getNIns());
                //add together
                Math::vadd(&WxX[0], 1, &WinxU[0], 1, x, 1, net->getNRes());

            }else{
                memcpy(x, &WxX[0], sizeof(T) * net->getNRes());
            }

            //outs * fb
            Math::mmul(&ro->getFbWeights()[0], 1, &ro->getOutputs()[0], 1, &WxY[0], 1, net->getNRes(), 1, ro->getSize());
            //add together
            Math::vadd(&WxY[0], 1, x, 1, x, 1, net->getNRes());
            
            //activation function
            net->getActivationFunction()->process(x, net->getNRes());
            //        debugArray<float>("Act:", x, net->getNRes()); 
            
        }
        inline Reservoir<T>* getRes() {return net;}
    protected:
        vector<T> WinxU, WxX, WxY;
        T *res, *x, *ins;
        Reservoir<T> *net;
        ReadOut<T> *ro;
        vector<T> inputsSS;
        dSFMTRand rand;   
        vector<double> noiseBuffer;
        vector<T> noiseVect;
        T twiceNoise;
    };
    
    template<> inline void Simulator<double>::randArray(vector<double> &vals) {
        rand.randArray(&vals[0], vals.size());
    }

    template<> inline void Simulator<float>::randArray(vector<float> &vals) {
        rand.randArray(&noiseBuffer[0], vals.size());
        vDSP_vdpsp(&noiseBuffer[0], 1, &vals[0], 1, vals.size());
    }

    template <typename T>
    class SimulatorLI : public Simulator<T> {
    public:
        SimulatorLI(Reservoir<T> &_net, ReadOut<T> &_ro, T leakRate) : Simulator<T>(_net, _ro), alpha(1-leakRate) {
            leakyX.resize(_net.getNRes(),0);
        }
        
        inline void simulate(T *inputs) {
            //(1-alpha)Xn
            Math::vsmul(this->x, 1, &alpha, &leakyX[0], 1, this->net->getNRes());
            this->simulateOneEpoch(inputs);
            Math::vadd(this->x, 1, &leakyX[0], 1, this->x, 1, this->net->getNRes());
            this->ro->update();
        }
        

    protected:
        T alpha;
        vector<T> leakyX;
    };
    
    
    class TrainingException: public std::runtime_error { public: TrainingException(): std::runtime_error("An error occured during training.\n") {} };

    template<typename T>
    class TrainerLeastSquares {
    public:
        
        /**
         *Construct the trainer. Matrices are in row-major format
         */
        TrainerLeastSquares(Simulator<T> *_sim, ReadOut<T> *_rOut, vector<T> &inputsMatrix, vector<T> &desiredOutputsMatrix, uint trainDataCount, const uint _washout) : sim(_sim), ro(_rOut), trainSize(trainDataCount), washout(_washout) {
            inputData = &inputsMatrix[0];
            outputDataCopy = desiredOutputsMatrix; //copy
            outputData = &outputDataCopy[0];
            stateSize = sim->getRes()->getNRes();
            if (ro->insAreMappedToOuts()) {
                stateSize += sim->getRes()->getNIns();
            }
            collectedStatesCount = (trainSize - washout);
            extStates.resize(collectedStatesCount * stateSize);
        }
        void train() {
            int inputIdx=0;
            //run to washout
            for(int i=0; i < trainSize; i++) {
                sim->simulate(inputData + (inputIdx * sim->getRes()->getNIns()));
                ro->teacherForce(outputData + (i * ro->getSize()));
                inputIdx++;
            }
            //harvest states
            for(int i=0; i < collectedStatesCount; i++) {
                //run an iteration of the system
                sim->simulate(inputData + (inputIdx * sim->getRes()->getNIns()));
                ro->teacherForce(outputData + ((i+washout) * ro->getSize()));
                
                //copy states
                memcpy(&extStates[0] + (i * stateSize), &sim->getRes()->getActivations()[0], sim->getRes()->getNRes() * sizeof(T));
                if (ro->insAreMappedToOuts()) {
                    memcpy(&extStates[0] + (i * stateSize) + sim->getRes()->getNRes(), &sim->getRes()->getInputs()[0], sim->getRes()->getNIns() * sizeof(T));
                }
                inputIdx++;
            }
            //inverse-activate desired outs
            ro->getActivationFunction()->invProcess(outputData + (washout * ro->getSize()), collectedStatesCount * ro->getSize());
            
            solvedWeights.resize(stateSize * ro->getSize());
            solveOutputWeights();
//            debugArray<T>("neww", &solvedWeights[0], solvedWeights.size());
            ro->setOutputWeights(&solvedWeights[0]);
        }

        virtual void solveOutputWeights() {
            //transpose to col major for clapack
            vector<T> extStatesCM(extStates.size());
            vector<T> desOutsCM(collectedStatesCount * ro->getSize());
            Math::rowMajorToColMajor(&extStates[0], stateSize, collectedStatesCount, &extStatesCM[0]);
            Math::rowMajorToColMajor(outputData + (washout * ro->getSize()), ro->getSize(), collectedStatesCount, &desOutsCM[0]);
            char trans = 'N';
            __CLPK_integer m=collectedStatesCount;
            __CLPK_integer n=stateSize;
            __CLPK_integer nrhs = ro->getSize();
            __CLPK_integer lwork = -1;
            T wkopt;
            __CLPK_integer info;
//            debugArray<T>("ext", &extStatesCM[0], extStatesCM.size());
//            debugArray<T>("des", &desOutsCM[0], desOutsCM.size());

            Math::gels(&trans, &m, &n, &nrhs, &extStatesCM[0], &m, &desOutsCM[0], &m, &wkopt, &lwork, &info);
            lwork = (int)wkopt;
            vector<T> work(lwork);
            Math::gels(&trans, &m, &n, &nrhs, &extStatesCM[0], &m, &desOutsCM[0], &m, &work[0], &lwork, &info);
            cout << "info: " << info << endl;
            if (info > 0) {
                throw(TrainingException());
            }else{
                //            debugArray<T>("Result:", &desOutsCM[0], desOutsCM.size());
                vector<T> result(nrhs * n);
                for(int i=0; i < nrhs; i++) {
                    for(int j=0; j < n; j++) {
                        result[(i*n) + j] = desOutsCM[(i*m) + j];
                    }
                }
                Math::colMajorToRowMajor<T>(&result[0], nrhs, stateSize, &solvedWeights[0]);
            }
        }
        
    protected:
        T* inputData;
        T* outputData;
        vector<T> outputDataCopy;
        uint trainSize;
        uint collectedStatesCount;
        uint washout;
        vector<T> extStates;
        Simulator<T> *sim;
        ReadOut<T> * ro;
        int stateSize;
        vector<T> solvedWeights;
    };
    
    template<typename T>
    class TrainerPseudoInverse : public TrainerLeastSquares<T> {
    public:
        TrainerPseudoInverse(Simulator<T> *_sim, ReadOut<T> *_rOut, vector<T> &inputsMatrix, vector<T> &desiredOutputsMatrix, uint trainDataCount, const uint _washout) : TrainerLeastSquares<T>(_sim, _rOut, inputsMatrix, desiredOutputsMatrix, trainDataCount, _washout) {
            
        }

        void solveOutputWeights() {
            //transpose to col major for clapack
            vector<T> extStatesCM(this->extStates.size());
            vector<T> desOutsCM(this->collectedStatesCount * this->ro->getSize());
            Math::rowMajorToColMajor(&this->extStates[0], this->stateSize, this->collectedStatesCount, &extStatesCM[0]);
            Math::rowMajorToColMajor(this->outputData + (this->washout * this->ro->getSize()), this->ro->getSize(), this->collectedStatesCount, &desOutsCM[0]);
            __CLPK_integer m=this->collectedStatesCount;
            __CLPK_integer n=this->stateSize;
            __CLPK_integer nrhs = this->ro->getSize();
            __CLPK_integer lwork = -1;
            T wkopt;
            __CLPK_integer info;
            //            debugArray<T>("ext", &extStatesCM[0], extStatesCM.size());
            //            debugArray<T>("des", &desOutsCM[0], desOutsCM.size());
            vector<T> s(min(m,n));
            T rcond = -1;
            __CLPK_integer rank;
            lwork = -1;
            Math::gelss(&m, &n, &nrhs, &extStatesCM[0], &m, &desOutsCM[0], &m, &s[0], &rcond, &rank, &wkopt, &lwork, &info);
            lwork = (int)wkopt;
            vector<T> work2(lwork);
            Math::gelss(&m, &n, &nrhs, &extStatesCM[0], &m, &desOutsCM[0], &m, &s[0], &rcond, &rank, &work2[0], &lwork, &info);
            cout << "info: " << info << endl;
            if(info > 0) {
                throw(TrainingException());
            }else{
    //            debugArray<T>("Result:", &desOutsCM[0], desOutsCM.size());
                vector<T> result(nrhs * n);
                for(int i=0; i < nrhs; i++) {
                    for(int j=0; j < n; j++) {
                        result[(i*n) + j] = desOutsCM[(i*m) + j];
                    }
                }
                Math::colMajorToRowMajor<T>(&result[0], nrhs, this->stateSize, &this->solvedWeights[0]);
            }
        }
    };
    


    

}

#endif
