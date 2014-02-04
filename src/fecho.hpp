#ifndef Esnesnesn_fecho_h
#define Esnesnesn_fecho_h

#include <iostream>
#include <sys/types.h>
#include <vector>
#include <stdexcept>
#include "fechoActivations.hpp"
#include "fechoUtil.hpp"

#include "armadillo"

using namespace std;
using namespace arma;

namespace Fecho {

    template<typename T>
    class Reservoir {
    public:
        Reservoir() {}
        Reservoir(const uint inputSize, const uint reservoirSize, ActivationFunctionBase<T> *_act) {
            init(inputSize, reservoirSize, _act);
        }
        void init(const uint inputSize, const uint reservoirSize, ActivationFunctionBase<T> *_act) {
            x.set_size(reservoirSize);
            inWeights.set_size(reservoirSize, inputSize);
            resWeights.set_size(reservoirSize, reservoirSize);
            inputs.set_size(inputSize);
            nRes = reservoirSize;
            nIns = inputSize;
            act = _act;
            noise=0;
            scale = 1.0;
        }
        
        inline Reservoir& setNoise(T val) {noise = val; return *this;}
        
        inline Col<T> &getActivations() {return x;}
        inline const Mat<T> &getRes() {return resWeights;}
        inline void setResWeights(Mat<T> newWeights) {resWeights = newWeights; orgResWeights = newWeights;}
        inline Mat<T> &getIns() {return inWeights;}
        inline Col<T> &getInputs() {return inputs;}
        void setInputs(Col<T> &newInputs) {
            inputs = newInputs;
        }
        inline const uint getNRes() {return nRes;}
        inline const uint getNIns() {return nIns;}
        void dump() {
            cout << "Inputs: " << nIns << "\tReservoir Nodes: " << nRes;
            cout << "\nInput weights: \n";
            cout << inWeights << endl;
            cout << "\nReservoir weights: \n";
            cout << resWeights << endl;
        }
        inline ActivationFunctionBase<T>* getActivationFunction() {return act;}
        void setActivationFunction(ActivationFunctionBase<T> *newAct){act = newAct;}
        void resetActivations() {
            x.fill(0);
        }
        void randomiseActivations() {
            x.randu();
            x = (x * 2.0) - 1.0;
        }
        void setActivations(Col<T> newStates) {x = newStates;}
        inline T getNoise(){return noise;}
        inline void scaleReservoir(float newScale) {scale = newScale; resWeights = orgResWeights * scale;}
        inline T getScale() {return scale;}
    protected:
        Col<T> x; //activations
        Mat<T> inWeights; //input weights
        Mat<T> resWeights;  //reservoir weights
        Mat<T> orgResWeights; // original weights, used for realtime alpha modifications
        Col<T> inputs; //input values
        uint nRes, nIns;
        ActivationFunctionBase<T> *act;        
        T noise;
        T scale;
    };

    
    template<typename T>
    class ReadOut {
    public:
        ReadOut() {}
        
        ReadOut(Reservoir<T> &_res, const uint _size, ActivationFunctionBase<T> *_act) {
            init(_res, _size, _act);
        }
        void init(Reservoir<T> &_res, const uint _size, ActivationFunctionBase<T> *_act) {
            size = _size;
            res = &_res;
            weightsRes.set_size(size, res->getNRes());
            weightsRes.fill(0);
            weightsIn.set_size(size, res->getNIns());
            weightsIn.fill(0);
            fbWeights.set_size(res->getNRes(), size);
            fbWeights.fill(0);
            outputs.set_size(size);
            outputs.fill(0);
            outputsFromIns.set_size(size);
            outputsFromIns.fill(0);
            act = _act;
            mapInsToOuts = true;
            feedbackOn = false;
        }
        
        inline void update() {
            outputs = weightsRes * res->getActivations(); //todo: lose the trans here
            
            if(res->getNIns() > 0 && mapInsToOuts) {
                outputsFromIns = weightsIn * res->getInputs();
                outputs = outputs + outputsFromIns;
            }
            act->process(outputs);
            
        }
        
        /**
         Force the output units to specific values. For when training with feedback weights.
         */
        inline void teacherForce(Col<T> forcedValues) {
            //copy forced outputs to actual outputs
            outputs = forcedValues;
            act->process(outputs);
        }
        
        inline Col<T> &getOutputs() {return outputs;}
        inline const uint getSize() {return size;}
        inline Reservoir<T> &getRes() {return res;}
        inline ActivationFunctionBase<T>* getActivationFunction() {return act;}
        void setActivationFunction(ActivationFunctionBase<T> *newAct){act = newAct;}
        inline void setMapInsToOuts(bool val) {mapInsToOuts = val;}
        inline bool insAreMappedToOuts() {return mapInsToOuts;}
        inline Mat<T> &getFbWeights() {return fbWeights;}
        inline Mat<T> &getResWeights() {return weightsRes;}

        //input is (nRes + nIns) x nOuts matrix,
        void setOutputWeights(Mat<T> &newWeights) {
            weightsRes = trans(newWeights.rows(0, res->getNRes()-1));
            if (mapInsToOuts)
                weightsIn = trans(newWeights.rows(res->getNRes(), res->getNRes() + res->getNIns() - 1));
        }
        
        void dump() {
            cout << "Readout: \n";
            cout << weightsRes << endl;
            cout << weightsIn << endl;
            cout << fbWeights << endl;
        }
        inline bool isFeedbackOn() {return feedbackOn;}
        inline void setFeedbackOn(bool newVal) {feedbackOn = newVal;}
        
        
    protected:
        uint size;
        Reservoir<T> *res;
        Mat<T> weightsRes;
        Mat<T> weightsIn;
        Col<T> outputs, outputsFromIns;
        Mat<T> fbWeights;
        ActivationFunctionBase<T> *act;
        bool mapInsToOuts;
        bool feedbackOn;
    };
    
    template <typename T>
    class ReservoirInitialiser {
    public:
        class EVException: public std::runtime_error { public: EVException(): std::runtime_error("Exception: Failed to compute the eigenvalues of the reservoir\n") {} };
        
        ReservoirInitialiser() : resLow(-1), resHigh(1), resConnectivity(0.1), inLow(-1), inHigh(1), inConnectivity(0.1), alpha(0.9)
        {
            updateResRange();
            updateInRange();
        }
        
        inline void updateResRange() {resRange = resHigh - resLow;}
        inline void updateInRange() {inRange = inHigh - inLow;}
        inline ReservoirInitialiser& setSpectralRadius(T val) {alpha=val; return *this;}
        inline ReservoirInitialiser& setResConnectivity(T val) {resConnectivity=val; return *this;}
        inline ReservoirInitialiser& setResRangeLow(T val) {resLow=val; updateResRange(); return *this;}
        inline ReservoirInitialiser& setResRangeHigh(T val) {resHigh=val; updateResRange(); return *this;}
        inline ReservoirInitialiser& setInConnectivity(T val) {inConnectivity=val; return *this;}
        inline ReservoirInitialiser& setInRangeLow(T val) {inLow=val; updateInRange(); return *this;}
        inline ReservoirInitialiser& setInRangeHigh(T val) {inHigh=val; updateInRange(); return *this;}
        
        virtual void init(Reservoir<T> &net) {
            
            //choose non-zero connections
            Mat<T> res = net.getRes();
            randomise<T>::randomiseMatrix(res, resConnectivity, resLow, resRange);
            
            if (net.getNIns() > 0) {
                Mat<T> &ins = net.getIns();
                randomise<T>::randomiseMatrix(ins, inConnectivity, inLow, inRange);
            }
            
            Col<std::complex<T> > eigval;
            Mat<std::complex<T> > eigvec;
            
            if (eig_gen(eigval, eigvec, res)) {
                T scaleFactor = alpha / abs((max(eigval)));
                res = res * scaleFactor;
                net.setResWeights(res);
            }else{
                throw(EVException());
            }
        }
        
    private:
        T resRange, resLow, resHigh;
        T inRange, inLow, inHigh;
        T resConnectivity, inConnectivity;
        T alpha;
    };
    
    template <typename T>
    class ReadOutInitialiser {
    public:
        ReadOutInitialiser() : fbLow(-1), fbHigh(1.0), fbConnectivity(1.0)
        {
            updateFbRange();
        }
        
        inline void updateFbRange() {fbRange = fbHigh - fbLow;}
        inline ReadOutInitialiser& setFbConnectivity(T val) {fbConnectivity=val; return *this;}
        inline ReadOutInitialiser& setFbRangeLow(T val) {fbLow=val; updateFbRange(); return *this;}
        inline ReadOutInitialiser& setFbRangeHigh(T val) {fbHigh=val; updateFbRange(); return *this;}
        
        virtual void init(ReadOut<T> &ro) {
            ro.setFeedbackOn(fbConnectivity > 0 && fbRange > 0);
            if (ro.isFeedbackOn()) {
                Mat<T> &fbWeights = ro.getFbWeights();
                randomise<T>::randomiseMatrix(fbWeights, fbConnectivity, fbLow, fbRange);
            }
        }
        
    private:
        T fbRange, fbLow, fbHigh;
        T fbConnectivity;
    };
    
    template <typename T>
    class Simulator {
    public:
        Simulator() {}
        Simulator(Reservoir<T> &_net, ReadOut<T> &_ro) {
            init(_net, _ro);
        }
        
        void init(Reservoir<T> &_net, ReadOut<T> &_ro) {
            net = &_net;
            readOuts.push_back(&_ro);
//            ro = &_ro;
            WinxU.set_size(net->getNRes());
            WxX.set_size(net->getNRes());
            WxY.set_size(net->getNRes());
            x = &net->getActivations();
            ins = &net->getIns();
            noiseVect.set_size(net->getNRes());
            feedbackScale = 1.0;
        }
       
        void addReadOut(ReadOut<T> &ro) {
            readOuts.push_back(&ro);
        }
        
        virtual inline void simulate(Col<T> &inputs) {
            simulateOneEpoch(inputs);
            for(int i=0; i < readOuts.size(); i++) {
                readOuts[i]->update();
            }
        }
        
        inline void randArray(vector<T> &vals);
        
        virtual inline void simulateOneEpoch(Col<T> &inputs) {
            net->setInputs(inputs);
            
            T noiseLevel = net->getNoise();
            if (noiseLevel > 0) {
                noiseVect.randu();
                *x = *x + ((noiseVect - 0.5) * (noiseLevel * 2.0));
            }
            
            WxX = net->getRes() * *x;
            
            if (net->getNIns() > 0) {
                WinxU = *ins * inputs;
                *x = WxX + WinxU;
            }else{
                *x = WxX;
            }

            //outs(n-1) * fb
            for(int i=0; i < readOuts.size(); i++) {
                //readOuts[i]->update();
                if (readOuts[i]->isFeedbackOn()) {
                    WxY = (readOuts[i]->getFbWeights() * feedbackScale) * readOuts[i]->getOutputs();
                    //                WxY = WxY * feedbackScale;
                    *x = *x + WxY;
                }
            }
            
            net->getActivationFunction()->process(*x);
        }
        inline Reservoir<T>* getRes() {return net;}
        inline void setFeedbackScale(T newScale) {feedbackScale = newScale;}
        inline T getFeedbackScale() {return feedbackScale;}
    protected:
        Col<T> WinxU, WxX, WxY;
        Col<T> *x;
        Mat<T> *ins;
        Reservoir<T> *net;
        vector<ReadOut<T>*> readOuts;
        Col<T> noiseVect;
        T feedbackScale;
    };

    template <typename T>
    class SimulatorLI : public Simulator<T> {
    public:
        SimulatorLI() {}
        SimulatorLI(Reservoir<T> &_net, ReadOut<T> &_ro, T leakRate) : Simulator<T>(_net, _ro), alpha(leakRate), oneMinusAlpha(1-leakRate) {
            leakyX.set_size(_net.getNRes());
            leakyX.fill(0);
        }
        
        inline void simulate(Col<T> inputs) {
            //(1-alpha)Xn
            leakyX = *this->x * alpha;
            this->simulateOneEpoch(inputs);
            *this->x = (oneMinusAlpha * *this->x) + leakyX;
            for(int i=0; i < this->readOuts.size(); i++) {
                this->readOuts[i]->update();
            }
        }
        
        inline void setLeakRate(T newRate) {alpha = 1.0 - newRate;}
        inline T getLeakRate() {return 1.0 - alpha;}
    protected:
        T alpha, oneMinusAlpha;
        Col<T> leakyX;
    };
    
    
    class TrainingException: public std::runtime_error { public: TrainingException(): std::runtime_error("An error occured during training.\n") {} };

    template<typename T>
    class TrainerLeastSquares {
    public:
        
        /**
         *Construct the trainer. 
         * expects inputs and desired outputs with examples set out rowwise in the matrices
         * e.g. for res with 2 ins, 1 out, 
         * inputs
         *  x1a, x1b,
         *  x2a, x2b
         *  ...
         * outputs
         *  y1,
         *  y2
         *  ...
         
         */
        TrainerLeastSquares(Simulator<T> *_sim, ReadOut<T> *_rOut, Mat<T> &inputsMatrix, Mat<T> &desiredOutputsMatrix, const uint _washout) : sim(_sim), trainSize(desiredOutputsMatrix.n_rows), washout(_washout), ro(_rOut) {
            inputData = &inputsMatrix;
            outputData = &desiredOutputsMatrix;
            stateSize = sim->getRes()->getNRes();
            if (ro->insAreMappedToOuts()) {
                stateSize += sim->getRes()->getNIns();
            }
            collectedStatesCount = (trainSize - washout);
            extStates.set_size(collectedStatesCount, stateSize);
        }
        
        void train() {
            
            runAndCollectActivations();
            prepareTeacherSignal();
            calculateOutputWeights();
        }

        
        void runAndCollectActivations() {
            int inputIdx=0;
            //run to washout
            for(int i=0; i < washout; i++) {
                Col<T> dataIn = trans(inputData->row(inputIdx));
                sim->simulate(dataIn);
                if (ro->isFeedbackOn()) {
                    Col<T> fbData = trans(outputData->row(inputIdx));
                    ro->teacherForce(fbData);
                }
                inputIdx++;
            }
            //harvest states
            for(int i=0; i < collectedStatesCount; i++) {
                //run an iteration of the system
                Col<T> dataIn = trans(inputData->row(inputIdx));
                sim->simulate(dataIn);
                if (ro->isFeedbackOn())
                    ro->teacherForce(trans(outputData->row(inputIdx)));
                
                //copy states
                extStates(i, span(0,sim->getRes()->getNRes() - 1)) = trans(sim->getRes()->getActivations());
                if (ro->insAreMappedToOuts() && sim->getRes()->getNIns() > 0) {
                    extStates(i, span(sim->getRes()->getNRes(), sim->getRes()->getNRes() + sim->getRes()->getNIns() - 1)) = trans(sim->getRes()->getInputs());
                }
                inputIdx++;
            }
            
        }
        
        void prepareTeacherSignal() {
            //inverse-activate desired outs
            outputDataInv = outputData->rows(washout, outputData->n_rows-1);
            ro->getActivationFunction()->invProcess(outputDataInv);            
        }
        
        void calculateOutputWeights() {
            solvedWeights.set_size(stateSize, ro->getSize());
            solveOutputWeights();
            ro->setOutputWeights(solvedWeights);
        }
        
        Mat<T> &getExtStates() {return extStates;}

        virtual void solveOutputWeights() {
            
            solvedWeights = solve(extStates, outputDataInv);
            
        }

    protected:
        

        
        Mat<T> *inputData;
        Mat<T> *outputData;
        uint trainSize;
        uint collectedStatesCount;
        uint washout;
        Mat<T> extStates;
        Simulator<T> *sim;
        ReadOut<T> * ro;
        int stateSize;
        Mat<T> solvedWeights;
        Mat<T> outputDataInv;
    };
    
    template<typename T>
    class TrainerPseudoInverse : public TrainerLeastSquares<T> {
    public:
        TrainerPseudoInverse(Simulator<T> *_sim, ReadOut<T> *_rOut, Mat<T> &inputsMatrix, Mat<T> &desiredOutputsMatrix, const uint _washout) : TrainerLeastSquares<T>(_sim, _rOut, inputsMatrix, desiredOutputsMatrix, _washout) {
            
        }

        void solveOutputWeights() {
            this->solvedWeights = pinv(this->extStates) * this->outputDataInv;
        }
    };
    
    template<typename T>
    class TrainerRidgeRegression : public TrainerLeastSquares<T> {
    public:
        TrainerRidgeRegression(Simulator<T> *_sim, ReadOut<T> *_rOut, Mat<T> &inputsMatrix, Mat<T> &desiredOutputsMatrix, const uint _washout, T _alpha) : TrainerLeastSquares<T>(_sim, _rOut, inputsMatrix, desiredOutputsMatrix, _washout), alpha(_alpha) {
        }
        
        void solveOutputWeights() {
            Mat<T> P = trans(this->extStates) * this->outputDataInv;
            Mat<T> R = trans(this->extStates) * this->extStates;
//            this->solvedWeights = inv(R + (alpha * alpha * eye(R.n_cols, R.n_cols))) * P;
//            this->solvedWeights = (R + (alpha * alpha * eye(R.n_cols, R.n_cols))).i() * P;
            Mat<T> idm(R.n_cols, R.n_cols);
            idm.eye();
            this->solvedWeights = (R + (alpha * alpha * idm)).i() * P;
        }
    protected:
        T alpha;
    };


    

}

#endif
