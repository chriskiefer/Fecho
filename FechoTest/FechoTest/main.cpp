#include <iostream>
#include "fecho.h"
#include <Accelerate/Accelerate.h>
#include <fstream>

using namespace Fecho;

ofstream pyCode;
dSFMTRand mtrnd;

//see http://books.nips.cc/papers/files/nips15/AA14.pdf
void narma10(vector<float> &input, vector<float> &narma) {
    narma.resize(input.size());
    for(int i=10; i < narma.size(); i++) {
        narma[i] = 0.3*narma[i-1] + 0.05*narma[i-1]*(narma[i-1]+narma[i-2]+narma[i-3]+narma[i-4]+narma[i-5]+narma[i-6]+narma[i-7]+narma[i-8]+narma[i-9]+narma[i-10]) + 1.5*input[i-10]*input[i-1] + 0.1;
    }
}

void sineWaveTest() {
    clock_t ts = clock();

    ActivationFunctionTanh<float> resAct;
    Reservoir<float> net(0, 20, &resAct);
    net.setInScale(0).setInShift(0).setNoise(0.00001);
    
    ActivationFunctionLinear<float> roAct;
    ReadOut<float> ro(net, 1, &roAct);
    ro.setMapInsToOuts(true);
    
    try {
        Initialiser<float> netInit;
        netInit.setResRangeLow(-1).setResRangeHigh(1).setResConnectivity(0.15)
        .setSpectralRadius(0.9)
        .setInConnectivity(0.0).setInRangeLow(-1).setInRangeHigh(1)
        .setFbConnectivity(1.0).setFbRangeLow(-0.1).setFbRangeHigh(0.7)
        .init(net, ro);
        net.dump();
    } catch (Initialiser<float>::EVException e) {
        cout << e.what() << endl;
    }
    
    
    
//    SimulatorLI<float> sim(net, ro, 0.9);
    Simulator<float> sim(net, ro);
    
    vector<float> ins;
    ins.push_back(1.0);
    
    vector<float> trainIn, trainOut;
    int trainSize = 300;
    trainIn.resize(0);
    trainOut.resize(trainSize);
    for(int i=0; i < trainSize; i++) {
        float r = 0.5 *sin(i/4.0);
        //or something more complex...
//        float r = (0.5 *sin(i/4.0)) + (0.1 * sin(i/7.0)) + (0.3 *sin(i));
//        r = pow(r,3);
        trainOut[i] = r;
    }
    TrainerPseudoInverse<float> tr(&sim, &ro, trainIn, trainOut, trainSize, 100);
    cout << "Training...\n";
    tr.train();
    cout << "processed in " << ((clock() - ts) / (double) CLOCKS_PER_SEC) << " secs\n";
    
    cout << "Running trained ESN\n";
    int runSize = 1000;
    vector<float> res(runSize);
//    net.resetStates();
    for(int i=0; i < runSize; i++) {
        sim.simulate(&trainIn[0]);
        res[i] = ro.getOutputs()[0];
    }
    pyCode << "simIn = array([";
//    for(int i=0; i < runSize; i++) {
//        cout << trainIn[i] << ",";
//        pyCode << trainIn[i] << ",";
//    }
//    cout << endl;
    pyCode << "])\n";
    cout << "Result: \n";
    pyCode << "simOut = array([";
    for(int i=0; i < runSize; i++) {
        cout << res[i] << ",";
        pyCode << res[i] << ",";
    }
    pyCode << "])\n";
    cout << endl;
    
    ts = clock() - ts;
    double timeSecs =  ts / (double) CLOCKS_PER_SEC;
    cout << "processed in " << timeSecs << " secs\n";

}

int main(int argc, const char * argv[])
{
    pyCode.open("/tmp/esn.py", ios_base::out);
    sineWaveTest();
    pyCode.close();
    
    
    
    //gels test
    
//    cout << endl << endl;
    //    float a[5*3] = {1,1,1,2,3,4,3,5,2,4,2,5,5,4,3};
    //    float b[5*2] = {-10,-3,12,14,14,12,16,16,18,16};
    //    
    //    vector<float> acm(15);
    //    Math::rowMajorToColMajor(a, 3, 5, &acm[0]);
    //    for(float &v: acm) {
    //        cout << v << ",";
    //    }
    //    cout << endl;
    //
    //    vector<float> bcm(10);
    //    Math::rowMajorToColMajor(b, 2, 5, &bcm[0]);
    //    for(float &v: bcm) {
    //        cout << v << ",";
    //    }
    //    cout << endl;
    //    
    //    char trans = 'N';
    //    int m=5;
    //    int n=3;
    //    int nrhs = 2;
    //    int lda = 5;
    //    int ldb = 5;
    //    
    //    int lwork = -1;
    //    float wkopt;
    //    int info;
    //    
    //    Math::gels(&trans, &m, &n, &nrhs, &acm[0], &lda, &bcm[0], &ldb, &wkopt, &lwork, &info);
    //    lwork = (int)wkopt;
    //    vector<float> work(lwork);
    //    Math::gels(&trans, &m, &n, &nrhs, &acm[0], &lda, &bcm[0], &ldb, &work[0], &lwork, &info);
    //    cout << "info: " << info << endl;
    //    for(float &v: bcm) {
    //        cout << v << ",";
    //    }
    //    cout << endl;
    //    
    //    vector<float> result(nrhs * n);
    //    for(int i=0; i < nrhs; i++) {
    //        for(int j=0; j < n; j++) {
    //            result[(i*n) + j] = bcm[(i*m) + j];
    //        }
    //    }
    //    for(float &v: result) {
    //        cout << v << ",";
    //    }
    //    cout << endl << "gelss:\n";
    
    //now try with gelss
    //re-init bcm
    //    Math::rowMajorToColMajor(b, 2, 5, &bcm[0]);
    //    for(float &v: bcm) {
    //        cout << v << ",";
    //    }
    //    cout << endl;
    //    vector<float> s(min(m,n));
    //    float rcond = 0.01;
    //    int rank;
    //    lwork = -1;
    //    Math::gelss(&m, &n, &nrhs, &acm[0], &lda, &bcm[0], &ldb, &s[0], &rcond, &rank, &wkopt, &lwork, &info);
    //    lwork = (int)wkopt;
    //    vector<float> work2(lwork);
    //    Math::gelss(&m, &n, &nrhs, &acm[0], &lda, &bcm[0], &ldb, &s[0], &rcond, &rank, &work2[0], &lwork, &info);
    //    cout << "info: " << info << endl;
    //    cout << "Rank: " << rank << endl;
    //    for(float &v: bcm) {
    //        cout << v << ",";
    //    }
    //    cout << endl;
    //    for(int i=0; i < nrhs; i++) {
    //        for(int j=0; j < n; j++) {
    //            result[(i*n) + j] = bcm[(i*m) + j];
    //        }
    //    }
    //    for(float &v: result) {
    //        cout << v << ",";
    //    }
    
    
    
    
    return 0;
}

