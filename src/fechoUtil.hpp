//
//  fechoUtil.h
//  libFecho
//
//  Created by Chris Kiefer on 01/11/2012.
//

#ifndef libFecho_fechoUtil_h
#define libFecho_fechoUtil_h

#include <sstream>
#include <iostream>
#include <vector>
#include <algorithm>

namespace Fecho {

    template <typename T>
    class MultiStreamRecorder {
    public:
        void setChannels(int num) {
            streams.resize(num);
            clear();
        }
        void addFrame(arma::Col<T> &vals) {
            for(int i=0; i < streams.size(); i++) {
                streams[i].push_back(vals(i));
            }
        }
        void clear() {
            for(int i=0; i < streams.size(); i++) {
                streams[i].clear();
            }
        };
        
        std::vector<std::vector<T> > streams;
        
        std::string dumpToPyCode() {
            std::stringstream s;
            s << "x = array([";
            for(int i=0; i < streams.size(); i++) {
                s << "[";
                for(int j=0; j < streams[i].size(); j++) {
                    s << streams[i][j] << ",";
                }
                s << "],";
            }
            s << "]);\n";
            return s.str();
        }
    };
    
    template<typename T>
    std::string toPyCode(std::string name, arma::Col<T> &x) {
        std::stringstream s;
        s << name << " = array([";
        for(int i=0; i < x.size(); i++) {
            s << x(i) << ",";
        }
        s << "])\n";
        return s.str();
    }

    template<typename T>
    std::string toPyCode(std::string name, arma::Mat<T> &x) {
        std::stringstream s;
        for(int i=0; i < x.n_rows; i++) {
            s << name << i << " = array([";
            for(int j=0; j < x.n_cols; j++) {
                s << x(i,j) << ",";
            }
            s << "])\n";
            
        }
        return s.str();
    }

    template<typename T>
    class error {
    public:
        static T MSE(Col<T> &seq1, Col<T> &seq2) {
            Col<T> diff = seq2 - seq1;
            T sqError = sum(square(diff));
            return sqError / static_cast<T>(seq1.n_rows);
        }
        static T RMSE(Col<T> &seq1, Col<T> &seq2) {
            return sqrt(MSE(seq1, seq1));
        }
        static T calc(Col<T> &seq1, Col<T> &seq2) {
            T xrange = std::max(seq1.max(), seq2.max()) - std::min(seq1.min(), seq2.min());
            T nmrse = xrange == 0 ? 0 : RMSE(seq1, seq2) / xrange;
            return nmrse;
        }

    };
    
    template<typename T>
    class randomise {
    public:
        static void randomiseMatrix(Mat<T> &mat, float connectivity, float low, float range) {
            //randomise
            mat.randu();
            //map to range
            mat = low + (range * mat);
            int matDimC = mat.n_cols;
            int matDimR = mat.n_rows;
            //reshape to one dimension
            mat.reshape(mat.n_cols * mat.n_rows, 1);
            //zero out some elements
            if (connectivity < 1) {
                mat.rows(floor(mat.n_rows * connectivity), mat.n_rows-1).fill(0);
            }
            //shuffle
            mat = shuffle(mat);
            //back into shape
            mat.reshape(matDimR, matDimC);
        }
    };
    
//    template<typename T>
//    void zeroCrossings(Col<T> signal, float &period, float &variance) {
//        int lastZeroCrossingIdx = -1;
//        Col<T> periods;
//        for(int i=1; i < signal.n_rows; i++) {
//            
//        }
//    }
}

#endif
