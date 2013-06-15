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

//    template<typename T>
//    class MSE {
//    public:
//        static T calc(vector<T> &seq1, vector<T> &seq2) {
//            vector<T> diff(seq1.size());
//            Math::vecsub(&seq1[0], 1, &seq2[0], 1, &diff[0], 1, seq1.size());
//            Math::vecsq(&diff[0], 1, &diff[0], 1, diff.size());
//            T mean;
//            Math::vecmean(&diff[0], 1, &mean, diff.size());
//            return mean;
//        }
//    };
}

#endif
