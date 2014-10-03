#include <immintrin.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <bitset>
#include <thread>
#include <future>

template<size_t N>
size_t hamming_distance(const std::bitset<N>& a, const std::bitset<N>& b){
    return (a & b).count();
}

template<size_t N>
bool compute(const size_t cmp, const std::vector<std::bitset<N>>& bits, const std::vector<std::string>& labels){
    size_t max = 0;
    size_t index = 0;
    for(size_t i = 0; i < labels.size(); i++){
        if(cmp == i) continue;
        //const size_t dist = hamming_distance(bits[cmp], bits[i]);
        const size_t dist = hamming_distance(bits.at(cmp), bits.at(i));
        if(dist > max){
            max = dist;
            index = i;
        }
    }
    return labels[cmp] == labels[index];
}

int main(int argc, char* argv[]){
    int num_features = 0;
    const int FEAT_LEN = 2048;
    if(argc > 1){
        num_features = atoi(argv[1]);
    }
    else{
        return 2;
    }
    //load data
    std::vector<std::string> labels(num_features);
    std::vector<std::bitset<FEAT_LEN>> bits(num_features);
    for(int i = 0; i  < num_features; i++){
        std::cin >> labels[i];
        std::cin >> bits[i];
    }
    
    int right = 0;
    const size_t num_threads = 8;
    for(size_t i = 0; i < num_features; i += num_threads){
        std::future<bool> futs[num_threads];
        for(size_t j = 0; j < num_threads; j++){
            futs[j] = std::async(std::launch::async, compute<FEAT_LEN>, i + j, std::ref(bits), std::ref(labels));
        }
        for(auto& fut : futs){
            if(fut.get()){
                right++;
            }
        }
    }
    std::cout << (right / (double) num_features) << std::endl;
    return 0;
}
