#include <immintrin.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>
#include <bitset>
#include <thread>
#include <future>
#include <algorithm>
#include <unordered_map>

#include "threadpool.hpp"
        
typedef std::unordered_map<std::string,size_t> counter; 

template <size_t N>
class kNN {
    std::vector<std::string> labels;
    std::vector<std::bitset<N>> bits;
    std::vector<std::bitset<N>> test_set;
public:
    kNN () = default;

    void train(const std::string& feature_file){
        std::ifstream in(feature_file);
        while(in.good()){
            std::string label;
            std::bitset<N> temp;
            in >> label;
            in >> temp;
            labels.push_back(label);
            bits.push_back(temp);
        }
        labels.pop_back(); // this is a newline
        bits.pop_back(); // this is a newline
        std::cerr << "Loaded " << labels.size() << " training examples." << std::endl;
    }

    void load_test_set(const std::string& test_file){
        std::ifstream in(test_file);
        int id;
        while(in.good()){
            std::bitset<N> temp;
            in >> id;
            in >> temp;
            test_set.push_back(temp);
        }
        test_set.pop_back(); // this is an newline
        std::cerr << "Loaded " << test_set.size() << " test examples." << std::endl;
    }

    bool validate_one(const size_t cmp) const {
        size_t max = 0;
        size_t index = 0;
        const size_t num_examples = labels.size();
        for(size_t i = 0; i < num_examples; i++){
            if(cmp == i) continue;
            const size_t dist = (bits[cmp] & bits[i]).count();
            if(dist > max){
                max = dist;
                index = i;
            }
        }
        return labels[cmp] == labels[index];
    }

    std::string predict(const size_t cmp, const size_t K) const{
        const size_t num_examples = labels.size();
        std::vector<char> dists(num_examples, -1);
        for(size_t i = 0; i < num_examples; i++){
            if(i == cmp) continue;
            dists[i] = (bits[cmp] & bits[i]).count();

        }
        counter counts;
        for(size_t i = 0; i < K; i++){
            auto iter = std::max_element(dists.cbegin(), dists.cend());
            size_t idx = std::distance(dists.cbegin(), iter);
            dists[idx] = -1; // Remove the top result.
            counts[labels[idx]] += (K-i);
        }
        auto iter = std::max_element(counts.cbegin(), counts.cend(),
                [](const counter::value_type& a, const counter::value_type& b){ return a.second < b.second; });

        return iter->first;
    }

    bool validate_N(const size_t cmp, const size_t K) const{
        const size_t num_examples = labels.size();
        std::vector<char> dists(num_examples, -1);
        for(size_t i = 0; i < num_examples; i++){
            if(i == cmp) continue;
            dists[i] = (bits[cmp] & bits[i]).count();

        }
        counter counts;
        for(size_t i = 0; i < K; i++){
            auto iter = std::max_element(dists.cbegin(), dists.cend());
            size_t idx = std::distance(dists.cbegin(), iter);
            dists[idx] = -1; // Remove the top result.
            counts[labels[idx]] += (K-i);
        }
        auto iter = std::max_element(counts.cbegin(), counts.cend(),
                [](const counter::value_type& a, const counter::value_type& b){ return a.second < b.second; });

        return labels[cmp] == iter->first;
    }


    void validate(const size_t num_threads, const size_t K) const{
        const size_t num_examples = labels.size();
        for(size_t i = 0; i < num_examples; i += num_threads){
            std::vector<std::future<std::string>> futs;
            for(size_t j = 0; j < num_threads && i + j < num_examples; j++){
                futs.push_back(std::async(std::launch::async, &kNN<N>::predict, this, i + j, K));
            }
            for(size_t j = 0; j < num_threads; j++){
                std::cout << labels[i+j] << "," << futs[j].get() << std::endl;
                
            }
        }
        return;
    }
    
    double validate_pooled(const size_t K) const{
        int right = 0;
        const size_t num_examples = labels.size();
        threadpool tp;
        std::vector<std::future<bool>> futs;
        for(size_t i = 0; i < num_examples; i++){
            futs.push_back(tp.enqueue(&kNN<N>::validate_N, this, i, K));
        }
        for(auto& fut : futs)
            if(fut.get()) right++;

        return right / (double) num_examples;
    }

    std::string test_N(const std::bitset<N> cmp, const size_t n) const{
        const size_t num_examples = labels.size();
        std::vector<char> dists(num_examples, -1);
        for(size_t i = 0; i < num_examples; i++){
            dists[i] = (cmp & bits[i]).count();
        }
        counter counts;
        for(size_t i = 0; i < n; i++){
            auto iter = std::max_element(dists.cbegin(), dists.cend());
            size_t idx = std::distance(dists.cbegin(), iter);
            dists[idx] = -1; // Remove the top result.
            counts[labels[idx]] += 1;
        }
        auto iter = std::max_element(counts.cbegin(), counts.cend(),
                [](const counter::value_type& a, const counter::value_type& b){ return a.second < b.second; });

        return iter->first;
    }

    void cross_validate(const size_t n_folds){
        
    }

    void test(const size_t num_threads, const size_t K) const{
        std::cout << "\"id\",\"category\"" << std::endl;
        const size_t test_size = test_set.size();
        for(size_t i = 0; i < test_size; i += num_threads){
            std::vector<std::future<std::string>> futs;
            for(size_t j = 0; j < num_threads && i + j < test_size; j++){
                futs.push_back(std::async(std::launch::async, &kNN<N>::test_N, this, test_set[i + j], K));
            }
            for(size_t j = 0; j < futs.size(); j++){
                std::cout << '"' << i + j << "\",\"" << futs[j].get() << '"' << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]){
    size_t K = 0;
    if(argc > 2){
        K = std::stoul(argv[1]);
    }
    else{
        return 2;
    }
    //load data 
    kNN<4096> classifier;
    classifier.train(argv[3]);
    if( std::string(argv[2]) == "test") {
        classifier.load_test_set("test.out");
        classifier.test(1,K);
    }
    else if(std::string(argv[2]) == "pooled"){
        std::cout << classifier.validate_pooled(K);
    }
    else {
        classifier.validate(8,K);
    }
    return 0;
}
