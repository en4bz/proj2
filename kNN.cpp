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

template <size_t N>
class kNN {
    const size_t num_examples;
    std::vector<std::string> labels;
    std::vector<std::bitset<N>> bits;
    std::vector<std::bitset<N>> test_set;
public:
    kNN (const size_t num)
        : num_examples(num), labels(num), bits(num) {}

    void train(const std::string& feature_file){
        std::ifstream in(feature_file);
        for(size_t i = 0; i < num_examples; i++){
            in >> labels[i];
            in >> bits[i];
        }
    }

    void load_test_set(const std::string& test_file){
        std::ifstream in(test_file);
        int id;
        while(in.good()){
            in >> id;
            std::bitset<N> temp;
            in >> temp;
            test_set.push_back(temp);
        }
    }

    bool validate_one(const size_t cmp) const {
        size_t max = 0;
        size_t index = 0;
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

    bool validate_N(const size_t cmp, const size_t n) const{
        std::vector<char> dists(num_examples, -1);
        for(size_t i = 0; i < num_examples; i++){
            if(i == cmp) continue;
            dists[i] = (bits[cmp] & bits[i]).count();

        }
        typedef std::unordered_map<std::string,size_t> counter; 
        counter counts;
        for(size_t i = 0; i < n; i++){
            auto iter = std::max_element(dists.cbegin(), dists.cend());
            size_t idx = std::distance(dists.cbegin(), iter);
            dists[idx] = -1; // Remove the top result.
            counts[labels[idx]] += 1;
        }
        auto iter = std::max_element(counts.cbegin(), counts.cend(),
                [](const counter::value_type a, const counter::value_type b){ return a.second < b.second; });

        return labels[cmp] == iter->first;;
    }


    double validate(const size_t num_threads, const size_t num_n) const{
        int right = 0;
        for(size_t i = 0; i < num_examples; i += num_threads){
            std::future<bool> futs[num_threads];
            for(size_t j = 0; j < num_threads; j++){
                futs[j] = std::async(std::launch::async, &kNN<2048>::validate_N, this, i + j, num_n);
            }
            for(auto& fut : futs){
                if(fut.get()){
                    right++;
                }
            }
        }
        return right / (double) num_examples;
    }
    
    std::string test_N(const std::bitset<N> cmp, const size_t n) const{
        std::vector<char> dists(num_examples, -1);
        for(size_t i = 0; i < num_examples; i++){
            dists[i] = (cmp & bits[i]).count();
        }
        typedef std::unordered_map<std::string,size_t> counter; 
        counter counts;
        for(size_t i = 0; i < n; i++){
            auto iter = std::max_element(dists.cbegin(), dists.cend());
            size_t idx = std::distance(dists.cbegin(), iter);
            dists[idx] = -1; // Remove the top result.
            counts[labels[idx]] += 1;
        }
        auto iter = std::max_element(counts.cbegin(), counts.cend(),
                [](const counter::value_type a, const counter::value_type b){ return a.second < b.second; });

        return iter->first;
    }

    void test(const size_t num_threads, const size_t num_n) const{
        std::cout << "\"id\",\"category\"" << std::endl;
        for(size_t i = 0; i < test_set.size(); i += num_threads){
            std::future<std::string> futs[num_threads];
            for(size_t j = 0; j < num_threads; j++){
                futs[j] = std::async(std::launch::async, &kNN<N>::test_N, this, test_set[i + j], num_n);
            }
            for(size_t j = 0; j < num_threads; j++){
                std::cout << '"' << i + j << "\",\"" << futs[j].get() << '"' << std::endl;
            }
        }
    }
};

int main(int argc, char* argv[]){
    size_t num_features = 0;
    if(argc > 1){
        num_features = std::stoul(argv[1]);
    }
    else{
        return 2;
    }
    //load data 
    kNN<2048> classifier(num_features);
    classifier.train("features.out");
    classifier.load_test_set("test.out");
    //double valid = classifier.validate(8,25);
    //std::cout << valid << std::endl;
    classifier.test(10,25);

    return 0;
}
