  __  __                ____                                          
 |  \/  |              |  _ \                                         
 | \  / |  _ __        | |_) |  _ __    ___   __      __  _ __    ___ 
 | |\/| | | '__|       |  _ <  | '__|  / _ \  \ \ /\ / / | '_ \  / __|
 | |  | | | |     _    | |_) | | |    | (_) |  \ V  V /  | | | | \__ \
 |_|  |_| |_|    (_)   |____/  |_|     \___/    \_/\_/   |_| |_| |___/
                                                                      

* Producing kNN Train Set

    python produce_features.py

* Compiling kNN

    g++ -std=c++11 -O3 -DNF=NUMBER_OF_FEATURES kNN.cpp -pthread

* Running kNN validation.

    ./a.out K pooled feature_file
