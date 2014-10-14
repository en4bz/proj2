  __  __                ____                                          
 |  \/  |              |  _ \                                         
 | \  / |  _ __        | |_) |  _ __    ___   __      __  _ __    ___ 
 | |\/| | | '__|       |  _ <  | '__|  / _ \  \ \ /\ / / | '_ \  / __|
 | |  | | | |     _    | |_) | | |    | (_) |  \ V  V /  | | | | \__ \
 |_|  |_| |_|    (_)   |____/  |_|     \___/    \_/\_/   |_| |_| |___/
                                                                      

* Producing kNN Train Set

    $ python produce_features.py

This script will create feature vectors from the train set and output them to a
text file. To change the length of the feature vectors edit the MAX variable in
produce_features.py

A power of 2 for max is suggested.

    $ paste labels_file features.out > features.train

The feature file does not contain any labels. Use the paste command to join the
labels file and the features file into one and output them into a new file.

* Compiling kNN

    $ g++ -std=c++11 -O3 -march=native -DNF=NUMBER_OF_FEATURES kNN.cpp -pthread

Requires a C++11 compiler (g++ 4.8, clang 3.5)

* Running kNN validation.

    $ ./a.out K pooled feature_file

Where K is the number of neighbours to to use and feature_file is the output
of feature creation step. If everything works properly a.out should print 
"Loaded 96210 training examples" to stderr when it starts running.

This will take anywhere from 30 minutes to 3hours of CPU time depending on the
feature length. Divide the CPU time by the number of cores your machine has to
get an esitmate of the true run time.
