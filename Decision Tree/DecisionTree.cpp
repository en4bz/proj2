#include<iostream>
#include<fstream>
#include<cstring>
#include<set>
#include<map>
#include<algorithm>
#include<cstdlib>
#include<exception>
#include<vector>
#include<cmath>

using namespace std;

typedef unsigned long long ulint;
const int trainSize = 96200;//data points to train on
int testSize = 10000;//the leave one out value

//Main Parameters
int featSize = 400;//Number of features
int maxDepth = 40;//Maximum depth of tree
//const int featDim = 1 + (featSize/64 + 1);//increase for every 64 features
const int featDim = 160;

char path[] = "train_input.csv"; //File formatted to have id, category, abstract
//char path[] = "/home/ubuntu/DecTree/train_input.csv";
//char path[] = "E:/Workspace/AML/DecisionTree/train_input.csv";

struct dict{
    string word;
    int count;
}sortedDict[30000];

struct compare{
	bool operator () (const dict& a,const dict& b)const{
		return a.count > b.count;
	}
}cmp;

//Determines entropy
double findIG( int a, int b, int c, int d ){
    int ar[4];
    ar[0] = a, ar[1] = b, ar[2] = c, ar[3] = d;
    int id;
    if( ar[0] > ar[1] ) id = 0;
    else id = 1;
    if( ar[id] < ar[2] )
        id = 2;
    if( ar[id] < ar[3] )
        id = 3;
    double total = ar[0] + ar[1] + ar[2] + ar[3];
    //if( total == 0 ) return 0;
    double entropy = 0;
    for( int i = 0; i < 4; i++ )
        entropy += ( -(ar[i]/total) * log2(ar[i]/total) );
    return entropy;
}

int findMaxCat( int ar[] ){
    int id;
    if( ar[0] > ar[1] ) id = 0;
    else id = 1;
    if( ar[id] < ar[2] )
        id = 2;
    if( ar[id] < ar[3] )
        id = 3;
    return id;
}

struct treeNode{
    bool leaf;
    int attNum;
    ulint category; //maybe fix this
    treeNode *left, *right;
    treeNode(){
        leaf = false;
        attNum = 0;
        category = 0;
        left = right = NULL;
    }
};

//Parameters - Parent node, training matrix, size of matrix, available features, depth of node
void makeDecisionTree( treeNode* decisionTree, ulint train[][featDim], int row, ulint ft[], int depth ){
    //Initialize
    int col = featSize;
    ulint feat[featDim - 1];
    for( int i = 0; i < featDim - 1; i++ )
        feat[i] = ft[i];

    bool sameVal = true;
    for( int i = 1; i < row; i++ ){
        if( train[i][0] != train[i - 1][0] ){
            sameVal = false;
            break;
        }
    }

    //Breaking Conditions  - when all y value same or depth of tree exceeds limit
    if( sameVal == true || row <= 1 || depth >= maxDepth ){
        decisionTree->leaf = true;
        decisionTree->left = decisionTree->right = NULL;
        decisionTree->attNum = 0;
        if( depth > maxDepth ){
            int cat[4] = {0};
            for( int i = 0; i < row; i++ ) cat[train[i][0]]++;
            decisionTree->category = findMaxCat(cat);
        }
        else
            decisionTree->category = train[0][0];
        return;
    }

    //Choose best attribute using information gain
    double maxGain = -20;
    int minErrorAtt = 0;

    //For each attribute
    for( int i = 0; i < col; i++ ){//starts from 0th bit of 2nd element in train
        //find error percentage
        if( ( feat[0 + (i/64)] & ((ulint)1<<(i%64)) ) != 0 )//If attribute already used
            continue;

        double parentEntropy = 0;
        double childEntropy = 0;
        int category[2][4];
        memset( category, 0, sizeof(category) );

        //Calculate entropy for each branch
        for( int x = 0; x < 2; x++ ){//for each boolean value

            for( int j = 0; j < row; j++ ){//For each example
                ulint var = ulint(train[j][1 + (i/64)]);
                int check;
                if( x == 0 ){//check for bit is 0
                    if(  ( var & ((ulint)1<<(i%64)) ) != 0 ) continue;
                }
                else{//check for bit is 1
                    if(  ( var & ((ulint)1<<(i%64)) ) == 0 ) continue;
                }

                category[x][train[j][0]]++;
            }
            double total = category[x][0] + category[x][1] + category[x][2] + category[x][3];
            if( total == 0 ) continue;
            childEntropy += ( (total/row) * findIG( category[x][0], category[x][1], category[x][2], category[x][3] ) );
        }

        //Calculate parent entropy
        for( int j = 0; j < 4; j++ ){
            double totalCatOcc = category[0][j] + category[1][j];
            if( totalCatOcc == 0 ) continue;
            parentEntropy += ( -(totalCatOcc/row) * log2(totalCatOcc/row) );
        }

        double gain = parentEntropy - childEntropy;
        if( gain > maxGain ){
            maxGain = gain;
            minErrorAtt = i;
        }
    }
    //Find number of examples for each branch
    int numZero = 0, numOne = 0;
    for( int i = 0; i < row; i++ ){
        if( ( train[i][1 + (minErrorAtt/64)] & ((ulint)1<<(minErrorAtt%64)) ) == 0 )
            numZero++;
        else
            numOne++;
    }
    //Create new matrices for the branches
    ulint(*zeroSet)[featDim] = new ulint[numZero + 1][featDim];
    ulint(*oneSet)[featDim] = new ulint[numOne + 1][featDim];

    //Copy values from the table where best attribute is 0 to a new array and leave out this attribute
    int zeroSetPtr = 0, oneSetPtr = 0;

    for( int i = 0; i < numZero + 1; i++ ) for( int j = 0; j < featDim; j++ ) zeroSet[i][j] = 0;
    for( int i = 0; i < numOne + 1; i++ ) for( int j = 0; j < featDim; j++ ) oneSet[i][j] = 0;

    for( int i = 0; i < row; i++ ){
        if( ( train[i][1 + (minErrorAtt/64)] & ((ulint)1<<(minErrorAtt%64)) ) == 0 ){
            zeroSet[zeroSetPtr][0] = train[i][0];
            for( int j = 0; j < col; j++ ){
                if( (train[i][1 + (j/64)] & ((ulint)1<<(j%64))) != 0 )
                    zeroSet[zeroSetPtr][1 + (j/64)] |= ((ulint)1<<(j%64));
            }
            zeroSetPtr++;
        }
        else{
            oneSet[oneSetPtr][0] = train[i][0];
            for( int j = 0; j < col; j++ ){
                if( (train[i][1 + (j/64)] & ((ulint)1<<(j%64))) != 0 )
                    oneSet[oneSetPtr][1 + (j/64)] |= ((ulint)1<<(j%64));
            }
            oneSetPtr++;
        }
    }

    //Set this feature as used
    feat[0 + (minErrorAtt/64)] |= ((ulint)1<<(minErrorAtt%64));

    decisionTree->attNum = minErrorAtt;

    //Check if attributes value is same for all examples. Terminate if so.
    if( numOne == 0 || numZero == 0 ){
        decisionTree->leaf = true;
        int cat[4] = {0};
        for( int i = 0; i < row; i++ ) cat[train[i][0]]++;
        decisionTree->category = findMaxCat(cat);
        bool allOne = true;
    }
    //Create child nodes
    else{
        decisionTree->leaf = false;

        decisionTree->left = new treeNode();
        makeDecisionTree( decisionTree->left, oneSet, numOne, feat, depth + 1 );

        decisionTree->right = new treeNode();
        makeDecisionTree( decisionTree->right, zeroSet, numZero, feat, depth + 1 );
    }
    delete(zeroSet);
    delete(oneSet);
}

string catName[4] = {"cs", "math", "stat", "physics"};

/*
vector<string> displayTree[5];
void printTree( treeNode* node, int depth ){
    string word = sortedDict[node->attNum].word;
    string child = "";
    child = "(";
    if( node->left != NULL ) child.append("L");
    if( node->right != NULL ) child.append("R");
    child.append(")");
    word.append(child);
    if( node->leaf )
        word.append(catName[node->category]);
    displayTree[depth].push_back(word);
    if( node->leaf )
        return;
    else{
        if( node->left != NULL )
            printTree(node->left, depth + 1);
        if( node->right != NULL )
            printTree(node->right, depth + 1);
    }
}*/

//Find category of a test example
int categorize( string x, treeNode* node ){
    string word = sortedDict[node->attNum].word;
    if( node->leaf )
        return node->category;
    if( x.find(word) != string::npos ){
        if( node->left != NULL )
            return categorize( x, node->left );
    }
    else{
        if( node->right != NULL )
            return categorize( x, node->right );
    }
    return 2;//default category
}


int main( int argc, char* argv[] ){
    int i, j;
    //To send arguments from commandline or shell script use the lines below:
    //featSize = atoi(argv[1]);
    //maxDepth = atoi(argv[2]);
    //int testStart = atoi(argv[3]);
    featSize = 70;
    maxDepth = 30;
    int testStart = 0;//the starting position for the test data. 0 implies 0 to 10000 will be tested and left out of training.
    //cout<<"Features: "<<featSize<<" Max Depth: "<<maxDepth<<endl;
    set<string> stopWords;

    //Get stopwords from file
    ifstream myfile("stopwords.csv");
    //ifstream myfile("/home/ubuntu/DecTree/stopwords.csv");
    //ifstream myfile("E:/Workspace/AML/DecisionTree/stopwords.csv");
    string word;
    while( getline( myfile, word) ){
        stopWords.insert(word);
    }
    myfile.close();
    myfile.clear();

    char line[50000];
    string gettheline;
    char abst[50000];
    string id;
    string abstract;
    string category;
    map<string, long long int> dictionary;
    int cs = 0, math = 0, stat = 0, physics = 0;

    //Find all unique words and their frequency
    int iteration = -1;
    myfile.open(path);
    getline(myfile, gettheline);
    while( getline(myfile, gettheline) ){
        iteration++;
        if( iteration >= testStart && iteration < testStart + testSize ) continue;
        if( iteration == trainSize ) break;
        gettheline.copy( line, gettheline.length() );
        line[gettheline.length()] = '\0';
        int len = strlen(line);
        id = "", category = "", abstract = "";
        int ptr = 0;
        while( line[ptr] != ',' ){
            id.append(&line[ptr], 1);
            ptr++;
        }
        ptr++;
        while( line[ptr] != ',' ){
            category.append(&line[ptr], 1);
            ptr++;
        }
        ptr += 2;
        while( line[ptr] != '\0' ){
            abstract.append(&line[ptr], 1);
            ptr++;
        }
        abstract = abstract.substr(0, abstract.size() - 1);
        if(category == "category") continue;

        for( i = 0; i <= abstract.size(); i++ )
           abst[i] = abstract[i];

        char* pch = strtok( abst, " ,.()/[]");
        while (pch != NULL){
          for( i = 0; i < strlen(pch); i++ ) pch[i] = tolower(pch[i]);
          if( stopWords.find(pch) == stopWords.end() && strlen(pch) > 3 ){
              dictionary[pch]++;
          }
          pch = strtok (NULL, " ,.()/[]");
        }
        //if( iteration % 10000 == 0 ) cout<<iteration<<" ";
    }
    //cout<<endl;
    myfile.close();
    myfile.clear();

    //Sort words according to highest frequency
    map<string, long long int>::iterator it;
    int cnt = 0;
    for( it = dictionary.begin(); it != dictionary.end(); it++ ){
        if( it->second < 20 ) continue;
        sortedDict[cnt].word = it->first;
        sortedDict[cnt].count = it->second;
        cnt++;
    }
    sort( sortedDict, sortedDict + cnt, cmp );

    //Create the training matrix
    ulint(*train)[featDim] = new ulint[trainSize + 100][featDim];
    for( i = 0; i < trainSize; i++ ) for( j = 0; j < featDim; j++ ) train[i][j] = 0;

    iteration = -1;
    myfile.open(path);
    getline(myfile, gettheline);
    while( getline(myfile, gettheline) ){
        iteration++;
        if( iteration >= testStart && iteration < testStart + testSize ) continue;
        gettheline.copy( line, gettheline.length() );
        line[gettheline.length()] = '\0';
        int len = strlen(line);
        id = "", category = "", abstract = "";
        int ptr = 0;
        while( line[ptr] != ',' ){
            id.append(&line[ptr], 1);
            ptr++;
        }
        ptr++;
        while( line[ptr] != ',' ){
            category.append(&line[ptr], 1);
            ptr++;
        }
        ptr += 2;
        while( line[ptr] != '\0' ){
            abstract.append(&line[ptr], 1);
            ptr++;
        }
        abstract = abstract.substr(0, abstract.size() - 1);
        if(category == "category") continue;

        if( category == "cs" ) train[iteration][0] = 0;
        else if( category == "math" ) train[iteration][0] = 1;
        else if( category == "stat" ) train[iteration][0] = 2;
        else if( category == "physics" ) train[iteration][0] = 3;

        std::transform(abstract.begin(), abstract.end(), abstract.begin(), ::tolower);
        for( i = 0; i < featSize; i++ ){
            if( abstract.find(sortedDict[i].word) != string::npos ){
                train[iteration][1 + (i/64)] |= ((ulint)1<<(i % 64));
            }
        }
        //if( iteration % 10000 == 0 ) cout<<iteration<<" ";
    }
    //cout<<endl;
    myfile.close();
    myfile.clear();
    ulint feat[featDim - 1] = {0};
    treeNode* decisionTree = new treeNode();

    //printf("Training...");
    //printf("on %d data points and %d words\n", trainSize, cnt);

    makeDecisionTree( decisionTree, train, iteration, feat, 0);
    delete(train);

   /* printTree( decisionTree, 0 );
    for( i = 0; i < 5; i++ ){
        for( j = 0; j < displayTree[i].size() && j < 15; j++ )
            cout<<displayTree[i][j]<<" ";
        printf("\n");
    }*/

    /////////////////////////////////////////////////////////
    //USE Decision Tree
    string ab;
    iteration = 0;
    int successful = 0;
    //printf("Testing...");
    //printf("on %d data points\n", testSize);
    iteration = -1;
    myfile.open(path);
    getline(myfile, gettheline);
    while( getline(myfile, gettheline) ){
        iteration++;
        if( iteration == trainSize || iteration == testStart + testSize ) break;
        if( iteration < testStart ) continue;
        gettheline.copy( line, gettheline.length() );
        line[gettheline.length()] = '\0';
        int len = strlen(line);
        id = "", category = "", abstract = "";
        int ptr = 0;
        while( line[ptr] != ',' ){
            id.append(&line[ptr], 1);
            ptr++;
        }
        ptr++;
        while( line[ptr] != ',' ){
            category.append(&line[ptr], 1);
            ptr++;
        }
        ptr += 2;
        while( line[ptr] != '\0' ){
            abstract.append(&line[ptr], 1);
            ptr++;
        }
        abstract = abstract.substr(0, abstract.size() - 1);
        if(category == "category") continue;

        std::transform(abstract.begin(), abstract.end(), abstract.begin(), ::tolower);
        int result = categorize(abstract, decisionTree);
        if( catName[result] == category ){
            successful++;
        }
        //if( iteration % 10000 == 0 ) cout<<iteration<<" ";
    }
    //cout<<endl;
    myfile.close();
    myfile.clear();

    cout<<featSize<<","<<maxDepth<<","<<((float)successful/testSize)*100<<endl;
    //cout<<((float)successful/testSize)*100<<endl;
    return 0;
}
