#include<fstream>
#include<iterator>
#include<iostream>
#include<sstream>
#include<string>
#include<vector>
using namespace std;

/*
  PREPROCESS HIGGS data
  28 features (22 low level feature, 5 high level feature derived from low
  level data + 1 particle mass)

  28 + 1 (padding for alfa in feed forwarding)
*/

const int col = 29, classNum = 2;
const double MAX_MASS = 1500;
template <typename Out>
void split(const std::string &s, char delim, Out result) {
    std::istringstream iss(s);
    std::string item;
    while (std::getline(iss, item, delim)) {
        *result++ = item;
    }
}

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    split(s, delim, std::back_inserter(elems));
    return elems;
}


void preprocess(string rawFileName, string xFileName, string yFileName){
	/*	We are going to read the raw file line by line
		and write each line (which means each row of data)
		into corresponding output file_x and file_y
	*/
	ifstream rawFile(rawFileName);
	ofstream xFile(xFileName, ios::trunc | ios::binary);
	ofstream yFile(yFileName, ios::trunc | ios::binary);
	string line;
  int r;
	for(r=1; rawFile >> line; r++){
    if (r<=2 || line.size()<=2){
      // SKIPPING HEADER
      continue;
    }
    float xArr[col];
    for (int c=0;c<col;c++){
      xArr[c]=0;
    }

    float yArr[classNum];
    for (int cn=0;cn<classNum;cn++){
      yArr[cn]=0;
    }

		vector<string> vec = split(line, ',');
    istringstream yStr(vec[0]);
    float yVal;
    yStr >> yVal;
    // cout << "YVAL :"<<yVal<<"\n";
    int yIdx = (int) yVal;
    yArr[yIdx]=1;
    // cout << yIdx <<": ";

    for(int idx=0;idx<col-1;idx++){
      istringstream xStr(vec[idx+1]);
      float xVal;
      xStr >> xVal;
      xArr[idx] = xVal;
      if (idx == col-2){
        // NORMALIZE MASS VALUE
        xArr[idx]/=MAX_MASS;
      }
      // cout << xArr[idx]<<"("<<vec[idx+1]<<") ";
    }
    // cout <<"\n";
    xArr[col-1]=1;


		// Write to output files
		yFile.write((char*) &yArr, classNum*sizeof(float));
    xFile.write((char*) &xArr, col*sizeof(float));
	}
	cout << "ROW = " << r-3 <<"\n";
	// rawFile.close();
	// xFile.close();
	// yFile.close();
}

int main(){
	ios_base::sync_with_stdio(false);
	// Preprocess training
	preprocess("raw/training", "training/file_x.bin", "training/file_y.bin");
  preprocess("raw/test", "test/file_x.bin", "test/file_y.bin");
}
